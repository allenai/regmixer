import concurrent.futures
import logging
import random
from collections import defaultdict

import numpy as np
import s3fs
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size
from tqdm import tqdm

logger = logging.getLogger(__name__)


from regmixer.aliases import (
    ExperimentConfig,
    SourceConfig,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Temperature for the prior distribution, if your distribution is too skewed, you can use a temperature to smooth it
TEMP = 0.7

# The minimum and maximum strength for the dirichlet distribution.
# With a small value, the distribution will be more concentrated, and with a large value, the distribution will be more uniform.
MIN_STRENGH = 0.1
MAX_STRENGH = 5.0

# We first sample num_samples * SAMPLE_MULTIPLIER times, then randomly select some of them
SAMPLE_MULTIPLIER = 500
MAXIMUM_USAGE = 1
MINIMUM_WEIGHT = 2e-4


def generate_weights_dirichlet(
    prior_dist,
    minimum_weight: float,
    num_samples_out: int,
    temperature: float,
    enable_bound: bool = False,
):
    final_samples = []

    if enable_bound:
        # generate the bound for reject sampling
        weight_bound = []
        for i in range(len(prior_dist)):
            weight_bound.append([0.0, min(prior_dist[i] * MAXIMUM_USAGE, 1.0)])
    else:
        weight_bound = None

    # apply temperature
    if temperature < 1.0:
        prior_dist = prior_dist**TEMP
        prior_dist = prior_dist / np.sum(prior_dist)

    # combine reject sampling with dirichlet distribution
    for i in range(num_samples_out * SAMPLE_MULTIPLIER):
        if MIN_STRENGH == MAX_STRENGH:
            samples = np.random.dirichlet(prior_dist * MIN_STRENGH, 1)
        else:
            samples = []
            min_strength_log = np.log10(MIN_STRENGH)
            max_strength_log = np.log10(MAX_STRENGH)

            for strength in np.logspace(min_strength_log, max_strength_log, 1000):
                # add noise to the strength
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                samples.append(samples_per_strength)

            valid_samples = []
            if weight_bound is not None:
                # Filter samples to those that are within the bounds
                for sample in samples:
                    for j in range(len(sample[0])):
                        if not (weight_bound[j][0] <= sample[0][j] <= weight_bound[j][1]):
                            logger.info(
                                f"Sample {sample[0][j]} is outside of bounds for index {j}: "
                                f"({weight_bound[j][0]}, {weight_bound[j][1]})"
                            )
                            break
                    else:
                        valid_samples.append(sample)
            else:
                valid_samples = samples

            if not valid_samples:
                raise ValueError("No valid samples found for bounds!")

            samples = random.choice(valid_samples)

        # post normalization, set zero for any value less than minimum_number
        samples = np.where(samples < minimum_weight, 0, samples)
        # round samples into the same scale of minimum_number
        samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
        samples = np.round(samples / minimum_weight) * minimum_weight
        # add the samples to the final_samples
        final_samples.append(samples[0])

    if len(final_samples) < num_samples_out:
        raise ValueError(
            f"The number of samples '{len(final_samples)}' is less than the required number of samples '{num_samples_out: int}'!"
        )

    final_samples = sort_and_deduplicate(np.array(final_samples))

    selected_samples = random.sample(final_samples, num_samples_out)
    selected_samples = np.stack(selected_samples, axis=0)

    return selected_samples


def mk_mixtures(config: ExperimentConfig):
    num_samples = config.variants
    sources = config.sources
    prior_config = calculate_priors(sources, config.dtype)

    random.seed(config.seed)
    np.random.seed(config.seed)

    logger.info(f"Using seed: {config.seed}")
    logger.info("Source distribution:")
    logger.info(prior_config)

    prior_dist = []
    for _, v in prior_config.items():
        prior_dist.append(v)

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)
    train_weights = generate_weights_dirichlet(
        prior_dist=prior_dist,
        minimum_weight=MINIMUM_WEIGHT,
        num_samples_out=num_samples,
        temperature=TEMP,
    )

    weight_maps = []
    for weights in train_weights:
        weight_map = {}
        for key, value in zip(prior_config.keys(), weights):
            weight_map[key] = value
        weight_maps.append(weight_map)

    return weight_maps


def _bytes_to_tokens(num_bytes: int, dtype: NumpyDatasetDType) -> int:
    """
    Convert bytes to tokens based on the dtype.
    """
    npdtype = dtype.as_np_dtype()
    return num_bytes // npdtype(int(0)).itemsize


def _count_tokens_for_file(path: PathOrStr, dtype: NumpyDatasetDType) -> int:
    return _bytes_to_tokens(get_file_size(path), dtype)


def calculate_priors(
    source_configs: list[SourceConfig], dtype: NumpyDatasetDType
) -> dict[str, float]:
    fs = s3fs.S3FileSystem(anon=False)

    token_counts = defaultdict(int)
    # Count tokens in each source directory
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_to_source = {
            executor.submit(
                lambda sc: {
                    sc.name: sum(
                        executor.submit(
                            lambda path: sum(
                                _count_tokens_for_file(f"s3://{match}", dtype)
                                for match in fs.glob(path)
                            ),
                            path,
                        ).result()
                        for path in sc.paths
                    )
                },
                source_config,
            ): source_config
            for source_config in source_configs
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_source),
            total=len(future_to_source),
            desc="Counting source tokens",
        ):
            source_config = future_to_source[future]
            try:
                result = future.result()
                token_counts.update(result)
            except Exception as e:
                logger.info(f"Error processing {source_config.name}: {str(e)}")
                token_counts[source_config.name] = 0

    # Calculate relative sizes
    total_tokens = sum(token_counts.values())
    logger.info(f"Total tokens for config: {total_tokens:,}")
    if total_tokens == 0:
        logger.info(f"Error processing config, no tokens found")
        return {}

    relative_sizes = {path: count / total_tokens for path, count in token_counts.items()}

    return relative_sizes


def sort_and_deduplicate(data, threshold=1e-5):
    """
    Remove identical configs to avoid duplicated training.
    """
    arr = np.array(data)
    sorted_indices = np.lexsort(arr.T)
    sorted_arr = arr[sorted_indices]
    result = [sorted_arr[0]]

    for i in range(1, len(sorted_arr)):
        diff = np.sum(np.abs(sorted_arr[i] - result[-1]))
        if diff > threshold:
            result.append(sorted_arr[i])

    return result
