import concurrent.futures
import logging
import os
import pathlib
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import s3fs
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)


import hashlib
import json

from regmixer.aliases import ExperimentConfig, SourceConfig


class ConfigDefaults:
    temp: float = 1.0
    min_strength: float = 0.1
    max_strength: float = 5.0
    sample_multiplier: int = 10
    maximum_repetition: int = 1
    minimum_weight: float = 2e-3  # 0.002


def generate_weights_dirichlet(
    domains: list[str],
    prior_dist: np.ndarray,
    minimum_weight: float,
    num_samples_out: int,
    temperature: float,
    max_tokens: int,
    source_tokens: int,
    allow_repetition: bool,
    enable_bound: bool = True,
):
    """
    Generate weights for each domain group using a dirichlet distribution.
    """

    token_scale = source_tokens / max_tokens
    logger.info(f"Source token population is {token_scale:.2f}:1 target population.")

    collected_samples: list[Tuple[np.ndarray, np.ndarray]] = []
    weight_bounds = None

    if enable_bound:
        weight_bounds = [
            (0.0, min(prior_dist[idx] * token_scale, 1.0)) for idx in range(len(prior_dist))
        ]
        grouped_bounds = {domain: weight_bounds[idx] for idx, domain in enumerate(domains)}
        logger.info("Weight bounds:")
        logger.info(grouped_bounds)

    if temperature < 1.0:
        prior_dist = prior_dist**temperature
        prior_dist = prior_dist / np.sum(prior_dist)

    for _ in range(num_samples_out * ConfigDefaults.sample_multiplier):
        candidates = []
        if ConfigDefaults.min_strength == ConfigDefaults.max_strength:
            candidates.append(np.random.dirichlet(prior_dist * ConfigDefaults.min_strength, 1))
        else:
            min_strength_log = np.log10(ConfigDefaults.min_strength)
            max_strength_log = np.log10(ConfigDefaults.max_strength)

            for strength in np.logspace(min_strength_log, max_strength_log, 15):
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                candidates.append(samples_per_strength)

        filtered_candidates = []

        if weight_bounds and not allow_repetition:
            # Check that all domains in the sample are within bounds otherwise reject
            logger.info("Limiting candidates to within bounds, repetition is disabled...")
            filtered_candidates = [
                sample
                for sample in candidates
                if all(
                    lower <= sample[0][idx] <= upper
                    for idx, (lower, upper) in enumerate(weight_bounds)
                )
            ]
            filtered_candidates = candidates
        else:
            filtered_candidates = candidates

        if not filtered_candidates:
            continue

        candidates = random.choice(filtered_candidates)
        candidates = np.where(candidates < minimum_weight, 0, candidates)
        candidates = candidates / np.sum(candidates).reshape(-1, 1)
        candidates = np.round(candidates / minimum_weight) * minimum_weight
        candidates = candidates / np.sum(candidates)

        selected: Tuple[np.ndarray, np.ndarray] = (
            candidates[0],
            np.ones(candidates.shape[1]),
        )

        if allow_repetition:
            for idx, _ in enumerate(prior_dist):
                available_tokens = int(prior_dist[idx] * source_tokens)
                required_tokens = int(selected[0][idx] * max_tokens)

                # Don't divide by zero
                if required_tokens == 0:
                    continue

                repetition = required_tokens / available_tokens
                selected[1][idx] = max(1, repetition)

        collected_samples.append(selected)

    deduped = sort_and_deduplicate(collected_samples)

    if len(collected_samples) < num_samples_out:
        raise ValueError(
            f"The number of collected samples '{len(collected_samples)}' is less than the required number of samples '{num_samples_out}'!"
        )

    selected_samples = random.sample(deduped, num_samples_out)
    selected_samples = np.stack(selected_samples, axis=0)

    return selected_samples


def mk_mixtures(config: ExperimentConfig, use_cache: bool = True):
    random.seed(config.seed)
    np.random.seed(config.seed)

    num_samples = config.variants
    sources = config.sources
    source_dist, source_total = calculate_priors(sources, config.dtype, use_cache=use_cache)

    logger.info(f"Total tokens for config: {source_total:,}")
    logger.info(f"Using seed: {config.seed}")
    logger.info("Source distribution:")
    logger.info(source_dist)

    prior_dist = [v for _, v in source_dist.items()]

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)
    weights = generate_weights_dirichlet(
        domains=list(source_dist.keys()),
        prior_dist=prior_dist,
        minimum_weight=ConfigDefaults.minimum_weight,
        num_samples_out=num_samples,
        temperature=config.mix_temperature,
        allow_repetition=config.allow_repetition,
        max_tokens=config.max_tokens,
        source_tokens=source_total,
    )

    weight_maps = []
    for result in weights:
        weight_map = {}
        domains = list(source_dist.keys())
        for name, weight, repetition in zip(domains, result[0], result[1]):
            weight_map[name] = (weight, repetition)

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
    source_configs: list[SourceConfig], dtype: NumpyDatasetDType, use_cache: bool
) -> Tuple[dict[str, float], int]:
    config_hash = hashlib.md5(
        json.dumps(
            [(sc.name, sc.paths) for sc in source_configs],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    cache_path = pathlib.Path(f"/tmp/regmixer/priors_cache_{config_hash}.json")
    if use_cache:
        try:
            with open(cache_path, "r") as f:
                logger.info(
                    "Source distribution cache found, using cached values! This can be disabled by setting use_cache=False."
                )
                obj = json.load(f)
                return (obj["relative_sizes"], obj["total_tokens"])
        except FileNotFoundError:
            logger.info("No cache file found, calculating from source files...")

    fs = s3fs.S3FileSystem(anon=False)

    token_counts = defaultdict(int)
    # Count tokens in each source directory
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
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

    if total_tokens == 0:
        raise Exception(f"Error processing config, no tokens found!")

    relative_sizes = {path: count / total_tokens for path, count in token_counts.items()}

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"relative_sizes": relative_sizes, "total_tokens": total_tokens}, f)

    return (relative_sizes, total_tokens)


def sort_and_deduplicate(
    samples: list[Tuple[np.ndarray, np.ndarray]], threshold=1e-5
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Remove identical configs to avoid duplicated training.
    """
    arr = np.array([sample[0] for sample in samples])
    sorted_indices = np.lexsort(arr.T)
    sorted_arr = arr[sorted_indices]
    result = [sorted_arr[0]]

    for i in range(1, len(sorted_arr)):
        diff = np.sum(np.abs(sorted_arr[i] - result[-1]))

        if diff > threshold:
            result.append(sorted_arr[i])

    return [(res, samples[sorted_indices[i]][1]) for i, res in enumerate(result)]
