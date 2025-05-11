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
    maximum_repetition: int = 5
    minimum_weight: float = 2e-3  # 0.002


def generate_weights_dirichlet(
    domains: list[str],
    prior_dist: np.ndarray,
    minimum_weight: float,
    num_samples_out: int,
    temperature: float,
    min_strength: float, 
    max_strength: float,
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
        logger.info(f"Prior distribution after temperature scaling: {prior_dist}")

    if not allow_repetition and weight_bounds:
        logger.info("Limiting candidates to within bounds, repetition is disabled...")

    for _ in range(num_samples_out * ConfigDefaults.sample_multiplier):
        candidates = []
        if min_strength == max_strength:
            candidates.append(np.random.dirichlet(prior_dist * min_strength, 1))
        else:
            min_strength_log = np.log10(min_strength)
            max_strength_log = np.log10(max_strength)
            for strength in np.logspace(min_strength_log, max_strength_log, 15):
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                candidates.append(samples_per_strength)

        filtered_candidates = []

        # If we don't allow repetition, we need to filter out candidates that are outside the bounds
        if weight_bounds and not allow_repetition:
            filtered_candidates = [
                sample
                for sample in candidates
                if all(
                    lower <= sample[0][idx] <= upper
                    for idx, (lower, upper) in enumerate(weight_bounds)
                )
            ]
        else:
            filtered_candidates = candidates

        if not filtered_candidates:
            continue

        candidates = random.choice(filtered_candidates)
        candidates = np.where(candidates < minimum_weight, 0, candidates)
        candidates = candidates / np.sum(candidates).reshape(-1, 1)
        candidates = np.round(candidates / minimum_weight) * minimum_weight
        candidates = candidates / np.sum(candidates)

        if weight_bounds and not allow_repetition:
            # need to check for out-of-bounds candidates again, in case normalization caused bounds to be violated.
            if any(candidates[0][idx] < lower or candidates[0][idx] > upper for idx, (lower, upper), in enumerate(weight_bounds)):
                continue

        selected: Tuple[np.ndarray, np.ndarray] = (
            candidates[0],
            np.ones(candidates.shape[1]),
        )

        reject = False

        if allow_repetition:
            for idx, _ in enumerate(domains):
                available_tokens = int(prior_dist[idx] * source_tokens)
                required_tokens = int(selected[0][idx] * max_tokens)

                repetition = (
                    np.ceil(required_tokens / available_tokens * 1000) / 1000
                    if available_tokens != 0
                    else 0
                )

                if repetition > ConfigDefaults.maximum_repetition:
                    reject = True
                    break

                selected[1][idx] = max(1, repetition)

        if not reject:
            collected_samples.append(selected)

    if len(collected_samples) == 0:
        raise ValueError("No valid samples were generated, please check the configuration!")

    deduped = sort_and_deduplicate(collected_samples)

    if len(collected_samples) < num_samples_out:
        raise ValueError(
            f"The number of collected samples '{len(collected_samples)}' is less than the required number of samples '{num_samples_out}'!"
        )

    selected_samples = random.sample(deduped, num_samples_out)
    selected_samples = np.stack(selected_samples, axis=0)

    print([len(np.where(selected_samples[i][0] != 0)[0]) for i in range(len(selected_samples))])

    all_diffs = []
    for i in range(len(selected_samples)):
        for j in range(i + 1, len(selected_samples)):
            diff = np.linalg.norm(selected_samples[i][0] - selected_samples[j][0])
            if diff < 0.01:
                logger.info(f"Sample {i} and Sample {j} are too close to each other!")
                logger.info(f"Sample {i}: {selected_samples[i][0]}")
                logger.info(f"Sample {j}: {selected_samples[j][0]}")
            all_diffs.append(diff)
            
    return selected_samples


def mk_mixtures(
    config: ExperimentConfig, use_cache: bool = True
) -> list[dict[str, Tuple[float, float]]]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    num_samples = config.variants
    sources = config.sources
    source_dist, source_total, source_tokens = calculate_priors(
        sources, config.dtype, use_cache=use_cache
    )

    logger.info(f"Total tokens for config: {source_total:,}")
    logger.info(f"Using seed: {config.seed}")

    logger.info("Source distribution:")
    logger.info(source_dist)
    logger.info("Source tokens:")

    source_tokens = {k: f"{v:,}" for k, v in source_tokens.items() if v > 0}
    logger.info(source_tokens)

    source_items = list(source_dist.items())
    prior_dist = [v for _, v in source_items]
    domains = [k for k, _ in source_items]

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)

    mixtures = generate_weights_dirichlet(
        domains=domains,
        prior_dist=prior_dist,
        minimum_weight=config.minimum_weight or ConfigDefaults.minimum_weight,
        num_samples_out=num_samples,
        temperature=config.mix_temperature,
        min_strength=config.min_strength,
        max_strength=config.max_strength,
        allow_repetition=config.allow_repetition,
        max_tokens=config.max_tokens,
        source_tokens=source_total,
    )

    weight_maps = []
    for mix in mixtures:
        weight_map = {}
        for idx in range(len(domains)):
            weight_map[domains[idx]] = (mix[0][idx], mix[1][idx])

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
) -> Tuple[dict[str, float], int, dict[str, int]]:
    config_hash = hashlib.md5(
        json.dumps(
            [(sc.name, sc.paths) for sc in source_configs],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    pathlib.Path("cache/").mkdir(parents=True, exist_ok=True)
    cache_path = pathlib.Path(f"cache/priors_cache_{config_hash}.json")
    if use_cache:
        try:
            with open(cache_path, "r") as f:
                logger.info(
                    "Source distribution cache found, using cached values! This can be disabled by setting use_cache=False."
                )
                obj = json.load(f)
                return (obj["relative_sizes"], obj["total_tokens"], obj["token_counts"])
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
                logger.info(f"Error processing {source_config.name}: {str(e)}, exiting!")
                raise e

    # Calculate relative sizes
    total_tokens = sum(token_counts.values())

    token_counts = dict(sorted(token_counts.items()))

    if total_tokens == 0:
        raise Exception(f"Error processing config, no tokens found for sources!")

    relative_sizes = {path: count / total_tokens for path, count in token_counts.items()}

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "relative_sizes": relative_sizes,
                    "total_tokens": total_tokens,
                    "token_counts": token_counts,
                },
                f,
            )

    return (relative_sizes, total_tokens, token_counts)


def sort_and_deduplicate(
    samples: list[Tuple[np.ndarray, np.ndarray]], threshold=1e-5
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Remove identical configs to avoid duplicated training.
    """
    unique_samples = []
    for sample in samples:
        is_duplicate = any(
            np.allclose(sample[0], unique_sample[0], atol=threshold)
            for unique_sample in unique_samples
        )
        if not is_duplicate:
            unique_samples.append(sample)

    logger.info(
        f"Filtered {len(samples) - len(unique_samples)} duplicate distributions from candidate pool..."
    )
    return unique_samples
