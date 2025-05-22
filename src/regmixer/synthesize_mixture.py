import concurrent.futures
import logging
import os
import pathlib
import random
from collections import defaultdict
from typing import Tuple, Optional
from copy import deepcopy

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
    nonzero_weight: Optional[list[str]] = None
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
        prior_dist_temp = prior_dist**temperature
        prior_dist_temp = prior_dist_temp / np.sum(prior_dist_temp)
        logger.info(f"Prior distribution after temperature scaling: {prior_dist_temp}")
    else:
        prior_dist_temp = prior_dist

    if not allow_repetition and weight_bounds:
        logger.info("Limiting candidates to within bounds, repetition is disabled...")

    for _ in range(num_samples_out * ConfigDefaults.sample_multiplier):
        candidates = []
        if min_strength == max_strength:
            candidates.append(np.random.dirichlet(prior_dist_temp * min_strength, 1))
        else:
            min_strength_log = np.log10(min_strength)
            max_strength_log = np.log10(max_strength)
            for strength in np.logspace(min_strength_log, max_strength_log, 15):
                samples_per_strength = np.random.dirichlet(prior_dist_temp * strength, 1)
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

        if nonzero_weight:
            nonzero_domains = [domains.index(d) for d in nonzero_weight]
            filtered_candidates = [sample for sample in filtered_candidates if all(sample[0][idx] > minimum_weight for idx in nonzero_domains)]

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

    logger.info(f"Number of nonzero domains per swarm run: ")
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


def clip_candidates_by_level(candidates, idx_to_level, minimum_source_weight, minimum_topic_weight):
    assert len(candidates[0]) == len(idx_to_level), f"Length mismatch: {len(candidates)} vs {len(idx_to_level)}"
    clipped = np.array(candidates)  # defensive copy

    for i, level in enumerate(idx_to_level):
        if level == "source" and clipped[0][i] < minimum_source_weight:
            clipped[0][i] = 0.0
        elif level == "topic" and clipped[0][i] < minimum_topic_weight:
            clipped[0][i] = 0.0

    total = clipped.sum()
    if total > 0:
        clipped = clipped / total
    else:
        raise ValueError("All weights were clipped to zero.")

    return clipped

def sample_has_required_sources(sample_vector, domains, nonzero_sources, minimum_source_weight):
    # Compute source-level sums
    source_sums = defaultdict(float)

    for idx, weight in enumerate(sample_vector):
        domain = domains[idx]
        if ':' in domain:
            source = domain.split(':', 1)[0]
        else:
            source = domain
        source_sums[source] += weight

    return all(source_sums[source] > minimum_source_weight for source in nonzero_sources)




def generate_weights_dirichlet_hierarchical(
    sources: list[SourceConfig], # flat 
    source_dist: dict[str, float],
    minimum_source_weight: float,
    minimum_topic_weight: float,
    num_samples_out: int,
    source_temperature: float,
    topic_temperature: float,
    min_source_strength: float, 
    max_source_strength: float,
    min_topic_strength: float,
    max_topic_strength: float,
    max_tokens: int,
    source_tokens: int,
    allow_repetition: bool,
    enable_bound: bool = True,
    nonzero_weight: Optional[list[str]] = None
):
    """
    Generate weights for each domain group using a dirichlet distribution.
    """

    token_scale = source_tokens / max_tokens
    logger.info(f"Source token population is {token_scale:.2f}:1 target population.")

    collected_samples: list[Tuple[np.ndarray, np.ndarray]] = []
    weight_bounds = None


    prior_dist = np.array([v for _, v in source_dist.items()])
    domains = [k for k, _ in source_dist.items()]
    source_names = [source.name for source in sources]
    idx_to_level = ["source" if name in source_names else "topic" for name in source_dist]

    if enable_bound:
        # weight bounds are at the leaf level.
        weight_bounds = [
            (0.0, min(prior_dist[idx] * token_scale, 1.0)) for idx in range(len(prior_dist))
        ]
        grouped_bounds = {domain: weight_bounds[idx] for idx, domain in enumerate(domains)}
        logger.info("Weight bounds:")
        logger.info(grouped_bounds)


    # split prior distribution into source and topic distributions 
    topic_distributions = {}
    source_distribution = []
    for source_config in sorted(sources, key=lambda x: x.name):
        if source_config.topics:
            weights = np.array([source_dist[f"{source_config.name}:{topic.name}"] for topic in source_config.topics])
            normalized_weights = weights / weights.sum()
            topic_distributions[source_config.name] = normalized_weights 
            source_distribution.append(weights.sum())
        else:
            topic_distributions[source_config.name] = np.array([1.0])
            source_distribution.append(source_dist[source_config.name])

    source_distribution = np.array(source_distribution)
    source_distribution /= source_distribution.sum()


    if source_temperature < 1.0:
        source_prior = source_distribution**source_temperature
        source_prior = source_prior / np.sum(source_prior)
        logger.info(f"Source prior distribution after temperature scaling: {source_prior}")
    else:
        source_prior = source_distribution

    if topic_temperature < 1.0:
        topic_priors = deepcopy(topic_distributions)
        for source, topic_prior in topic_priors.items():
            topic_prior = topic_prior**topic_temperature
            topic_prior = topic_prior / np.sum(topic_prior)
            topic_priors[source] = topic_prior
    else:
        topic_priors = deepcopy(topic_distributions)

    if not allow_repetition and weight_bounds:
        logger.info("Limiting candidates to within bounds, repetition is disabled...")

    breakpoint()
    for _ in range(num_samples_out * ConfigDefaults.sample_multiplier):
        candidates = []

        # first, generate source-level weights

        if min_source_strength == max_source_strength:
            source_samples = np.random.dirichlet(source_prior * min_source_strength, 1)
        else:
            min_source_strength_log = np.log10(min_source_strength)
            max_source_strength_log = np.log10(max_source_strength)
            source_samples = []
            for strength in np.logspace(min_source_strength_log, max_source_strength_log, 15):
                samples_per_strength = np.random.dirichlet(source_prior * strength, 1)
                source_samples.append(samples_per_strength)


        # then, generate topic-level weights
        topic_samples = defaultdict(list)
        for source, topic_prior in topic_priors.items():
            if min_topic_strength == max_topic_strength:
                topic_samples[source].append(np.random.dirichlet(topic_prior * min_topic_strength, 1))
            else:
                min_topic_strength_log = np.log10(min_topic_strength)
                max_topic_strength_log = np.log10(max_topic_strength)
                for strength in np.logspace(min_topic_strength_log, max_topic_strength_log, 15):
                    samples_per_strength = np.random.dirichlet(topic_prior * strength, 1)
                    topic_samples[source].append(samples_per_strength)

        # convert from source_samples and topic_samples back to a set of leaf-level samples 
        candidates = []
        for i, source_sample in enumerate(source_samples):
            leaf_level_sample = {source: samples[i][0]*source_sample[0, j] for j, (source, samples) in enumerate(topic_samples.items())}
            flattened_sample = np.concatenate([arr for arr in list(leaf_level_sample.values())]).reshape(1, -1)
            candidates.append(flattened_sample)

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

        if nonzero_weight:
            source_names = set(nonzero_weight)
            # Filter candidates
            filtered_candidates = [
                sample for sample in filtered_candidates
                if sample_has_required_sources(sample[0], domains, source_names, minimum_source_weight)
            ]

        if not filtered_candidates:
            continue

        candidates = random.choice(filtered_candidates)


        if minimum_source_weight == minimum_topic_weight:
            candidates = np.where(candidates < minimum_source_weight, 0, candidates)
            candidates = candidates / np.sum(candidates).reshape(-1, 1)
            candidates = np.round(candidates / minimum_source_weight) * minimum_source_weight
            candidates = candidates / np.sum(candidates)
        else:
            candidates = clip_candidates_by_level(
                candidates,
                idx_to_level,
                minimum_source_weight,
                minimum_topic_weight
            )

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

    logger.info(f"Number of nonzero domains per swarm run: ")
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

    # convert single-level sampling params into topic/source level
    minimum_source_weight = config.minimum_source_weight if config.minimum_source_weight else config.minimum_weight if config.minimum_weight else ConfigDefaults.minimum_weight
    minimum_topic_weight = config.minimum_topic_weight if config.minimum_topic_weight else config.minimum_weight if config.minimum_weight else ConfigDefaults.minimum_weight

    source_mix_temperature = config.source_mix_temperature if config.source_mix_temperature else config.mix_temperature
    topic_mix_temperature = config.topic_mix_temperature if config.topic_mix_temperature else config.mix_temperature

    min_source_strength = config.min_source_strength if config.min_source_strength else config.min_strength
    max_source_strength = config.max_source_strength if config.max_source_strength else config.max_strength

    min_topic_strength = config.min_topic_strength if config.min_topic_strength else config.min_strength
    max_topic_strength = config.max_topic_strength if config.max_topic_strength else config.max_strength

    mixtures = generate_weights_dirichlet_hierarchical(
        sources=sources,
        source_dist=source_dist,
        minimum_source_weight=minimum_source_weight,
        minimum_topic_weight=minimum_topic_weight,
        num_samples_out=num_samples,
        source_temperature=source_mix_temperature,
        topic_temperature=topic_mix_temperature,
        min_source_strength=min_source_strength,
        max_source_strength=max_source_strength,
        min_topic_strength=min_topic_strength,
        max_topic_strength=max_topic_strength,
        allow_repetition=config.allow_repetition,
        max_tokens=config.max_tokens,
        source_tokens=source_total,
        enable_bound=True,
        nonzero_weight=config.nonzero_weight,
    )

    """mixtures = generate_weights_dirichlet(
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
        nonzero_weight=config.nonzero_weight,
    )"""

    weight_maps = []
    for mix in mixtures:
        weight_map = {}
        for idx in range(len(domains)):
            weight_map[domains[idx]] = (mix[0][idx], mix[1][idx])

        weight_maps.append(weight_map)

    for i in range(len(domains)):
        weights = np.array([mix[0][i] for mix in mixtures])
        logger.info(f"Domain {domains[i]}, min: {weights.min()}, max: {weights.max()}")

    return weight_maps


def _bytes_to_tokens(num_bytes: int, dtype: NumpyDatasetDType) -> int:
    """
    Convert bytes to tokens based on the dtype.
    """
    npdtype = dtype.as_np_dtype()
    return num_bytes // npdtype(int(0)).itemsize


def _count_tokens_for_file(path: PathOrStr, dtype: NumpyDatasetDType) -> int:
    return _bytes_to_tokens(get_file_size(path), dtype)


def count_tokens(paths: list[str], dtype: NumpyDatasetDType, fs) -> int:
    """Helper to count tokens across a list of paths using glob expansion."""
    total = 0
    for path in paths:
        matches = fs.glob(path)
        for match in matches:
            total += _count_tokens_for_file(f"s3://{match}", dtype)
    return total


def get_leaf_configs(source_config):
    """Return a list of (name, paths) tuples representing the leaf nodes."""
    if source_config.topics:
        return [
            (f"{source_config.name}:{topic.name}", topic.paths)
            for topic in source_config.topics
        ]
    else:
        return [(source_config.name, source_config.paths)]


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
    # Count tokens in each "leaf": the prior distribution is represented at the leaf level.
    leaf_configs = []
    for source_config in source_configs:
        leaf_configs.extend(get_leaf_configs(source_config))

    # Multithreaded token counting at leaf level
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        future_to_leaf = {
            executor.submit(count_tokens, paths, dtype, fs): name
            for name, paths in leaf_configs
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_leaf),
            total=len(future_to_leaf),
            desc="Counting tokens (leaf level)",
        ):
            name = future_to_leaf[future]
            try:
                count = future.result()
                token_counts[name] = count
            except Exception as e:
                logger.info(f"Error processing {name}: {str(e)}, exiting!")
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
