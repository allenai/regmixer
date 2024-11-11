import argparse
import logging
import os
import random
from collections import defaultdict
from urllib.parse import urlparse

import boto3
import numpy as np
import s3fs
import torch
import yaml
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size

logger = logging.getLogger(__name__)


from regmixer.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Temperature for the prior distribution, if your distribution is too skewed, you can use a temperature to smooth it
TEMP = 0.5

# The minimum and maximum strength for the dirichlet distribution.
# With a small value, the distribution will be more concentrated, and with a large value, the distribution will be more uniform.
MIN_STRENGH = 0.1
MAX_STRENGH = 5.0

# We first sample SAMPLE_MULTIPLIER times more samples than randomly select some of them
SAMPLE_MULTIPLIER = 100

# How many epochs are allowed for each domain for the large-scale model training. This hyper-parameter
#   is used because the natural trade off between the reweighting v.s. the number of avaiable tokens in each domain.
#   Usually we think repeating 4 epochs is okay for language model pre-training, and here we set it as 15
#   because the avaiable token of The Pile is much larger than the token amount for training Chinchilla-Optimal 1B models (i.e., 25B tokens).
#   However, if you want to train the large-scale model with all avaiable tokens, you can use less than 4 epochs also in the proxy
#   model training.
MAXIMUM_USAGE = 2

# Assume that we have 1B (512,000 examples, and 2048 tokens per example) tokens
#   for the proxy model training, the minimum sampling rate 2e-4 indicates that
#   at least there will be 100 examples for each domain, which is statistically significant.
#
# If you use less tokens for training the proxy models, you may increase the minimum sampling rate
#   to ensure the statistical significance of the domain. I personally recommend using at least 1e-5
#   if you have 1B tokens for training the proxy models.
MINIMUM = 2e-4


def generate_weights_dirichlet(
    prior_dist, train_groups, minimum_number, num_samples=128, enable_bound=True, temperature=1.0
):
    final_samples = []

    if enable_bound:
        # generate the bound for reject sampling
        number_bound = []
        for i in range(len(prior_dist)):
            # the token cannot be used more than 4 times
            number_bound.append([0.0, min(prior_dist[i] * MAXIMUM_USAGE, 1.0)])
    else:
        number_bound = None

    # apply temperature
    if temperature < 1.0:
        prior_dist = prior_dist**TEMP
        prior_dist = prior_dist / np.sum(prior_dist)

    # combine reject sampling with dirichlet distribution
    for i in range(num_samples * SAMPLE_MULTIPLIER):
        if MIN_STRENGH == MAX_STRENGH:
            samples = np.random.dirichlet(prior_dist * MIN_STRENGH, 1)
        else:
            samples = []
            min_strength_log = np.log10(MIN_STRENGH)
            max_strength_log = np.log10(MAX_STRENGH)
            for strength in np.logspace(min_strength_log, max_strength_log, 15):
                # add a noise to the strength
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                samples.append(samples_per_strength)
            # random sample one
            samples = random.choice(samples)
        # if there is a bound, the bound is a list of tuples indicating the lower and upper bound of each group
        ensure_flag = True
        if number_bound is not None:
            for j in range(len(samples[0])):
                if samples[0][j] < number_bound[j][0] or samples[0][j] > number_bound[j][1]:
                    ensure_flag = False
                    break
        if ensure_flag is False:
            continue
        # post normalization, set zero for the number less than minimum_number
        samples = np.where(samples < minimum_number, 0, samples)
        # round samples into the same scale of minimum_number
        samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
        samples = np.round(samples / minimum_number) * minimum_number
        # add the samples to the final_samples
        final_samples.append(samples[0])

    final_samples = sort_and_deduplicate(np.array(final_samples))
    selected_samples = random.sample(final_samples, num_samples)
    selected_samples = np.stack(selected_samples, axis=0)
    return selected_samples


def mk_mixtures(config: ExperimentConfig):
    num_samples = config.variants
    sources = config.sources
    prior_config = calculate_priors(sources)

    logger.info("Prior Distribution:")
    logger.info("\n".join([f"{key} : {value}" for key, value in prior_config.items()]))

    train_groups, prior_dist = [], []
    for k, v in prior_config.items():
        train_groups.append(k)
        prior_dist.append(v)

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)

    train_weights = generate_weights_dirichlet(
        prior_dist, train_groups, MINIMUM, num_samples, temperature=TEMP
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


def _count_tokens_for_file(path: PathOrStr) -> int:
    return _bytes_to_tokens(get_file_size(path), NumpyDatasetDType.uint8)


def calculate_priors(source_configs: list[SourceConfig]):
    fs = s3fs.S3FileSystem(anon=False)

    token_counts = defaultdict(int)

    # Count tokens in each folder
    for source_config in source_configs:
        try:
            token_count = 0
            for path in source_config.paths:
                matches = fs.glob(path)
                for match in matches:
                    token_count += _count_tokens_for_file(f"s3://{match}")

            token_counts[source_config.name] = token_count

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
