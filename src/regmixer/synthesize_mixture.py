import numpy as np
import random
import yaml
import argparse
import os
import s3fs

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
#   is used because the natura trade off between the reweighting v.s. the number of avaiable tokens in each domain.
#   Usually we think repeating 4 epochs is okay for language model pre-training, and here we set it as 15
#   because the avaiable token of The Pile is much larger than the token amount for training Chinchilla-Optimal 1B models (i.e., 25B tokens).
#   However, if you want to train the large-scale model with all avaiable tokens, you can use less than 4 epochs also in the proxy
#   model training.
MAXIMUM_USAGE = 15

# Assume that we have 1B (512,000 examples, and 2048 tokens per example) tokens
#   for the proxy model training, the minimum sampling rate 2e-4 indicates that
#   at least there will be 100 examples for each domain, which is statistically significant.
#
# If you use less tokens for training the proxy models, you may increase the minimum sampling rate
#   to ensure the statistical significance of the domain. I personally recommend using at least 1e-5
#   if you have 1B tokens for training the proxy models.
MINIMUM = 2e-4  

def generate_train_group(groups, weights, precision=5):
    """
    Generate a formatted string of groups and their corresponding weights.

    Args:
    groups (list): List of group names.
    weights (list): List of corresponding weights.
    sample_folder (str, optional): If provided, will be included in the group name.
    prefix (str, optional): Prefix to add before each group name. Defaults to 'train'.
    precision (int, optional): Number of decimal places for rounding weights. Defaults to 4.

    Returns:
    str: Formatted string of groups and weights.
    """
    assert len(groups) == len(weights), "Length of groups and weights must be equal"
    
    def format_weight(weight):
        return f"{weight:.{precision}f}".rstrip('0').rstrip('.')
    
    output_group = [f"  {group}: {format_weight(num)}" 
                    for group, num in zip(groups, weights)]
    
    return "\n".join(output_group)

def generate_valid_group(groups):
    weights = [1.0] * len(groups)
    output_group = [f"  {group}: {num}" for group, num in zip(groups, weights)]
    return "\n".join(output_group)


def generate_weights_dirichlet(prior_dist,
                               train_groups,
                               minimum_number,
                               num_samples=128, 
                               enable_bound=True,
                               temperature=1.0):

    final_samples = []
    
    if enable_bound:
        # generate the bound for reject sampling
        number_bound = []
        for i in range(len(prior_dist)):
            # the token cannot be used more than 4 times
            number_bound.append([0.0,
                                 min(prior_dist[i] * MAXIMUM_USAGE, 1.0)])
    else:
        number_bound = None
        
    # apply temperature
    if temperature < 1.0:
        prior_dist = prior_dist ** TEMP
        prior_dist = prior_dist / np.sum(prior_dist)

    print("\n\nThe domain usage bound (maximum domain weight): ")
    # print the bound for each group
    for i in range(len(prior_dist)):
        print(f"{train_groups[i]}: {number_bound[i][1]}")

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
        samples = np.where(samples < minimum_number, 0.0, samples)
        # round samples into the same scale of minimum_number
        samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
        samples = np.round(samples / minimum_number) * minimum_number
        # add the samples to the final_samples
        final_samples.append(samples[0])

    # remove the samples with the nearly same values
    print("\nThe number of avaiable samples: ", len(final_samples))
    # deduplicate the samples
    final_samples = sort_and_deduplicate(np.array(final_samples))
    # remove the samples with the nearly same values
    print("The number of deduplicated samples: ", len(final_samples))
    selected_samples = random.sample(final_samples, num_samples)
    print("The number of selected samples: ", len(selected_samples))
    selected_samples = np.stack(selected_samples, axis=0)
    return selected_samples



def generate_distributions_from_prior(num_samples, prior_config):
    # read the yaml file and get the prior distribution
    train_config = prior_config["train"]
    train_groups, prior_dist = [], []
    for k, v in train_config.items():
        train_groups.append(k)
        prior_dist.append(v)

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)
    valid_config = prior_config["valid"]
    valid_groups = list(valid_config.keys())
    
    train_weights = generate_weights_dirichlet(prior_dist, 
                                               train_groups,
                                               MINIMUM,
                                               num_samples,
                                               temperature=TEMP)
    
    weight_maps = []
    for weights in train_weights:
        weight_map = {}
        c = 0 
        for key in train_groups:
            weight_map[key.replace("valid_","")] = weights[c]
            c+=1 
        weight_maps.append(weight_map)

    return weight_maps   
    



def generate_config_from_prior(output_paths, prior_config):
    number_of_samples = len(output_paths)
    # read the yaml file and get the prior distribution
    train_config = prior_config["train"]
    train_groups, prior_dist = [], []
    for k, v in train_config.items():
        train_groups.append(k)
        prior_dist.append(v)

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)
    valid_config = prior_config["valid"]
    valid_groups = list(valid_config.keys())
    
    train_weights = generate_weights_dirichlet(prior_dist, 
                                               train_groups,
                                               MINIMUM,
                                               number_of_samples,
                                               temperature=TEMP)
    
    return train_weights   
    
def calculate_priors(source_configs:list[SourceConfig]):


    fs = s3fs.S3FileSystem(anon=False)
    import boto3
    from collections import defaultdict
    from urllib.parse import urlparse
    
    s3 = boto3.client('s3')
    file_counts = defaultdict(int)
    
    def parse_s3_path(s3_path):
        """Parse s3://bucket/prefix format into bucket and prefix."""
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path format: {s3_path}. Must start with 's3://'")
        
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        # Remove leading slash and ensure trailing slash for prefix
        prefix = parsed.path.lstrip('/')
        if prefix and not prefix.endswith('/'):
            prefix += '/'
            
        return bucket, prefix
    
    # Count files in each folder
    for source_config in source_configs:
        try:           
            files = []
            for path in source_config.paths:
                matches = fs.glob(path)
                files.extend([f for f in matches if fs.isfile(f)])

            file_counts[source_config.name] = len(files)
                
        except Exception as e:
            print(f"Error processing {source_config.name}: {str(e)}")
            file_counts[source_config.name] = 0
    
    # Calculate relative sizes
    total_files = sum(file_counts.values())
    if total_files == 0:
        return {path: 0 for path in folder_paths}
    
    relative_sizes = {
        path: count / total_files 
        for path, count in file_counts.items()
    }
    train = {f"valid_{key}":value for key,value in relative_sizes.items()}

    
    valid = {f"valid_{key}":1.0 for key in relative_sizes.keys()}

    return {"train": train, "valid": valid}



def get_mixes(sources:list[SourceConfig],num_samples: int):
    output_folder = "config_1m"
    output_paths = []
    for i in range(1, num_samples + 1):
        output_paths.append(f"{output_folder}/n{i}.yaml")

    prior_configs = calculate_priors(sources)
    return generate_distributions_from_prior(num_samples,prior_config=prior_configs)


def sort_and_deduplicate(data, threshold=1e-5):
    """
    Remove identify configs to avoid duplicated training.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    
    parser.add_argument("--output_folder", type=str, default="config_1m")
    parser.add_argument("--num_configs", type=int, default=512)
    
    args = parser.parse_args()
    output_folder = args.output_folder
    num_samples = args.num_configs
    
    # if not exist, create the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_paths = []
    for i in range(1, num_samples + 1):
        output_paths.append(f"{output_folder}/n{i}.yaml")

    keys = ["finance","health","entertainment"]

    domain_info= {"health":"s3://ai2-llm/pretraining-data/sources/ds-olmo-data/dclm_partitioned_v0.2/high0/domains_v3.8_tokens/Health/*.npy",
    "finance":"s3://ai2-llm/pretraining-data/sources/ds-olmo-data/dclm_partitioned_v0.2/high0/domains_v3.8_tokens/Finance_n_Business/*.npy",
    "entertainment":"s3://ai2-llm/pretraining-data/sources/ds-olmo-data/dclm_partitioned_v0.2/high0/domains_v3.8_tokens/Entertainment/*.npy"}
    generate_config_from_prior(output_paths,
                               prior_config=calculate_priors(domain_info))
    