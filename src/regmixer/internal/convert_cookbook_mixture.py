"""
    rmc-convert-cookbook convert -s output/62e7dc06/avg_mmlu_bpb_ntest_50_seed_1_optimal.json -d seed_1_regmix_optimal
"""

import click
import json
import os
import subprocess
import sys
from typing import Optional
import boto3
import yaml 
import logging 
from olmo_core.utils import prepare_cli_environment
logger = logging.getLogger(__name__)


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option("-t", "--partition-type", type=str, default="topic", help="Whether partitioned by topic or format")
@click.option("-s", "--source-file", required=True, type=str, help="Path to source file (json containing regmix weights)")
@click.option("-d", "--dest-file", required=True, type=str, help="Path to destination file (yaml containing source mixture, input to rmc-internal)")
@click.option("-c", "--cookbook-path", required=False, type=str, help="Path to cookbook directory", default="../olmo-cookbook/")
@click.option("-r", "--reference-file", type=str, default="../olmo-cookbook/src/cookbook/recipes/train-1b-v2-5xC-dclm-larger-natural.yaml", help="Path to reference file (yaml containing source mixture, input to rmc-internal)")
def convert(
    partition_type: str,
    source_file: str,
    dest_file: str,
    cookbook_path: str,
    reference_file: str, 
):

    if partition_type == "topic":
        with open(reference_file, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        raise NotImplementedError("Only topic partitioning is supported at this time")

    sources = json.load(open(source_file, "r"))
    source_dict = {}
    for source in sources:
        source_dict[source['domain']] = source['weight']

    for source in base_config['dataset']['sources']:
        source['target_ratio'] = source_dict[source['name']]


    if os.path.exists(f"{cookbook_path}/src/cookbook/recipes/{dest_file}.yaml"):
        logger.warning(f"Destination file {dest_file}.yaml already exists. Skipping.")
    else:
        with open(f"{cookbook_path}/src/cookbook/recipes/{dest_file}.yaml", "w") as f:
            yaml.dump(base_config, f, sort_keys=False)

        logger.info(f"Successfully converted mixture at {source_file} to source yaml at {dest_file}")


if __name__ == "__main__":
    cli()
