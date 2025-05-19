"""
    rmc-convert convert -s output/0cd83153/avg_mmlu_bpb_ntest_50_seed_1_optimal.json -d seed_1_regmix_optimal
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
@click.option("-r", "--reference-file", type=str, default="src/regmixer/internal/config/natural-larger-sample.yaml", help="Path to reference file (yaml containing source mixture, input to rmc-internal)")
def convert(
    partition_type: str,
    source_file: str,
    dest_file: str,
    reference_file: Optional[str],
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

    base_config['sources'] = [
        source for source in base_config['sources']
        if source['domain'] in source_dict
    ]

    for source in base_config['sources']:
        source['weight'] = source_dict[source['domain']]

    if os.path.exists(f"src/regmixer/internal/config/{dest_file}.yaml"):
        logger.warning(f"Destination file {dest_file}.yaml already exists. Skipping.")
    else:
        with open(f"src/regmixer/internal/config/{dest_file}.yaml", "w") as f:
            yaml.dump(base_config, f, sort_keys=False)

        logger.info(f"Successfully converted mixture at {source_file} to source yaml at {dest_file}")


if __name__ == "__main__":
    cli()
