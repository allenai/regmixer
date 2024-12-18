"""
Launch an OLMo-core training run with the specified configuration.
"""

import logging

import click
import yaml
from beaker import Priority
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.utils import generate_uuid, prepare_cli_environment

from regmixer.aliases import ExperimentConfig, LaunchGroup, SourceConfig
from regmixer.utils import mk_experiment_group, mk_launch_configs

logger = logging.getLogger(__name__)


def build_config():
    pass


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "-n",
    "--name",
    type=str,
    required=True,
    help="Experiment name",
)
@click.option(
    "-t",
    "--tokenizer",
    type=str,
    required=True,
    help="Tokenizer to use",
)
@click.option(
    "-c",
    "--cluster",
    type=str,
    required=True,
    help="Beaker cluster",
)
@click.option(
    "-w",
    "--workspace",
    type=str,
    required=True,
    help="Experiment workspace",
)
@click.option(
    "-b",
    "--budget",
    type=str,
    required=True,
    help="Experiment budget",
)
@click.option(
    "-d",
    "--description",
    type=str,
    default="",
    help="Experiment description",
)
@click.option(
    "-N",
    "--nodes",
    type=int,
    default=1,
    help="Number of nodes to use",
)
@click.option(
    "-g",
    "--gpus",
    type=int,
    default=1,
    help="Number of GPUs per node",
)
@click.option(
    "-i",
    "--model-identifier",
    type=str,
    default="olmo_1b",
    help="See regmixer/model/aliases.py for available model identifiers.",
)
@click.option(
    "-m",
    "--max-tokens",
    type=int,
    required=True,
    help="Max tokens to train on",
)
@click.option(
    "-l",
    "--sequence-length",
    type=int,
    required=True,
    help="Sequence length for the dataset",
)
@click.option(
    "-s",
    "--sources-file",
    type=str,
    required=True,
    help="Name of the source config file in regmixer.internal.config",
)
@click.option(
    "-D",
    "--dtype",
    type=str,
    default="uint32",
    help="Dtype for the token dataset",
)
@click.option(
    "-S",
    "--seed",
    type=int,
    default=42,
    help="Random seed for the experiment",
)
@click.option(
    "-p",
    "--priority",
    type=Priority,
    default=Priority.low,
    help="Beaker experiment priority level",
)
@click.option(
    "-r",
    "--max-repetition",
    type=float,
    default=5.0,
    help="Max repetition factor for sources (applies to all sources)",
)
def train(
    name: str,
    priority: Priority,
    cluster: str,
    tokenizer: str,
    workspace: str,
    budget: str,
    description: str,
    nodes: int,
    gpus: int,
    max_tokens: int,
    sequence_length: int,
    sources_file: str,
    model_identifier: str,
    seed: int,
    dtype: str,
):
    """Launch training run with the specified configuration properties."""

    with open(f"src/regmixer/internal/config/{sources_file}", "r") as f:
        config = yaml.safe_load(f)

    sources = [
        SourceConfig(
            name=source["domain"],
            paths=source["paths"],
            max_repetition_factor=source.get("max_repetition_factor", 5.0),
            max_source_ratio=source.get("max_source_ratio", 1.0),
        )
        for source in config["sources"]
    ]

    weights = {}

    for source in config["sources"]:
        weights[source["domain"]] = source["weight"]

    group_uuid = generate_uuid()[:8]
    experiment_config = ExperimentConfig(
        name=f"olmo-1b-{name}",
        description=description,
        budget=budget,
        workspace=workspace,
        # Always launch a single variant
        variants=1,
        nodes=nodes,
        gpus=gpus,
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        sources=sources,
        seed=seed,
        cluster=cluster,
        tokenizer=tokenizer,
        priority=priority,
        proxy_model_id=model_identifier,
        dtype=NumpyDatasetDType[dtype],
    )

    logger.info(experiment_config)
    logger.info(weights)

    if click.confirm("Launch experiment with this config and weights?", default=False):
        launch_group = LaunchGroup(
            instances=mk_launch_configs(
                mk_experiment_group(experiment_config, mixes=[weights], group_uuid=group_uuid)
            )
        )
        launch_group.instances[0].launch()
    else:
        logger.info("Launch cancelled!")
        return


if __name__ == "__main__":
    cli()

    """
    Example usage:
    rmc-internal train \
        -n regmixer-avg-bpb-weights \
        -t dolma2 \
        -c ai2/saturn-cirrascale \
        -w ai2/dolma2 \
        -b ai2/oe-data \
        -m 600_000_000 \
        -l 2048 \
        -p high \
        -g 8 \
        -N 1 \
        -d uint16 \
        -r 5.0 \
        -s avg_mmlu_bpb_alpha_7_0.yaml
    """
