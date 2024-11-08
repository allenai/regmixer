import concurrent.futures
import logging
from pathlib import Path

import click
import yaml
from olmo_core.data import TokenizerConfig
from olmo_core.utils import prepare_cli_environment

from regmixer.aliases import ExperimentConfig, LaunchGroup
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.utils import mk_experiment_group, mk_instance_cmd, mk_launch_configs

logger = logging.getLogger(__name__)


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
def launch(config: Path, dry_run: bool):
    """Launch an experiment."""

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)
    launch_group = LaunchGroup(instances=mk_launch_configs(mk_experiment_group(experiment_config)))

    logger.info("Launching experiment group...")
    try:
        if dry_run:
            logger.info("Dry run mode enabled. Printing experiment configurations...")
            for experiment in launch_group.instances:
                logger.info(experiment.build_experiment_spec())
            return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(experiment.launch) for experiment in launch_group.instances]
        results = [future.result() for future in futures]
        logger.info(results)
        logger.info("Experiment group launched successfully!")
    except KeyboardInterrupt:
        logger.warning("\nCancelling experiment group...")
        # TODO: Try to cancel the experiments in the group


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def validate(config: Path):
    """Validate an experiment configuration."""
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_group = mk_experiment_group(ExperimentConfig(**data))

    for experiment in experiment_group.instances:
        logger.info(mk_instance_cmd(experiment, experiment_group.config, experiment_group.group_id))
        transformer = TransformerConfigBuilder(
            group_id="validate-no-op",
            run_name="validate-no-op",
            max_tokens=experiment_group.config.max_tokens,
            sources=experiment.sources,
            overrides=[],
            sequence_length=experiment_group.config.sequence_length,
            seed=experiment_group.config.seed,
        ).build()
        dataset = transformer.dataset.build()
        dataset.prepare()


@cli.command()
def status():
    """Get the status of an experiment group."""
    raise NotImplementedError


@cli.command()
def stop():
    """Stop an experiment group."""
    raise NotImplementedError


@cli.command()
def list():
    """List all experiment groups."""
    raise NotImplementedError


if __name__ == "__main__":
    cli()
