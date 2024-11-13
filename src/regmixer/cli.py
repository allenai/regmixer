import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Optional

import click
import yaml
from beaker import Beaker
from beaker.services.job import JobClient
from olmo_core.utils import generate_uuid, prepare_cli_environment

from regmixer.aliases import ExperimentConfig, LaunchGroup
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.synthesize_mixture import mk_mixtures
from regmixer.utils import mk_experiment_group, mk_instance_cmd, mk_launch_configs

logger = logging.getLogger(__name__)


def config_from_path(config: Path) -> ExperimentConfig:
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


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
    "-m",
    "--mixture-file",
    help="(Optional) Relative path to a mixture configuration file.",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
def launch(config: Path, mixture_file: Optional[Path], dry_run: bool):
    """Launch an experiment."""

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)
    group_uuid = generate_uuid()[:8]

    if mixture_file:
        with open(mixture_file, "r") as f:
            predefined_mixes = json.load(f)
        launch_group = LaunchGroup(
            instances=mk_launch_configs(
                mk_experiment_group(
                    experiment_config, mixes=predefined_mixes["mixes"], group_uuid=group_uuid
                )
            )
        )
    else:
        mixes = _generate_mixes(config)

        if click.confirm("Launch experiment with this set of mixtures?", default=False):
            launch_group = LaunchGroup(
                instances=mk_launch_configs(
                    mk_experiment_group(experiment_config, mixes=mixes, group_uuid=group_uuid)
                )
            )
        else:
            logger.info("Launch cancelled")
            return

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
        logger.info(f"Experiment group '{group_uuid}' launched successfully!")
    except KeyboardInterrupt:
        logger.warning(
            "\nAborting experiment group launch! You may need to manually stop the launched experiments."
        )


def prettify_mixes(mixes: list[dict[str, float]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def _generate_mixes(config_file: Path, output: Optional[Path] = None):
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)

    config = ExperimentConfig(**data)
    mixes = mk_mixtures(config)
    mix_string = prettify_mixes(mixes)

    if not output:
        output = Path(f"/tmp/regmixer/{config.name}_{generate_uuid()[:6]}.json")

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)

        with open(output, "w") as f:
            f.write(mix_string)
        logger.info(f"Mixes saved to {output}:")
    logger.info(mix_string)

    return mixes


def status_for_group(path: Path, group_id: str):
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]
    statuses.sort(key=lambda x: x["display_name"])
    logger.info(statuses)


def stop_for_group(path: Path, group_id: str):
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]

    if len(jobs) == 0:
        logger.info(f"No jobs found for group {group_id}")
        return

    jobs.sort(key=lambda x: x["display_name"])
    logger.info("Jobs to cancel:")
    logger.info(jobs)
    if click.confirm("Cancel these jobs?", default=False):
        for job in jobs:
            logger.info(f"Stopping job {job['display_name']}...")
            client.stop(job["id"])


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path for the generated mixes (defaults to generated_mix.json if not specified)",
)
def generate_mixes(config: Path, output: Optional[Path] = None):
    """Generate a set of mixtures based on a provided config"""
    _generate_mixes(config, output)


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

    mixes = _generate_mixes(config)
    experiment_group = mk_experiment_group(ExperimentConfig(**data), mixes, generate_uuid()[:8])

    for experiment in experiment_group.instances:
        logger.info(mk_instance_cmd(experiment, experiment_group.config, experiment_group.group_id))
        transformer = TransformerConfigBuilder(
            cluster=experiment_group.config.cluster,
            beaker_user="validate-no-op",
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
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def status(config: Path, group_id: str):
    """Get the status of a launched experiment group."""

    status_for_group(config, group_id)


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def cancel(config: Path, group_id: str):
    """Cancel all running jobs for an experiment group."""

    stop_for_group(config, group_id)


if __name__ == "__main__":
    cli()
