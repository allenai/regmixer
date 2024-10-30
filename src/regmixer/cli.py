from pathlib import Path
import concurrent.futures

import click
import yaml

from regmixer.aliases import ExperimentConfig, LaunchGroup
from regmixer.utils import mk_experiment_group, mk_launch_configs


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Follow the experiment logs.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment configuration without launching.",
)
def launch(config: Path, follow: bool, dry_run: bool):
    """Launch an experiment."""
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)
    launch_group = LaunchGroup(instances=mk_launch_configs(mk_experiment_group(experiment_config)))

    print("Launching experiment group...")
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    experiment.launch() if not dry_run else experiment.build_experiment_spec(),
                    follow,
                )
                for experiment in launch_group.instances
            ]
        results = [future.result() for future in futures]
        print(results)
        print("Experiment group launched successfully!")
    except KeyboardInterrupt:
        print("\nCancelling experiment group...")
        # TODO: Try to cancel the experiments in the group


@cli.command()
def status():
    """Get the status of an experiment."""
    raise NotImplementedError


@cli.command()
def stop():
    """Stop an experiment."""
    raise NotImplementedError


@cli.command()
def list():
    """List all experiments."""
    raise NotImplementedError


if __name__ == "__main__":
    cli()
