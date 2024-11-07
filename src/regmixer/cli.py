import concurrent.futures
import logging
from pathlib import Path

import click
import yaml
import json

from olmo_core.utils import prepare_cli_environment
from olmo_core.data import TokenizerConfig
from typing import Optional

from regmixer.aliases import ExperimentConfig, LaunchGroup
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.utils import mk_experiment_group, mk_launch_configs, mk_instance_cmd,mk_mixes

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
    "-m",
    "--mixture-file",
    help="(Optional) Relative path to a mixture configuration file.",
    type=click.Path(exists=True),
    required=False
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
def launch(config: Path, mixture_file: Path, dry_run: bool):
    """Launch an experiment."""

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)

    if mixture_file:
        with open(mixture_file, "r") as f:
            predefined_mixes = json.load(f)
        launch_group = LaunchGroup(instances=mk_launch_configs(
            mk_experiment_group(experiment_config, mixes=predefined_mixes)
        ))
    else:
        output_path = "generated_mix.json"
        logger.info(f"Mixture definition file not provided. Generating one now at {output_path}:")
        mixes = _generate_mixes(config,Path(output_path))
    
        if click.confirm("Launch experiment with this set of mixtures?",default=False):
            launch_group = LaunchGroup(instances=mk_launch_configs(
                mk_experiment_group(experiment_config, mixes=mixes)
            ))
    
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
        logger.info("Experiment group launched successfully!")
    except KeyboardInterrupt:
        logger.warning("\nCancelling experiment group...")
        # TODO: Try to cancel the experiments in the group


def prettify_mixes(mixes):
    mixes = {"mixes":mixes}
    return json.dumps(mixes,indent=2)




def _generate_mixes(config: Path, output:Optional[Path]=None):
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    mixes = mk_mixes(ExperimentConfig(**data))
    mix_string = prettify_mixes(mixes)
  
    if output:
        with open(output, "w") as f:
            f.write(mix_string)
        logger.info(f"Mixes saved to {output}:")
    logger.info(mix_string)
    return mixes


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
def generate_mixes(config: Path, output: Optional[Path] = Path("generated_mix.json")):
    """Generate a set of mixtures based on a provided config"""

    _generate_mixes(config, output)
    logger.info(f"\nMixtures saved to {output}")

  

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

    experiments = mk_experiment_group(ExperimentConfig(**data))
    tokenizer = TokenizerConfig.dolma2()

    for experiment in experiments.instances:
        logger.info(mk_instance_cmd(experiment, experiments.config))
        transformer = TransformerConfigBuilder(
            run_name="validate-no-op",
            max_tokens=experiments.config.max_tokens,
            sources=experiment.sources,
            overrides=[],
            sequence_length=experiments.config.sequence_length,
            seed=experiments.config.seed,
            tokenizer_config=tokenizer,
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
