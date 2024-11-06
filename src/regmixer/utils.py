from typing import List

from beaker import Beaker
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig
from olmo_core.utils import generate_uuid, prepare_cli_environment
from regmixer.synthesize_mixture import get_mixes
from regmixer.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)
from pathlib import Path
import yaml

def mk_source_instances(sources: list[SourceConfig],mix_map) -> list[SourceInstance]:

    return [
        SourceInstance(
            name=source.name,
            paths=source.paths,
            ratio=mix_map[source.name],
        )
        for source in sources
    ]


def mk_experiments(config: ExperimentConfig,mixes) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{idx:04}",
            sources=mk_source_instances(config.sources,mix),
        )
        for idx,mix in enumerate(mixes)
    ]


def mk_mixes(config:ExperimentConfig):

    return get_mixes(config.sources,config.variants)

def mk_experiment_group(config: ExperimentConfig) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""

    mixes = mk_mixes(config.sources,config.variants)

    experiments = mk_experiments(config,mixes)
    return ExperimentGroup(
        config=config,
        instances=experiments,
    )


def mk_instance_cmd(instance: ExperimentInstance, config: ExperimentConfig) -> List[str]:
    """Build a command for launching an experiment instance."""

    sources = []

    for source in instance.sources:
        paths = [f'"{path}"' for path in source.paths]
        source_str = f'-s ("{source.name}",[{",".join(paths)}],{source.ratio})'
        sources.append(source_str)

    return [
        "src/regmixer/train.py",
        "train",
        f"-n {instance.name}",
        f"-l {config.sequence_length}",
        f"-t {config.max_tokens}",
        f"-S {config.seed}",
        *sources,
    ]


def mk_launch_configs(group: ExperimentGroup) -> list[BeakerLaunchConfig]:
    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    """Build a beaker launch config from an experiment group."""
    group_uuid = generate_uuid()[:8]
    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}-{group_uuid}",
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment, group.config),
            clusters=group.config.clusters,
            num_nodes=group.config.nodes,
            num_gpus=group.config.gpus,
            shared_filesystem=group.config.shared_filesystem,
            nfs=group.config.shared_filesystem,
            allow_dirty=True,
            weka_buckets=group.config.weka,
            budget=group.config.budget or "ai2/oe-data",
            workspace=group.config.workspace,
            preemptible=group.config.preemptible,
            beaker_image="ai2-tylerm/olmo-core-regmixer",
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
                BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            ],
            setup_steps=[
                'git clone "$REPO_URL"',
                "conda shell.bash activate base",
                "cd regmixer",
                'git checkout "$GIT_REF"',
                "git submodule update --init --recursive",
                "pip install -e '.[all]' && pip install git+https://github.com/allenai/OLMo-core.git@main",
                "pip freeze",
                # Move AWS credentials from env to relevant files
                "mkdir -p ~/.aws",
                "printenv AWS_CONFIG > ~/.aws/config",
                "printenv AWS_CREDENTIALS > ~/.aws/credentials",
            ],
        )
        for experiment in group.instances
    ]


def mk_launch_group(group: ExperimentGroup) -> list[BeakerLaunchConfig]:
    """Build a launch group from an experiment group."""
    return mk_launch_configs(group)



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch RegMixer experiments")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration file"
    )
  
    args = parser.parse_args()
    if not args.config.exists():
        print(f"Configuration file not found: {args.config}")
        return 1
    with open(args.config, "r") as f:
        data = yaml.safe_load(f)

        mk_experiment_group(ExperimentConfig(**data))


if __name__ == "__main__":
    exit(main())
