from beaker import Beaker
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig
from olmo_core.utils import generate_uuid

from regmixer.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)


def mk_source_instances(sources: list[SourceConfig]) -> list[SourceInstance]:
    # TODO: Implement the randomized mixing ratios here
    return [
        SourceInstance(
            name=source.name,
            paths=source.paths,
            ratio=0.0,
        )
        for source in sources
    ]


def mk_experiments(config: ExperimentConfig) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{idx:04}",
            sources=mk_source_instances(config.sources),
        )
        for idx in range(config.variants)
    ]


def mk_experiment_group(config: ExperimentConfig) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""
    return ExperimentGroup(
        config=config,
        instances=mk_experiments(config),
    )


def mk_instance_cmd(instance: ExperimentInstance) -> list[str]:
    """Build a command for an experiment instance."""
    return [
        "python",
        "train.py",
        f"--name={instance.name}",
    ] + [
        f"--source={source.name} {','.join(source.paths)} {source.ratio}"
        for source in instance.sources
    ]


def mk_launch_configs(group: ExperimentGroup) -> list[BeakerLaunchConfig]:
    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    """Build a beaker launch config from an experiment group."""
    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}-{generate_uuid()[:8]}",  # TODO: Check if we need a UUID here
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment),
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
            env_secrets=[
                BeakerEnvSecret("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
                BeakerEnvSecret("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="COMET_API_KEY", secret=f"{beaker_user}_COMET_API_KEY"),
                BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
                BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            ],
            setup_steps=[
                # Clone repo.
                'git clone "$REPO_URL" .',
                'git checkout "$GIT_REF"',
                "git submodule update --init --recursive",
                # Setup python environment.
                "conda shell.bash activate base",
                "pip install -e '.[all]'",
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
