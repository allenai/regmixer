from typing import List

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
            ratio=1 / len(sources),
        )
        for source in sources
    ]


def mk_experiments(config: ExperimentConfig, group_uuid: str) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{idx:04}-{group_uuid}",
            sources=mk_source_instances(config.sources),
        )
        for idx in range(config.variants)
    ]


def mk_experiment_group(config: ExperimentConfig) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""
    group_uuid = generate_uuid()[:8]
    return ExperimentGroup(
        config=config,
        group_id=group_uuid,
        instances=mk_experiments(config, group_uuid),
    )


def mk_instance_cmd(
    instance: ExperimentInstance, config: ExperimentConfig, group_id: str
) -> List[str]:
    """Build a command for launching an experiment instance."""

    beaker_user = (Beaker.from_env().account.whoami().name).lower()
    sources = []

    for source in instance.sources:
        paths = [f'"{path}"' for path in source.paths]
        source_str = f'-s ("{source.name}",[{",".join(paths)}],{source.ratio})'
        sources.append(source_str)

    return [
        "src/regmixer/train.py",
        "train",
        f"-n {instance.name}",
        f"-g {group_id}",
        f"-l {config.sequence_length}",
        f"-t {config.max_tokens}",
        f"-S {config.seed}",
        f"-c {config.cluster}",
        f"-u {beaker_user}",
        *sources,
    ]


def mk_launch_configs(group: ExperimentGroup) -> list[BeakerLaunchConfig]:
    """Build a beaker launch config from an experiment group."""

    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}",
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment, group.config, group.group_id),
            clusters=[group.config.cluster],
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
            priority=group.config.priority,
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
