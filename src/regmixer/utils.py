import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
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
from regmixer.synthesize_mixture import mk_mixtures

logger = logging.getLogger(__name__)


def config_from_path(config: Path) -> ExperimentConfig:
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


def mk_source_instances(
    sources: list[SourceConfig], mix_map: dict[str, float]
) -> list[SourceInstance]:
    # Note: We filter out any sources that have a weight of 0
    filtered_sources = [source for source in sources if mix_map[source.name] > 0]
    return [
        SourceInstance(
            name=source.name,
            paths=source.paths,
            ratio=mix_map[source.name],
        )
        for source in filtered_sources
    ]


def mk_experiments(
    config: ExperimentConfig, mixes: list[dict[str, float]], group_uuid: str
) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{group_uuid}-{idx:04}",
            sources=mk_source_instances(config.sources, mix),
        )
        for idx, mix in enumerate(mixes)
    ]


def mk_experiment_group(
    config: ExperimentConfig, mixes: list[dict[str, float]], group_uuid: str
) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""

    return ExperimentGroup(
        config=config,
        group_id=group_uuid,
        instances=mk_experiments(config, mixes, group_uuid),
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
        f"-d {config.dtype.value}",
        f"-T {config.tokenizer}",
        f"-m {config.proxy_model_id}",
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
            beaker_image="ai2-tylerm/olmo-core-nightly-20241127132540",
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


def prettify_mixes(mixes: list[dict[str, float]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def mk_mixes(
    config_file: Path, output: Optional[Path] = None, use_cache: bool = True
) -> list[dict[str, float]]:
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)

    config = ExperimentConfig(**data)
    mixes = mk_mixtures(config, use_cache=use_cache)
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
