from pathlib import Path
from typing import Any, Union
from os import PathLike
from dataclasses import dataclass
from typing import List, cast

import click
import s3fs
from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureConfig, SourceMixtureDatasetConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import init_hybrid_shard_mesh
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all

from regmixer.aliases import SourceInstance

path_type = Union[Path, PathLike[Any], str]


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--max-tokens",
    "-t",
    type=int,
    help="Max tokens for the mixture dataset",
    required=True,
)
@cli.command()
@click.option(
    "--source",
    "-s",
    multiple=True,
    type=(str, str, str),
    help="Source datasets in the form `name path,path,... ratio`",
)
@click.option(
    "--run-name",
    "-n",
    type=str,
    help="Name of the run",
    required=True,
)
def train(run_name, max_tokens, source):
    sources = []
    for item in source:
        name, paths, ratio = item
        paths = paths.split(",")
        sources.append(SourceInstance(name=name, paths=paths, ratio=float(ratio)))

    run(build_config(run_name, [], sources, max_tokens))


def expand_globs(paths: List[str], fs: s3fs.S3FileSystem) -> List[path_type]:
    expanded_paths: List[path_type] = []
    for path in paths:
        new = [Path(f"s3://{obj}") for obj in fs.glob(path)]
        expanded_paths.extend(new)

    return expanded_paths


def build_source_mixtures(
    source_instances: List[dict], fs: s3fs.S3FileSystem
) -> List[SourceMixtureConfig]:
    source_mixtures: List[SourceMixtureConfig] = []
    for source_instance in source_instances:
        # TODO: Match these keys so we aren't messing with them
        name = source_instance["name"]
        paths = source_instance["paths"]
        ratio = source_instance["ratio"]
        source_mixtures.append(
            SourceMixtureConfig(source_name=name, paths=expand_globs(paths, fs), target_ratio=ratio)
        )

    return source_mixtures


def build_config(
    run_name: str, overrides: List[str], source_instances: List[dict], max_tokens: int
) -> ExperimentConfig:
    fs = s3fs.S3FileSystem()

    # TODO: Pass these from the experiment config
    sequence_length = 1024
    seed = 42
    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.llama2_271M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    source_configs = build_source_mixtures(source_instances, fs)
    source_mixture_config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint16,
        processes=1,
        seed=seed,
    )

    # TODO: Make this the mixture dataset
    dataset_config = NumpyDatasetConfig(
        source_mixture_config=source_mixture_config,
        name=NumpyDatasetType.fsl,
        sequence_length=sequence_length,
        tokenizer=tokenizer_config,
        work_dir="/tmp/dataset-cache",
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * sequence_length,
        seed=0,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            rank_microbatch_size=16 * sequence_length,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(
            "seq_len_scheduler",
            SequenceLengthSchedulerCallback(
                min_sequence_length=128, warmup_steps=100, enabled=False
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        # TODO: Add the right evaluators here
        .with_callback(
            "evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig(
                    paths=["/net/nfs/allennlp/llm-data/c4/en/c4-validation.00000-00008.npy"],
                    metadata=[{"label": "c4-validation"}],
                    name=NumpyDatasetType.padded_fsl,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir="/tmp/dataset-cache",
                ),
                eval_interval=250,
                eval_duration=Duration.steps(10),
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    ).merge(overrides)


def run(config: ExperimentConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=get_default_device(),
        dp_mesh=init_hybrid_shard_mesh(num_replicas=2),
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset)
    trainer = config.trainer.build(model, optim, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    cli()
    try:
        prepare_training_environment()
    finally:
        teardown_training_environment()
