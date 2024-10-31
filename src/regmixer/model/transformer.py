from dataclasses import dataclass
from typing import List

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
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

from regmixer.data.dataset import MixtureBuilder
from regmixer.aliases import SourceInstance


@dataclass
class ModelTrainConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


@dataclass
class TransformerConfigBuilder:
    run_name: str
    overrides: List[str]
    sources: List[SourceInstance]
    sequence_length: int
    max_tokens: int
    seed: int = 42
    tokenizer_config: TokenizerConfig = TokenizerConfig.gpt2()

    def build(self) -> ModelTrainConfig:
        # TODO: Make this configurable?
        model_config = TransformerConfig.llama2_271M(
            vocab_size=self.tokenizer_config.padded_vocab_size(),
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

        mixture_config = MixtureBuilder(
            sources=self.sources,
            max_tokens=self.max_tokens,
            sequence_length=self.sequence_length,
            seed=self.seed,
            processes=1,
            dtype=NumpyDatasetDType.uint16,
        ).build()

        dataset_config = NumpyDatasetConfig(
            source_mixture_config=mixture_config,
            name=NumpyDatasetType.fsl,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer_config,
            work_dir="/tmp/dataset-cache",
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=256 * self.sequence_length,
            seed=0,
            num_workers=4,
        )

        trainer_config = (
            TrainerConfig(
                save_folder=f"/tmp/{self.run_name}",
                rank_microbatch_size=16 * self.sequence_length,
                save_overwrite=True,
                metrics_collect_interval=5,
                cancel_check_interval=5,
            )
            .with_callback(
                "lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100))
            )
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
                "wandb",
                WandBCallback(
                    name=self.run_name,
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
                        sequence_length=self.sequence_length,
                        tokenizer=self.tokenizer_config,
                        work_dir="/tmp/dataset-cache",
                    ),
                    eval_interval=250,
                    eval_duration=Duration.steps(10),
                ),
            )
        )

        return ModelTrainConfig(
            model=model_config,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
        ).merge(self.overrides)
