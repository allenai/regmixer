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
    project: str = "regmixer"
    entity: str = "ai2-llm"


@dataclass
class TransformerConfigBuilder:
    run_name: str
    overrides: List[str]
    sources: List[SourceInstance]
    sequence_length: int
    max_tokens: int
    tokenizer_config: TokenizerConfig
    seed: int = 42

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
                    save_async=False,  # TODO: Figure out how to make this work, maybe hardware specific?
                ),
            )
            # TODO: Add the correct WANDB config so that this works
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
                        paths=[
                            "s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy"
                        ],
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

        # defaults?

        #         c4_en-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy
        # dolma_books-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_books/val/part-0-00000.npy
        # dolma_common-crawl-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_common-crawl/val/part-0-00000.npy
        # dolma_pes2o-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_pes2o/val/part-0-00000.npy
        # dolma_reddit-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_reddit/val/part-0-00000.npy
        # dolma_stack-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_stack/val/part-0-00000.npy
        # dolma_wiki-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_wiki/val/part-0-00000.npy
        # ice-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/ice/val/part-0-00000.npy
        # m2d2_s2orc-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/m2d2_s2orc/val/part-0-00000.npy
        # pile-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/pile/val/part-0-00000.npy
        # wikitext_103-validation:
        #   - s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/wikitext_103/val/part-0-00000.npy

        return ModelTrainConfig(
            model=model_config,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
        )
