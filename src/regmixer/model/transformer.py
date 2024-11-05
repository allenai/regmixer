from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class ModelConfig:
    parameters: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    weight_tying: bool
    alibi: bool
    rope: bool
    rope_theta: int
    flash_attention: bool
    attention_dropout: float
    attention_layer_norm: bool
    include_bias: bool
    layer_norm_type: str
    layer_norm_with_affine: bool
    layer_norm_eps: float
    bias_for_layer_norm: bool
    attention_layer_norm_with_affine: bool
    activation_type: str
    residual_dropout: float
    embedding_dropout: float
    max_sequence_length: int
    vocab_size: int
    embedding_size: int
    eos_token_id: int
    pad_token_id: int
    init_device: str
    init_fn: str
    init_std: float
    init_cutoff_factor: int
    norm_after: bool
    precision: str


DEFAULT_MODEL_CONFIG = ModelConfig(
    parameters=190_354_176,
    d_model=768,
    n_heads=12,
    n_layers=12,
    mlp_ratio=8,
    weight_tying=False,
    alibi=False,
    rope=True,
    rope_theta=500000,
    flash_attention=True,
    attention_dropout=0.0,
    attention_layer_norm=True,
    include_bias=False,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1e-6,
    bias_for_layer_norm=False,
    attention_layer_norm_with_affine=True,
    activation_type="swiglu",
    residual_dropout=0.0,
    embedding_dropout=0.0,
    max_sequence_length=4096,
    vocab_size=100278,
    embedding_size=100352,
    eos_token_id=100257,
    pad_token_id=100277,
    init_device="cuda",
    init_fn="normal",
    init_std=0.02,
    init_cutoff_factor=3,
    norm_after=True,
    precision="amp_bf16",
)


@dataclass
class TransformerConfigBuilder:
    """
    A builder class for configuring and creating a transformer model training configuration.

    Attributes:
        run_name (str): The name of the run.
        overrides (List[str]): A list of override strings.
        sources (List[SourceInstance]): A list of source instances.
        sequence_length (int): The sequence length for the model.
        max_tokens (int): The maximum number of tokens.
        group_id (str): The group ID for the run.
        seed (int): The random seed for reproducibility. Default is 42.
        config (Optional[ModelConfig]): An optional model configuration. Default is None.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, seed, config):
            Initializes the TransformerConfigBuilder with the provided parameters.

        get_tokenizer_config() -> TokenizerConfig:
            Returns the tokenizer configuration.

        get_warmup_steps() -> int:
            Returns the number of warmup steps.

        get_batch_size():
            Returns the global batch size based on the sequence length and model parameters.

        build() -> ModelTrainConfig:
            Builds and returns the model training configuration.
    """

    run_name: str
    sources: List[SourceInstance]
    sequence_length: int
    max_tokens: int
    model_config: ModelConfig
    group_id: str
    seed: int = 42
    config: Optional[ModelConfig] = None
    overrides: Optional[List[str]] = None

    def __init__(
        self,
        run_name: str,
        sources: List[SourceInstance],
        sequence_length: int,
        max_tokens: int,
        group_id: str,
        seed: int,
        config: Optional[ModelConfig] = None,
        overrides: List[str] = [],
    ):
        self.run_name = run_name
        self.sources = sources
        self.sequence_length = sequence_length
        self.max_tokens = max_tokens
        self.group_id = group_id
        self.seed = seed
        self.config = config
        self.overrides = overrides
        self.model_config = DEFAULT_MODEL_CONFIG

        self._default_device_batch_size = 8
        self._default_betas = (0.9, 0.95)
        self._default_weight_decay = 0.1
        self._default_eps = 1e-8
        self._default_decay_embeddings = False
        self._default_vocab_size = 100278
        self._default_batch_size_divisor = 32
        self._default_global_batch_size = self.get_batch_size()
        self._default_device_batch_size = 8
        self._default_dataparallel_type = DataParallelType.ddp

    def get_tokenizer_config(self) -> TokenizerConfig:
        return TokenizerConfig(
            vocab_size=self.model_config.vocab_size,
            eos_token_id=self.model_config.eos_token_id,
            pad_token_id=self.model_config.pad_token_id,
        )

    def get_warmup_steps(self) -> int:
        return round(
            self.model_config.parameters
            / (self._default_global_batch_size * self.model_config.max_sequence_length)
        )

    def get_batch_size(self):
        if self.sequence_length != 2048:
            raise NotImplementedError("Only sequence length 2048 is supported right now")

        global_batch_size = 160 * (self.model_config.parameters / 108000000) ** (2 / 3)
        global_batch_size /= self._default_batch_size_divisor
        global_batch_size = round(global_batch_size)
        global_batch_size *= self._default_batch_size_divisor

        return global_batch_size

    def build(self) -> ModelTrainConfig:
        tokenizer = self.get_tokenizer_config()
        lr = 4.7e-3 * (self.model_config.parameters / self._default_vocab_size) ** (-1 / 3)

        if self.sequence_length == 4096:
            lr /= 4

        model_config = TransformerConfig.llama_like(
            d_model=self.model_config.d_model,
            n_layers=self.model_config.n_layers,
            n_heads=self.model_config.n_heads,
            vocab_size=tokenizer.padded_vocab_size(),
            compile=True,
            dp_config=TransformerDataParallelConfig(
                name=self._default_dataparallel_type,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
            ),
        )

        optim_config = AdamWConfig(
            lr=lr,
            eps=self._default_eps,
            betas=self._default_betas,
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=self._default_weight_decay)
                )
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
            tokenizer=tokenizer,
            work_dir="/tmp/dataset-cache",
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=self._default_global_batch_size,
            seed=self.seed,
            num_workers=16,
        )

        trainer_config = (
            TrainerConfig(
                save_folder=f"/tmp/{self.run_name}",
                rank_microbatch_size=self._default_device_batch_size,
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=5,
            )
            .with_callback(
                "lr_scheduler",
                SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=self.get_warmup_steps())),
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
                    project="regmixer",
                    group=self.group_id,
                    cancel_check_interval=10,
                    enabled=True,
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
                        tokenizer=tokenizer,
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
        )
