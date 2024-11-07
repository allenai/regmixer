import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.io import is_url
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    WandBCallback,
)

from regmixer.aliases import SourceInstance
from regmixer.data.dataset import MixtureBuilder
from regmixer.model.evaluators import DownstreamEvaluators


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
        sources (List[SourceInstance]): A list of source instances.
        sequence_length (int): The sequence length for the model.
        max_tokens (int): The maximum number of tokens.
        model_config (ModelConfig): The model configuration.
        group_id (str): The group ID for the run.
        cluster (str): The cluster name.
        beaker_user (str): The Beaker user name.
        s3 (bool): Whether to use S3 for storage.
        seed (int): The random seed for reproducibility. Default is 42.
        config (Optional[ModelConfig]): An optional model configuration. Default is None.
        overrides (Optional[List[str]]): A list of override strings. Default is an empty list.
        profile (bool): Whether to enable profiling. Default is False.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, cluster, beaker_user, seed, s3, config, overrides, profile):
            Initializes the TransformerConfigBuilder with the provided parameters.

        get_read_location() -> str:
            Returns the read location based on whether S3 is used.

        get_tokenizer_config() -> TokenizerConfig:
            Returns the tokenizer configuration.

        get_warmup_steps() -> int:
            Returns the number of warmup steps.

        get_batch_size():
            Returns the global batch size based on the sequence length and model parameters.

        build_callbacks() -> Dict[str, Callback]:
            Builds and returns a dictionary of callbacks for the trainer.

        build() -> ModelTrainConfig:
            Builds and returns the model training configuration.
    """

    run_name: str
    sources: List[SourceInstance]
    sequence_length: int
    max_tokens: int
    model_config: ModelConfig
    group_id: str
    cluster: str
    beaker_user: str
    s3: bool
    seed: int
    config: Optional[ModelConfig] = None
    overrides: Optional[List[str]] = None
    profile: bool = False

    def __init__(
        self,
        run_name: str,
        sources: List[SourceInstance],
        sequence_length: int,
        max_tokens: int,
        group_id: str,
        cluster: str,
        beaker_user: str,
        seed: int = 42,
        s3: bool = True,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        overrides: List[str] = [],
        profile: bool = False,
    ):
        self.run_name = run_name
        self.sources = sources
        self.sequence_length = sequence_length
        self.max_tokens = max_tokens
        self.group_id = group_id
        self.seed = seed
        self.overrides = overrides
        self.model_config = config
        self.beaker_user = beaker_user
        self.profile = profile
        self.s3 = s3
        self.read_location = self.get_read_location()
        self.root_dir: str = "s3://ai2-llm"

        if "jupiter" in cluster and not s3:
            self.root_dir = "/weka/oe-training-default/ai2-llm"

        self._default_device_batch_size = 8
        self._default_betas = (0.9, 0.95)
        self._default_weight_decay = 0.1
        self._default_eps = 1e-8
        self._default_max_grad_norm = 1.0
        self._default_decay_embeddings = False
        self._default_vocab_size = 100278
        self._default_batch_size_divisor = 32
        self._default_global_batch_size = self.get_batch_size()
        self._default_device_batch_size = 8
        self._default_dataparallel_type = DataParallelType.ddp
        self._default_dataset_dtype = NumpyDatasetDType.uint32
        self._default_save_interval = 1000
        self._default_eval_interval = 200

    def get_read_location(self) -> str:
        return ("s3://ai2-llm" if self.s3 else "/weka/oe-training-default/ai2-llm").rstrip("/")

    def get_tokenizer_config(self) -> TokenizerConfig:
        # TODO: Decide whether to make this configurable
        return TokenizerConfig.dolma2()

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

        print(f"Global batch size: {global_batch_size}")
        return global_batch_size

    def build_callbacks(self) -> Dict[str, Callback]:
        return {
            "lr_scheduler": SchedulerCallback(
                scheduler=CosWithWarmup(warmup_steps=self.get_warmup_steps())
            ),
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "grad_clipper": GradClipperCallback(max_grad_norm=self._default_max_grad_norm),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "checkpointer": CheckpointerCallback(
                save_interval=self._default_save_interval,
                ephemeral_save_interval=100,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=self.run_name,
                project="regmixer",
                group=self.group_id,
                cancel_check_interval=10,
                enabled=True,
            ),
            "lm_evaluator": LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=self.root_dir,
                    sequence_length=self.sequence_length,
                    tokenizer=self.get_tokenizer_config(),
                    work_dir=f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/dataset-cache",
                ),
                eval_interval=self._default_eval_interval,
            ),
            "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
                tasks=[task.value for task in DownstreamEvaluators],
                tokenizer=self.get_tokenizer_config(),
                eval_interval=self._default_eval_interval,
            ),
        }

    def build(self) -> ModelTrainConfig:
        tokenizer = self.get_tokenizer_config()
        lr = 4.7e-3 * (self.model_config.parameters / self._default_vocab_size) ** (-1 / 3)

        if self.sequence_length == 4096:
            lr /= 4
            raise NotImplementedError("Only sequence length 2048 is supported right now")

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
            processes=min(os.cpu_count() or 1, 6),
            dtype=self._default_dataset_dtype,
        ).build()

        dataset_config = NumpyDatasetConfig(
            source_mixture_config=mixture_config,
            name=NumpyDatasetType.fsl,
            sequence_length=self.sequence_length,
            tokenizer=tokenizer,
            work_dir="/tmp/dataset-cache",
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=self._default_global_batch_size * self.sequence_length,
            seed=self.seed,
            num_workers=16,
        )

        trainer_config = TrainerConfig(
            save_folder=f"/tmp/{self.run_name}",
            rank_microbatch_size=self._default_device_batch_size * self.sequence_length,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=5,
        )

        for callback_name, callback in self.build_callbacks().items():
            trainer_config.callbacks[callback_name] = callback

        return ModelTrainConfig(
            model=model_config,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
        )
