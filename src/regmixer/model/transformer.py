import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.nn.transformer import TransformerConfig
import olmo_core.train.train_module as tm

from olmo_core.optim import (
    LinearWithWarmup,
    Scheduler,
)
from olmo_core.optim.scheduler import CosWithWarmupAndLinearDecay
from olmo_core.train import TrainerConfig, Duration
from olmo_core.train.common import LoadStrategy
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)

from regmixer.aliases import SourceInstance, TrainType
from regmixer.data.dataset import MixtureBuilder
from regmixer.model.aliases import (
    ModelConfig,
    ModelTrainConfig,
    SupportedModels,
    SupportedTokenizers,
)
from regmixer.model.evaluators import DownstreamEvaluatorsSmall, DownstreamEvaluators, CodeTasks

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfigBuilder:
    """
    A builder class for configuring and creating a transformer model training configuration.

    Attributes:
        run_name (str): The name of the run.
        sources (List[SourceInstance]): A list of source instances.
        sequence_length (int): The sequence length for the model.
        max_tokens (int): The maximum number of tokens to be processed in a batch.
        model_config (ModelConfig): The model configuration.
        group_id (str): The group ID for the run.
        cluster (str): The cluster name.
        beaker_user (str): The Beaker user name.
        s3 (bool): Whether to use S3 for storage.
        seed (int): The random seed for reproducibility. Default is 42.
        tokenizer (TokenizerConfig): The tokenizer configuration.
        dtype (str): The data type for the dataset.
        weka (bool): Whether to use Weka buckets. Default is False.
        train_type (TrainType): The training type. Default is TrainType.pretrain.
        load_path (Optional[str]): The path to load a pre-trained model. Default is None.
        profile (bool): Whether to enable profiling. Default is False.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, cluster, beaker_user,
                 tokenizer, dtype, model_identifier, weka, train_type=TrainType.pretrain, load_path=None, seed=42, s3=True, profile=False):
            Initializes the TransformerConfigBuilder.

        get_tokenizer_config(tokenizer: str) -> TokenizerConfig:
            Returns the tokenizer configuration based on the tokenizer identifier.

        get_warmup_steps(parameters: int) -> int:
            Returns the number of warmup steps based on the model parameters.

        get_batch_size(parameters: int) -> int:
            Returns the global batch size based on the sequence length and model parameters.

        next_power_of_2(x: int) -> int:
            Returns the next power of 2 greater than or equal to x.

        get_lr(model: TransformerConfig, tokenizer: TokenizerConfig) -> float:
            Returns the learning rate based on the model and tokenizer configurations.

        get_scheduler(model: TransformerConfig) -> Scheduler:
            Returns the learning rate scheduler based on the model configuration.

        build_callbacks(model: TransformerConfig) -> Dict[str, Callback]:
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
    tokenizer: TokenizerConfig
    dtype: str
    weka: bool
    device_batch_size: int
    load_path: Optional[str] = None
    profile: bool = False
    train_type: TrainType = TrainType.pretrain

    def __init__(
        self,
        run_name: str,
        sources: List[SourceInstance],
        sequence_length: int,
        max_tokens: int,
        group_id: str,
        cluster: str,
        beaker_user: str,
        tokenizer: str,
        dtype: str,
        model_identifier: str,
        weka: bool,
        device_batch_size: int,
        train_type: TrainType = TrainType.pretrain,
        load_path: Optional[str] = None,
        seed: int = 42,
        s3: bool = True,
        profile: bool = False,
        global_batch_size: Optional[int] = None,
    ):
        self.run_name = run_name
        self.sources = sources
        self.sequence_length = sequence_length
        self.max_tokens = max_tokens
        self.group_id = group_id
        self.seed = seed
        self.model_config = SupportedModels[model_identifier].value
        self.beaker_user = beaker_user
        self.profile = profile
        self.s3 = s3
        self.train_type = train_type
        self.load_path = load_path
        self.device_batch_size = device_batch_size
        self.global_batch_size = global_batch_size
        self.tokenizer = self.get_tokenizer_config(tokenizer=tokenizer)
        self.data_dir: str = "s3://ai2-llm"
        self.dataset_dtype = NumpyDatasetDType[dtype]
        self.root_dir = f"/tmp/{self.run_name}"
        self.cluster = cluster
        self.weka = weka

        # Default will always be s3 for checkpoints, and we override if Augusta or AUS+Weka
        self.checkpoint_dir = (
            f"{self.data_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
        )

        self._setup_dirs()

        self.dataset_cache = (
            f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"
        )

    def _setup_dirs(self) -> None:
        """Setup checkpoint directory based on cluster configuration."""
        if any(substring in self.cluster for substring in ["augusta"]):
            self.root_dir = "gs://ai2-llm"
        elif any(substring in self.cluster for substring in ["jupiter", "saturn"]) and self.weka:
            logger.info("Using Weka bucket as root dir")
            self.root_dir = "/weka/oe-training-default/ai2-llm"

        self.checkpoint_dir = (
            f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
        )

    def get_tokenizer_config(self, tokenizer) -> TokenizerConfig:
        try:
            return SupportedTokenizers[tokenizer].value
        except ValueError as e:
            logger.info(f"Invalid tokenizer identifier: {tokenizer}")
            raise e

    def get_warmup_steps(self, parameters: int) -> int:
        if self.train_type == TrainType.anneal:
            return 0
        bsz = (
            self.global_batch_size
            if self.global_batch_size is not None
            else self.get_batch_size(parameters)
        )
        return round(parameters / (bsz * self.model_config.max_sequence_length))

    def get_batch_size(self, parameters: int) -> int:
        """
        Taken directly from https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/model_ladder.py#L276
        Args:
        - parameters: number of non-embedding parameters
        """
        if self.train_type == TrainType.anneal:
            return 1024

        assert self.sequence_length in {2048, 4096, 8192}
        seq_len_divisor = self.sequence_length // 2048

        global_batch_size = 160 * (parameters / 108000000) ** (2 / 3)
        global_batch_size /= seq_len_divisor
        global_batch_size /= self.model_config.batch_divisor
        global_batch_size = round(global_batch_size)
        global_batch_size *= self.model_config.batch_divisor
        global_batch_size = self.next_power_of_2(global_batch_size)
        print(f"Global batch size is: {global_batch_size}")

        return global_batch_size

    def next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def get_lr(self, model: TransformerConfig, tokenizer: TokenizerConfig) -> float:
        """
        Taken from https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo2-ladder.py#L54
        """
        if self.train_type == TrainType.anneal:
            return 6.1852e-5  # Magic number pulled from OLMo-core examples

        assert self.sequence_length in {2048, 4096}
        lr = 0.0047 * (model.num_non_embedding_params / 108000000) ** (-1 / 3)
        if self.sequence_length == 4096:
            lr /= 4
        return lr

    def get_scheduler(self, model: TransformerConfig) -> Scheduler:
        if self.train_type == TrainType.anneal:
            return LinearWithWarmup(warmup_steps=0, t_max=self.max_tokens)

        return CosWithWarmupAndLinearDecay(
            warmup_steps=self.get_warmup_steps(model.num_params),
        )

    def build_callbacks(self) -> Dict[str, Callback]:
        return {
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "checkpointer": CheckpointerCallback(
                save_interval=self.model_config.save_interval,
                ephemeral_save_interval=100,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=self.run_name.strip(),
                project="regmixer",
                group=self.group_id.strip(),
                enabled=True,
            ),
        }

    def build(self) -> ModelTrainConfig:
        tokenizer = self.tokenizer
        model = TransformerConfig.llama_like(
            d_model=self.model_config.d_model,
            n_layers=self.model_config.n_layers,
            n_heads=self.model_config.n_heads,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_theta=self.model_config.rope_theta,
            layer_norm_eps=self.model_config.layer_norm_eps,
            qk_norm=self.model_config.qk_norm,
            block_name=self.model_config.block_type,
        )

        global_batch_size = (
            self.global_batch_size
            if self.global_batch_size is not None
            else self.get_batch_size(model.num_non_embedding_params)
        )
        learning_rate = self.get_lr(model, tokenizer)

        mixture_config = MixtureBuilder(
            sources=self.sources,
            max_tokens=self.max_tokens,
            sequence_length=self.sequence_length,
            seed=self.seed,
            processes=min(os.cpu_count() or 1, 16),
            dtype=self.dataset_dtype,
        ).build()

        dataset_config = NumpyDatasetConfig(
            source_mixture_config=mixture_config,
            name=NumpyDatasetType.fsl,
            sequence_length=self.sequence_length,
            tokenizer=tokenizer,
            work_dir=self.dataset_cache,
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=global_batch_size * self.sequence_length,
            work_dir=self.dataset_cache,
            seed=self.seed,
            num_workers=16,
        )

        train_module_config = tm.TransformerTrainModuleConfig(
            rank_microbatch_size=self.device_batch_size * self.sequence_length,
            max_sequence_length=self.sequence_length,
            optim=SkipStepAdamWConfig(
                lr=learning_rate,
                weight_decay=0.033,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
            ),
            compile_model=True,
            dp_config=tm.TransformerDataParallelConfig(
                name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            float8_config=Float8Config(enabled=False),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=self.get_scheduler(model),
        )

        trainer_config = TrainerConfig(
            save_folder=self.checkpoint_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            load_path=self.load_path,
            # We fail fast if an existing if we expect a checkpoint for annealing and one is not found.
            load_strategy=(
                LoadStrategy.always
                if self.train_type == TrainType.anneal
                else LoadStrategy.if_available
            ),
            max_duration=Duration.tokens(self.max_tokens),
        )

        for callback_name, callback in self.build_callbacks().items():
            trainer_config.callbacks[callback_name] = callback

        return ModelTrainConfig(
            model=model,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
            train_module=train_module_config,
        )
