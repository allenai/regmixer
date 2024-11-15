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
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
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
from regmixer.model.aliases import (
    ModelConfig,
    ModelTrainConfig,
    SupportedModels,
    SupportedTokenizers,
)


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
        tokenizer (str): The tokenizer to be used.
        model_config (ModelConfig): The model configuration to be used.
        profile (bool): Whether to enable profiling. Default is False.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, cluster, beaker_user, seed, s3, config, profile):
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
    tokenizer: TokenizerConfig
    dtype: str
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
        tokenizer: str,
        dtype: str,
        model_identifier: str,
        seed: int = 42,
        s3: bool = True,
        profile: bool = False,
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
        self.tokenizer = self.get_tokenizer_config(tokenizer=tokenizer)
        self.read_location = self.get_read_location()
        self.root_dir: str = "s3://ai2-llm"
        self.dataset_dtype = NumpyDatasetDType[dtype]

        if "jupiter" in cluster and not s3:
            self.root_dir = "/weka/oe-training-default/ai2-llm"

        # TODO: Move these to defaults into model/aliases.py
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
        self._default_save_interval = 1000
        self._default_eval_interval = 200

    def get_read_location(self) -> str:
        return ("s3://ai2-llm" if self.s3 else "/weka/oe-training-default/ai2-llm").rstrip("/")

    def get_tokenizer_config(self, tokenizer) -> TokenizerConfig:
        try:
            return SupportedTokenizers[tokenizer].value
        except ValueError as e:
            logger.info(f"Invalid tokenizer identifier: {tokenizer}")
            raise e

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

        global_batch_size = self.next_power_of_2(global_batch_size)
        print(f"Global batch size: {global_batch_size}")

        return global_batch_size

    def next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

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
                name=self.run_name.strip(),
                project="regmixer",
                group=self.group_id.strip(),
                cancel_check_interval=10,
                enabled=True,
            ),
            "lm_evaluator": LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=self.root_dir,
                    sequence_length=self.sequence_length,
                    tokenizer=self.tokenizer,
                    work_dir=f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/dataset-cache",
                ),
                eval_interval=self._default_eval_interval,
            ),
            "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
                tasks=[task.value for task in DownstreamEvaluators],
                tokenizer=self.tokenizer,
                eval_interval=self._default_eval_interval,
            ),
        }

    def build(self) -> ModelTrainConfig:
        tokenizer = self.tokenizer
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
            dtype=self.dataset_dtype,
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
