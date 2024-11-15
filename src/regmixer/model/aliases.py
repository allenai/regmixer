from dataclasses import dataclass
from enum import Enum

from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train import TrainerConfig


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
    embedding_size: int
    init_device: str
    init_fn: str
    init_std: float
    init_cutoff_factor: int
    norm_after: bool
    precision: str

    @classmethod
    def olmo_190m(cls) -> "ModelConfig":
        return ModelConfig(
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
            embedding_size=100352,
            init_device="cuda",
            init_fn="normal",
            init_std=0.02,
            init_cutoff_factor=3,
            norm_after=True,
            precision="amp_bf16",
        )

    @classmethod
    def olmo_30m(cls) -> "ModelConfig":
        return ModelConfig(
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
            embedding_size=100352,
            init_device="cuda",
            init_fn="normal",
            init_std=0.02,
            init_cutoff_factor=3,
            norm_after=True,
            precision="amp_bf16",
        )


class SupportedModels(Enum):
    olmo_190m = ModelConfig.olmo_190m()
    olmo_30m = ModelConfig.olmo_30m()


class SupportedTokenizers(Enum):
    dolma2 = TokenizerConfig.dolma2()
    gpt_neox = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
