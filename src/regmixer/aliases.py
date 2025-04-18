from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

from beaker import Priority
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.launch.beaker import BeakerLaunchConfig
from pydantic import BaseModel
import pydantic

PathType = Union[Path, PathLike[Any], str]


class TrainType(Enum):
    pretrain = "pretrain"
    anneal = "anneal"


class SourceConfig(BaseModel):
    name: str
    paths: list[str]
    max_repetition_factor: float = 1.0
    max_source_ratio: float = 1.0


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float
    repetition_factor: float = 1.0


class ExperimentConfig(BaseModel):
    name: str
    description: str
    budget: str
    workspace: str
    variants: int
    nodes: int
    gpus: int
    max_tokens: int
    sequence_length: int
    seed: int
    cluster: str
    tokenizer: str
    priority: Priority
    sources: list[SourceConfig]
    tokenizer: str
    proxy_model_id: str
    minimum_weight: Optional[float]
    checkpoint_path: Optional[str] = None
    train_type: TrainType = TrainType.pretrain
    allow_repetition: bool = True
    dtype: NumpyDatasetDType = NumpyDatasetDType.uint32
    mix_temperature: float = 1.0
    preemptible: bool = True
    shared_filesystem: bool = False
    weka: bool = False

    # TODO(undfined): Add field validation for weka/cluster/train_type here


class ExperimentInstance(BaseModel):
    name: str
    sources: list[SourceInstance]


class ExperimentGroup(BaseModel):
    config: ExperimentConfig
    group_id: str
    instances: list[ExperimentInstance]


class LaunchGroup(BaseModel):
    instances: list[BeakerLaunchConfig]
