from os import PathLike
from pathlib import Path
from typing import Any, Union
from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerWekaBucket
from pydantic import BaseModel


PathType = Union[Path, PathLike[Any], str]


class SourceConfig(BaseModel):
    name: str
    paths: list[str]
    max_repetition_factor: float = 1.0
    max_source_ratio: float = 1.0


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float


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
    clusters: list[str]
    sources: list[SourceConfig]
    preemptible: bool = True
    shared_filesystem: bool = False
    nfs: bool = False
    weka: list[BeakerWekaBucket] = []


class ExperimentInstance(BaseModel):
    name: str
    sources: list[SourceInstance]


class ExperimentGroup(BaseModel):
    config: ExperimentConfig
    instances: list[ExperimentInstance]


class LaunchGroup(BaseModel):
    instances: list[BeakerLaunchConfig]
