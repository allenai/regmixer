from dataclasses import dataclass
from pathlib import Path
from typing import List

import s3fs
from olmo_core.data.source_mixture import SourceMixtureConfig, SourceMixtureDatasetConfig
from olmo_core.data.types import NumpyDatasetDType

from regmixer.aliases import PathType, SourceInstance


@dataclass
class MixtureBuilder:
    sources: List[SourceInstance]
    max_tokens: int
    sequence_length: int
    seed: int
    dtype: NumpyDatasetDType
    processes: int = 1
    fs: s3fs.S3FileSystem = s3fs.S3FileSystem()

    def expand_globs(self, paths: List[str]) -> List[PathType]:
        expanded_paths: List[PathType] = []
        for path in paths:
            new = [Path(f"s3://{obj}") for obj in self.fs.glob(path)]
            expanded_paths.extend(new)

        return expanded_paths

    def build(self) -> SourceMixtureDatasetConfig:
        source_configs: List[SourceMixtureConfig] = []
        for source in self.sources:
            source_configs.append(
                SourceMixtureConfig(
                    source_name=source.name,
                    paths=self.expand_globs(source.paths),
                    target_ratio=source.ratio,
                )
            )

        return SourceMixtureDatasetConfig(
            source_configs=source_configs,
            max_tokens=self.max_tokens,
            sequence_length=self.sequence_length,
            seed=self.seed,
            dtype=self.dtype,
            processes=self.processes,
        )
