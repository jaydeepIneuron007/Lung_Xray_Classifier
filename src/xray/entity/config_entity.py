from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    train_path: Path
    test_path: Path
    params_test_ratio: float

@dataclass(frozen=True)
class DataIngestionArtifacts:
    ingested_train_dir : Path
    ingested_test_dir: Path

