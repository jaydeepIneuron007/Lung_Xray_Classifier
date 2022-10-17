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

@dataclass(frozen=True)
class TransformDataConfig:
    root_dir: Path
    unzip_dir: Path
    transform_train_path: Path
    transform_test_path: Path
    params_brightness: float
    params_contrast: float
    params_saturation: float 
    params_hue: float
    ingested_data: Path
    params_batch_size: float
    params_shuffle: float
    params_pin_memory: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    params_epochs: int
