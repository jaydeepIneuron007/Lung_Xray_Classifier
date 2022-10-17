from xray.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH
from xray.utils import read_yaml, create_directories
from xray.entity.config_entity import (
    DataIngestionConfig,
    DataIngestionArtifacts,
    TransformDataConfig,
    TrainingConfig,
)

from pathlib import Path
import os

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            train_path=config.train_path,
            test_path=config.test_path,
            params_test_ratio=self.params.TEST_RATIO,
        )

        return data_ingestion_config
    
    def get_transform_data_config(self) -> TransformDataConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        transform_data_config = TransformDataConfig(
            root_dir = Path(config.root_dir),
            unzip_dir = Path(config.unzip_dir),
            ingested_data = Path(config.ingested_data),
            transform_train_path = Path(config.transform_train_data),
            transform_test_path = Path(config.transform_test_data),
            params_brightness = self.params.BRIGHTNESS,
            params_contrast = self.params.CONTRAST,
            params_saturation = self.params.SATURATION,
            params_hue = self.params.HUE,
            params_batch_size = self.params.BATCH_SIZE,
            params_shuffle = self.params.SHUFFLE,
            params_pin_memory = self.params.PIN_MEMORY,
                                                    )
        return transform_data_config
        
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir)
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCH,
            
        )

        return training_config