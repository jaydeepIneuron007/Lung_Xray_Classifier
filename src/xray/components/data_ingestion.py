import os
import urllib.request as request
from zipfile import ZipFile
from xray.entity.config_entity import DataIngestionConfig,DataIngestionArtifacts
from xray import logger
from xray.utils import get_size
from tqdm import tqdm 
from pathlib import Path
import numpy as np
import shutil
import random

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )

    def _get_updated_list_of_files(self, list_of_files):
        return [f for f in list_of_files if f.endswith(".jpeg") and ("NORMAL" in f or "PNEUMONIA" in f)]

    def _preprocess(self, zf: ZipFile, f: str, working_dir: str):
        target_filepath = os.path.join(working_dir, f)
        if not os.path.exists(target_filepath):
            zf.extract(f, working_dir)
        
        if os.path.getsize(target_filepath) == 0:
            os.remove(target_filepath)
            
    def unzip_and_clean(self):
        
        with ZipFile(file=self.config.local_data_file, mode="r") as zf:
            #print(self.config.local_data_file)
            list_of_files = zf.namelist()
            updated_list_of_files = self._get_updated_list_of_files(list_of_files)
            #print(updated_list_of_files)
            for f in updated_list_of_files:
                self._preprocess(zf, f, self.config.unzip_dir)
    
    def train_test_split(self):
        """
        This function would split the raw data into train and test folder
        """
        try:
            # 1. make train and test folder 
            unzip_images = self.config.unzip_dir
            train_path = self.config.train_path
            test_path = self.config.test_path
            
            #params.yaml
            test_ratio = self.config.params_test_ratio
            # 1. make train and test folder 
            train_path = os.path.join(os.getcwd(), unzip_images, train_path)
            test_path = os.path.join(os.getcwd(), unzip_images, test_path)
            #print(train_path)
            classes_dir = ['NORMAL', 'PNEUMONIA']

            for cls in classes_dir:
                os.makedirs(os.path.join(train_path, cls), exist_ok= True)
                os.makedirs(os.path.join(test_path, cls), exist_ok=True)
            
            # 2. Split the raw data
            raw_data_path = os.path.join(os.getcwd(), unzip_images, 'chest_xray')
            for cls in classes_dir:
                allFileNames = os.listdir(os.path.join(raw_data_path, cls))
                #print(allFileNames)
                np.random.shuffle(allFileNames)
                train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                    [int(len(allFileNames)* (1 - test_ratio))])

                train_FileNames = [os.path.join(raw_data_path, cls, name) for name in train_FileNames.tolist()]
                test_FileNames = [os.path.join(raw_data_path, cls, name) for name in test_FileNames.tolist()]

                #print(train_FileNames)
                for name in train_FileNames:
                    shutil.copy(name, os.path.join(train_path, cls))

                for name in test_FileNames:
                    shutil.copy(name, os.path.join(test_path, cls))
                
                #print(test_FileNames)
            data_ingestion_artifact = DataIngestionArtifacts(ingested_train_dir=train_path,
                                                             ingested_test_dir= test_path)
            return data_ingestion_artifact

        except Exception as e:
            raise e

    def remove_raw_data_dir(self):
        
        shutil.rmtree(os.path.join(self.config.unzip_dir,'chest_xray'), ignore_errors=True)
    def run_data_ingestion(self):
        self.download_file()
        self.unzip_and_clean()
        data_ingestion_artifact = self.train_test_split()
        self.train_test_split()
        self.remove_raw_data_dir()
        return data_ingestion_artifact