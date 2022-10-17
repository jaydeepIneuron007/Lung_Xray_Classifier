import os 
from xray.entity.config_entity import TransformDataConfig
from xray import logger
from xray.utils import get_size
#from tqdm import tqdm 
#from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TransformData:
    def __init__(self, config: TransformDataConfig):
        self.config = config
    
    def transforming_training_data(self):
        train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=self.config.params_brightness, contrast=self.config.params_contrast, saturation=self.config.params_saturation, hue=self.config.params_hue),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
        ])
        return train_transform
    
    def transforming_testing_data(self):
        test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
        ])
        return test_transform
        
    def data_loader(self):
        data_path = self.config.ingested_data
        train_transform = self.transforming_training_data()
        test_transform = self.transforming_testing_data()
        os.makedirs(os.path.join(data_path, 'train'),exist_ok=True)
        os.makedirs(os.path.join(data_path, 'test'),exist_ok=True)

        train_data = datasets.ImageFolder(os.path.join(data_path, 'train'), transform= train_transform)
        test_data = datasets.ImageFolder(os.path.join(data_path, 'test'), transform= test_transform)
        
        
        train_loader = DataLoader(train_data,
                                  batch_size= self.config.params_batch_size, shuffle= self.config.params_shuffle, pin_memory= self.config.params_pin_memory)
        test_loader = DataLoader(test_data,
                                 batch_size=self.config.params_batch_size , shuffle= self.config.params_shuffle, pin_memory= self.config.params_pin_memory)
        class_names = train_data.classes
        print(class_names)
        print(f'Number of train images: {len(train_data)}')
        print(f'Number of test images: {len(test_data)}')
        return train_loader,test_loader
        
    def run_transformation_data(self):
#         self.get_file_names()
        # self.transforming_training_data()
        # self.transforming_testing_data()
        train_loader,test_loader = self.data_loader()
        return train_loader,test_loader