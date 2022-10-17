from src.xray.config.configuration import ConfigurationManager
from src.xray.components.image_transformation import TransformData
from src.xray.components.data_ingestion import DataIngestion
from src.xray.components.model import Net 
from src.xray.components.training import ModelTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary


# for configs
config = ConfigurationManager()
data_ingestion_config = config.get_data_ingestion_config()
transform_data_config = config.get_transform_data_config()
training_data_config = config.get_training_config()

#ingestion
data_ingestion_comp = DataIngestion(data_ingestion_config)
data_ingestion_comp.run_data_ingestion()

# transformation
transformation_data = TransformData(config = transform_data_config)
train_loader, test_loader = transformation_data.run_transformation_data()


# model
model = Net()
# To check weather cuda is available in the system or not 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Available processor {}".format(device))
# To check the model summary
summary(model, input_size=(3, 224, 224))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
# scheduler = StepLR(optimizer, step_size=6, gamma=0.5)

model_trainer_comp=ModelTrainer(epoch=2, model=model, train_loader=train_loader,
     test_loader=test_loader, optimizer=optimizer, device=device, config= training_data_config )
model_trainer_comp.initiate_training()
