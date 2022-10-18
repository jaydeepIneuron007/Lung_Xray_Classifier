from pickletools import optimize
import torch 
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pathlib import Path
from src.xray.entity.config_entity import EvaluationConfig, TransformDataConfig
from torch.utils.data import DataLoader
from src.xray.utils import save_json
from torchvision import datasets, transforms
from src.xray.components.image_transformation import TransformData
from torch.optim import SGD
from src.xray.components.model import Net


class Evaluation:

    def __init__(self,test_loader, device):
        # self.epoch = epoch
        # self.model = model
        # self.train_loader = train_loader
        self.test_loader = test_loader
        # self.optimizer = optimizer
        self.device = device
        self.TransformData  = TransformData()
        self.test_loss = 0
        self.test_accuracy = 0
        self.total = 0
        self.total_batch = 0


    def configuration(self):
        try:
            train_Dataloader, test_DataLoader = self.TransformData.data_loader()
            print(test_DataLoader)

            model = Net()

            model.load_state_dict(torch.load('artifacts/training/model.pt'))

            model.to(self.device)


            # net = torch.load('artifacts/training/model.pt').to(self.device)
            cost = CrossEntropyLoss()
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.8)
            model.eval()
            return test_DataLoader, model, cost, optimizer
        except Exception as e:
            raise e

    def test_net(self):
        try:
            test_DataLoader, net, cost, optimizer = self.configuration()
            with torch.no_grad():
                holder = []
                for batch, data in enumerate(test_DataLoader):
                    images = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    output = net(images)
                    loss = cost(output, labels)

                    predictions = torch.argmax(output, 1)

                    for i in zip(images, labels, predictions):
                        h = list(i)
                        # h[0] = wandb.Image(h[0])
                        holder.append(h)

                    print(f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}", )

                    self.test_loss += loss.item()
                    self.test_accuracy += (predictions == labels).sum().item()
                    self.total_batch += 1
                    self.total += labels.size(0)

                    print(f"Model  -->   Loss : {self.test_loss / self.total_batch} Accuracy : {(self.test_accuracy / self.total) * 100} %")
            

        except Exception as e:
            raise e

    