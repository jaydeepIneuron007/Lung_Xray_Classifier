import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from src.xray.components import model 
from tqdm import tqdm



# model_architecture = model().Net()
# # To check weather cuda is available in the system or not 
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# print("Available processor {}".format(device))
# model = model_architecture.to(device)
# # To check the model summary
# summary(model, input_size=(3, 224, 224))


class ModelTrainer:
    def __init__(self, epoch, model, train_loader, test_loader, optimizer, device):
        self.epoch = epoch
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

    def train(self,):
        """
        Description: To train the model 
        
        input: model,device,train_loader,optimizer,epoch 
        
        output: loss, batch id and accuracy
        """
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get data
            data, target = data.to(self.device), target.to(self.device)
            # Initialization of gradient
            self.optimizer.zero_grad()
            # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
            # or specific requirements
            ## prediction on data
            y_pred = self.model(data)
            # Calculating loss given the prediction
            loss = F.nll_loss(y_pred, target)
            # Backprop
            loss.backward()
            self.optimizer.step()
            # get the index of the log-probability corresponding to the max value
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')


    def test(self,):
        """
        Description: To test the model
        
        input: model, self.device, test_loader
        
        output: average loss and accuracy
        
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))

    def initiate_training(self):
        # Defining the params for training 
        model =  self.model.to(self.device)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        scheduler = StepLR(self.optimizer, step_size=6, gamma=0.5)
        # EPOCHS = 4
        # Training the model
        for epoch in range(self.epoch):
            print("EPOCH:", epoch)
            self.train()
            scheduler.step()
            print('current Learning Rate: ', self.optimizer.state_dict()["param_groups"][0]["lr"])
            self.test()