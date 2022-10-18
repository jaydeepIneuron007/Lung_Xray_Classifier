import torch
from src.xray.components.model import Net

model = Net()

model.load_state_dict(torch.load('artifacts/training/model.pt'))

print(model.to("cpu"))

