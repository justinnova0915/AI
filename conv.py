import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np

# init gpu
device = torch.device('cuda')

# set up conversion pipeline (image -> tensor)
transform = v2.Compose([
    # convert to tensor
    v2.ToImage(),
    # convert to float32 and normalize
    v2.ToDtype(torch.float32, scale=True)
])

# import MNIST dataset
train_data = torchvision.datasets.MNIST('./mnist', train=True, transform=transform)
test_data = torchvision.datasets.MNIST('./mnist', train=False, transform=transform)

# init Dataloader
train_load = torch.utils.data.DataLoader(train_data, 32, True, num_workers=2)
test_load = torch.utils.data.DataLoader(test_data, 32, False, num_workers=2)

# init model class
class classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, (1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2), 2)