import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# init gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up conversion pipeline (image -> tensor)
transform = v2.Compose([
    # convert to tensor
    v2.ToImage(),
    # convert to float32 and normalize
    v2.ToDtype(torch.float32, scale=True)
])

# model architecture
#     Feature extraction:
#         Conv -> ReLU -> Maxpool
#         X2 times
#     Classification
#         Flatten
#         Linear

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, (1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1, (1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2), 2)
        self.linear = nn.Linear(3136, 10)

    def forward(self, input):
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)
        maxpool1 = self.maxpool1(relu1)
        conv2 = self.conv2(maxpool1)
        relu2 = self.relu2(conv2)
        maxpool2 = self.maxpool2(relu2)
        flatten = torch.flatten(maxpool2, start_dim=1)
        result = self.linear(flatten)
        return result

if __name__ == '__main__':

    # import MNIST dataset
    train_data = torchvision.datasets.MNIST('./mnist', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST('./mnist', train=False, transform=transform, download=True)

    # init Dataloader
    train_load = torch.utils.data.DataLoader(train_data, 32, True, num_workers=2)
    test_load = torch.utils.data.DataLoader(test_data, 32, False, num_workers=2)

    # init model
    model = Classifier().to(device=device)
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # init criterion
    criterion = torch.nn.CrossEntropyLoss()

    #set epoch
    epoch = 3

    for i in tqdm(range(epoch)):
        # load images into the gpu
        pbar = tqdm(train_load, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # clear grad
            optimizer.zero_grad()

            # get prediction
            pred = model(images)

            # calculate loss
            loss = criterion(pred, labels)

            pbar.set_description(f"Loss: {loss.item():.4f}")

            # backpropagation
            loss.backward()

            # gradient decent
            optimizer.step()

    model.eval()

    correct = 0

    with torch.no_grad():
        for test_images, test_labels in test_load:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            _, test_result = torch.max(model(test_images), 1)
            correct += (test_result == test_labels).sum().item()

    print(f"Accuracy: {100 * correct / 10000}%")
    torch.save(model.state_dict(), "mnist_cnn.pth")