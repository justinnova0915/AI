import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device('cuda')

# generate data
noise = torch.rand(1000, 1, device=device) * 0.1

# generate data
x = torch.rand(1000, 1, device=device) * 11
real_y = torch.sin(x)/2
y = (real_y) + noise

# create the nn subclass
class ParabolaSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1)
    
    def forward(self, input):
        l1 = self.hidden(input)
        r1 = torch.relu(l1)
        l2 = self.hidden2(r1)
        r2 = torch.relu(l2)
        result = self.output(r2)
        return result

# init model
model = ParabolaSolver().to(device=device)

# init optimizer (the thing that tunes the params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# init criterion (the thing that calculates the loss)
criterion = torch.nn.MSELoss()

# training loop
for i in range(10000):
    # get prediction
    y_pred = model(x)

    # calculate loss
    loss = criterion(y_pred, y)

    # clear gradient
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # adjust parameters
    optimizer.step()

# get prediction
with torch.no_grad():
    pred = model(x)

pred = pred.to('cpu')
x = x.to('cpu')
y = y.to('cpu')

plt.scatter(x, y)
plt.scatter(x, pred, color='red')
plt.show()