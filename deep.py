import torch
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

# Init hidden and output layers
hidden_weight = (torch.randn(1, 100, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
hidden_bias = (torch.randn(100, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)

hidden2_weight = torch.randn()

output_weight = (torch.randn(100, 1, device=device, dtype=torch.float32) * np.sqrt(2/100)).detach().requires_grad_(True)
output_bias = (torch.randn(1, device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)

# training loop
for i in range(10000):

    # pass data through hidden layer
    hidden = (x @ hidden_weight) + hidden_bias

    # Apply ReLU
    hidden_relu = hidden.relu()

    # construct the final prediction
    output = (hidden_relu @ output_weight) + output_bias

    # calculate loss
    diff = (output - y)**2
    loss = diff.mean()

    # backpropagation
    loss.backward()

    # gradient descent
    with torch.no_grad():
        hidden_weight -= (hidden_weight.grad * 0.01)
        hidden_bias -= (hidden_bias.grad * 0.01)
        output_weight -= (output_weight.grad * 0.01)
        output_bias -= (output_bias.grad * 0.01)

    hidden_weight.grad = None
    hidden_bias.grad = None
    output_weight.grad = None
    output_bias.grad = None

# get prediction
y_pred = x @ hidden_weight + hidden_bias
y_pred.relu_()
y_pred = y_pred @ output_weight + output_bias

x = x.to('cpu')
y = y.to('cpu')
y_pred = y_pred.to('cpu')
y_pred.detach_()

print(y_pred)

plt.scatter(x, y)
plt.scatter(x, y_pred, color='red')
plt.show()