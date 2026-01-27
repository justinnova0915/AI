import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
import numpy as np
import gradio as gr

# init gpu
device = torch.device('cuda')

# set up conversion pipeline (image -> tensor)
transform = v2.Compose([
    # convert to tensor
    v2.ToImage(),
    v2.Grayscale(),
    v2.Resize((28, 28)),
    v2.Lambda(lambda x: v2.functional.invert(x)),
    # convert to float32 and normalize
    v2.ToDtype(torch.float32, scale=True)
])

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

model = Classifier().to(device=device)
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

def predict(input):
    input = input["composite"]
    input = input[:, :, :3]
    input = transform(input)
    input = input.unsqueeze(0)
    input = input.to(device)
    with torch.no_grad():
        infer = model(input)

    infer = torch.nn.functional.softmax(infer, dim=1)
    
    prediction = {str(i):infer[0][i] for i in range(10)}

    print(prediction)

    return prediction

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Sketchpad(type='numpy')],
    outputs=[gr.Label()],
    live=True
)

if __name__ == "__main__":
    demo.launch(share=True)