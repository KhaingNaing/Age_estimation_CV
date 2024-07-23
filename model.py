import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import config

# Custom Simple CNN model 
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_nodes):
        super(SimpleCNN, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes

        self.model = nn.Sequential(
            # 1st Convolutional Block
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2nd Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3rd Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4th Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Flatten
            nn.Flatten(),
            # 1st Fully Connected Layer
            nn.Linear(256 * (config['img_size'] // 16) * (config['img_size'] // 16), 512),
            nn.ReLU(inplace=True),
            # 2nd Fully Connected Layer
            nn.Linear(512, output_nodes),
        )

    def forward(self, x):
        return self.model(x)
    
