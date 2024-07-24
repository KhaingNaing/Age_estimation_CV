import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import config

# Custom Simple CNN model 
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_nodes, model_name="simple_cnn"):
        super(SimpleCNN, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes

        if model_name == "simple_cnn":
            self.model = nn.Sequential(
                # 1st Convolutional Block
                nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 2nd Convolutional Block
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 3rd Convolutional Block
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 4th Convolutional Block
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Flatten
                nn.Flatten(),
                # 1st Fully Connected Layer
                nn.Linear(128 * (config["img_size"]  // 16) * (config["img_size"] // 16), 64),
                nn.ReLU(inplace=True),
                # 2nd Fully Connected Layer
                nn.Linear(64, output_nodes),
            )
        elif model_name == "large_cnn":
            self.model = nn.Sequential(
                # 1st Convolutional Block
                nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),  # Increased from 8
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 2nd Convolutional Block
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Increased from 16
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 3rd Convolutional Block
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Increased from 32
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 4th Convolutional Block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Increased from 64
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Flatten
                nn.Flatten(),
                # 1st Fully Connected Layer
                nn.Linear(256 * (config["img_size"] // 16) * (config["img_size"] // 16), 512),  # Increased from 128
                nn.ReLU(inplace=True),
                # 2nd Fully Connected Layer
                nn.Linear(512, output_nodes),  # Added an extra layer to increase parameter count
            )

        else:
            raise ValueError(f"Unsupported model name: {model_name}!")

    def forward(self, x):
        return self.model(x)
    
def total_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

if __name__ == "__main__":
    model = SimpleCNN(input_dim=3, output_nodes=1, model_name="large_cnn").to(config["device"])
    total_parameters = total_params(model)

    print(f"Total number of parameters: {total_parameters}")
    
