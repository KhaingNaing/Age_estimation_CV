import torch 
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import os 
import matplotlib.pyplot as plt

from config import config
from model import SimpleCNN
from custom_dataset import train_loader, valid_loader

model = SimpleCNN(input_dim=3, output_nodes=1)