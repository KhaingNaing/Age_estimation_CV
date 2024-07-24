from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import csv
import prettytable as pt
import torchmetrics
import pandas as pd

from config import config
from model import SimpleCNN
from custom_dataset import train_set, train_loader
from functions import train_one_epoch

# Find the best hyperparameters

## Calculate the loss for an untrained model
def calculate_untrain_loss(model, train_loader, loss_fn):
    x_batch, y_batch = next(iter(train_loader))
    y_pred = model(x_batch.to(config["device"]))
    loss = loss_fn(y_pred, y_batch.to(config["device"]))
    print(f"Loss for an untrained model: {loss}")

## 1. Try to train and overfit the model on a small subset of dataset (1000 samples)
def calculate_small_train_loss(model, train_set, loss_fn, optimizer, num_epochs=20):
    large_data, small_data = random_split(train_set, [len(train_set)-1000, 1000])
    small_train_loader = DataLoader(small_data, batch_size=5)
    print(f"1. Train on a small dataset of {len(small_data)} samples")
    for epoch in range(num_epochs):
        model, train_loss, train_metric = train_one_epoch(
                                            model, 
                                            small_train_loader, 
                                            optimizer, 
                                            loss_fn, 
                                            metric)
        print(f"    Epoch {epoch+1}, Loss: {train_loss}, Metric: {train_metric}")
    # [final_loss] Epoch 20, Loss: 6.098835245768229, Metric: 6.098839282989502

## 2. Train the model for a few epochs on the full dataset (experiemtn with different hyperparameters)
def find_best_lr(num_epochs=20):
    print(f"2. Train model for a few epochs on the full dataset (experiment with different lr)")
    for lr in [0.001, 0.0001, 0.0005]:
        print(f"    Learning rate is: {lr}")
        model = SimpleCNN(input_dim=3, output_nodes=1).to(config["device"])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        for epoch in range(num_epochs):
            model, train_loss, train_metric = train_one_epoch(
                                                model, 
                                                train_loader, 
                                                optimizer, 
                                                loss_fn, 
                                                metric)
            print(f"    Epoch {epoch+1}, Loss: {train_loss}, Metric: {train_metric}")
        print("\n")

## 3. Create a grid using weight decay and the best lr 
def find_best_lr_wd_pair(num_epochs=30):
    print(f"3. Create a grid using weight decay and the best lr")
    grid_list = []
    for lr in [0.0005, 0.0008, 0.001]:
        for wd in [0.001, 0.0001, 0.00001]:
            print(f"    Learning rate is: {lr}, Weight decay is: {wd}")
            model = SimpleCNN(input_dim=3, output_nodes=1).to(config["device"])
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.L1Loss()
            for epoch in range(num_epochs):
                model, train_loss, train_metric = train_one_epoch(
                                                    model, 
                                                    train_loader, 
                                                    optimizer, 
                                                    loss_fn, 
                                                    metric)
                print(f"    Epoch {epoch+1}, Loss: {train_loss}, Metric: {train_metric}")
            print("\n")
            grid_list.append([lr, wd, train_loss])
    return grid_list


if __name__ == "__main__":
    # Define model
    model = SimpleCNN(input_dim=3, output_nodes=1).to(config["device"])
    # Adam Optimizer for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Mean absolute error loss function for regression
    loss_fn = nn.L1Loss() 
    # Mean absolute error metric for evaluation      
    metric = torchmetrics.MeanAbsoluteError().to(config["device"])

    calculate_untrain_loss(model, train_loader, loss_fn)
    calculate_small_train_loss(model, train_set, loss_fn, optimizer)
    find_best_lr()
    grid_list = find_best_lr_wd_pair()

    headers = ["Learning Rate", "Weight Decay", "Loss"]
    table = pt(headers)

    # Add row to the table
    for row in grid_list:
        table.add_row(row)
    print(table)

    # Save the grid to a csv file
    with open("hyperparameter_tunning.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(grid_list)

    print("Grid has been saved to hyperparameter_tunning.csv")
