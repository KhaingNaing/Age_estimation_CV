import torch 
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchmetrics
import os 

from config import config
from model import SimpleCNN
from functions import train_one_epoch, validate_one_epoch
from custom_dataset import train_loader, valid_loader

# Save checkpoints
checkpoint_path = os.path.join(os.getcwd(), "checkpoints")

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
else:
    print(checkpoint_path)

# Define model
model = SimpleCNN(input_dim=3, output_nodes=1, model_name=config["model_name"]).to(config["device"])
# Adam Optimizer for faster convergence
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
# Mean absolute error loss function for regression
loss_fn = nn.L1Loss() 
# Mean absolute error metric for evaluation      
metric = torchmetrics.MeanAbsoluteError().to(config["device"])

# Variables for tracking best loss
best_loss = torch.inf               # Initialize best loss to infinity
before_model_path = None            # Initialize best model path used to track previous best model checkpoint to None

# Define Tensorboard writer to capture training and validation metrics
writer = SummaryWriter(log_dir=config["log_dir"])

# Initialize metric histories 
train_loss_history = []
train_metric_history = []
valid_loss_history = []
valid_metric_history = []

# Training loop 
for epoch in range(config["epochs"]):
    # Train 
    model, train_loss, train_metric = train_one_epoch(
                                        model, 
                                        train_loader, 
                                        optimizer, 
                                        loss_fn, 
                                        metric)
    # Validate
    valid_loss, valid_metric = validate_one_epoch(
                                        model, 
                                        valid_loader, 
                                        loss_fn, 
                                        metric)
    
    # Log metrics to Tensorboard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Metric/train", train_metric, epoch)
    writer.add_scalar("Loss/valid", valid_loss, epoch)
    writer.add_scalar("Metric/valid", valid_metric, epoch)

    # Append metrics to history for plotting
    train_loss_history.append(train_loss)
    train_metric_history.append(train_metric)
    valid_loss_history.append(valid_loss)
    valid_metric_history.append(valid_metric)

    # Save checkpoint
    if valid_loss < best_loss:
        best_loss = valid_loss
        if before_model_path:
            os.remove(before_model_path)
        before_model_path = os.path.join(checkpoint_path, f"epoch-{epoch}-valid_loss-{best_loss:.3}.pth")
        # torch.save(model.state_dict(), before_model_path)
        # save full checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
        }, before_model_path)
        print(f"\nModel saved in epoch: {epoch}, Model saved at {before_model_path}")

    # if epoch % 5 == 0:
    print(f"Epoch: {epoch}, \
        \nTrain Loss: {train_loss:.3f}, Train Metric: {train_metric:.3f}, \
        \nValid Loss: {valid_loss:.3f}, Valid Metric: {valid_metric:.3f}")
    # Close Tensorboard writer
    writer.close()

train_loss_history_cpu = train_loss_history.cpu().numpy()
valid_loss_history_cpu = valid_loss_history.cpu().numpy()
train_metric_history_cpu = train_metric_history.cpu().numpy()
valid_metric_history_cpu = valid_metric_history.cpu().numpy()
# Plot learning curve
plt.figure(figsize=(10, 7))
plt.plot(range(config["epochs"]), train_loss_history, "o-r", label="Train Loss")
plt.plot(range(config["epochs"]), valid_loss_history, "^g", label="Valid Loss")
plt.plot(range(config["epochs"]), train_metric_history, "s-b",label="Train Metric")
plt.plot(range(config["epochs"]), valid_metric_history, "D--b", label="Valid Metric")
plt.xlabel("Epochs")
plt.ylabel("Loss/Metric")
plt.grid(True)
plt.legend()
plt.savefig("figs/metric_plot.png")

    


