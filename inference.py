import torch 
import os
from PIL import Image, ImageDraw, ImageFont 
import torchvision.transforms as transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error
import glob
from torch import nn
import numpy as np

from config import config
from model import SimpleCNN
from custom_dataset import test_loader

def inference(model, image_path, out_path):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((config["img_width"], config["img_height"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"])
        ])
        input = transform(image).unsqueeze(0).to(config["device"])
        output = model(input)               # Forward pass

        # Get the predicted class
        age_pred = output.item()

        # Create a new image with the predicted age value on it
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        text = f"Age: {age_pred:.2f}"
        draw.text((10, 10), text, fill="red")

        # Save the output image 
        output_image.save(out_path)

def model_accuracy(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            outputs = model(images)
            outputs = outputs.squeeze()

            total_loss += loss_fn(outputs, labels).item()
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Flatten or concatenate arrays if they have consistent shapes
        try:
            all_labels = np.concatenate(all_labels).flatten()
            all_predictions = np.concatenate(all_predictions).flatten()
        except ValueError as e:
            print(f"Error concatenating arrays: {e}")
            return None, None, None

        mae = mean_absolute_error(all_labels, all_predictions)
        mse = mean_squared_error(all_labels, all_predictions)
    return total_loss/len(test_loader) , mae, mse


if __name__ == "__main__":
    path = "/home/norakami/age-prediction/checkpoints/"
    # return list of checkpoint files that match the pattern
    checkpoint_files = glob.glob(os.path.join(path, "epoch-*-valid_loss-*.pth"))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    model = SimpleCNN(input_dim=3, output_nodes=1, model_name="simple_cnn").to(config["device"])
    # load best model from the latest checkpoint
    latest_checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(latest_checkpoint["model_state_dict"])

    # for testing model accuracy 
    ## prepare test data
    loss_fn = nn.L1Loss() 
    test_loss, mae, mse = model_accuracy(model, test_loader, loss_fn)
    print(f"Test Loss: {test_loss},\n Test MAE: {mae},\n Test MSE: {mse}")

    # for inference one image at a time
    image_path = config["image_path"]
    out_path = config["output_path"]
    inference(model, image_path, out_path)


