import torch 
import os
from PIL import Image, ImageDraw, ImageFont 
import torchvision.transforms as transforms
import glob

from config import config
from model import SimpleCNN

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

if __name__ == "__main__":
    path = "/home/norakami/age-prediction/checkpoints/"
    # return list of checkpoint files that match the pattern
    checkpoint_files = glob.glob(os.path.join(path, "epoch-*-valid_loss-*.pth"))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    model = SimpleCNN(input_dim=3, output_nodes=1, model_name="large_cnn").to(config["device"])
    # load best model from the latest checkpoint
    latest_checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(latest_checkpoint["model_state_dict"])

    image_path = config["image_path"]
    out_path = config["output_path"]
    inference(model, image_path, out_path)


