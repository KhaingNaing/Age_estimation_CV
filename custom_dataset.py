import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import cv2

from config import config

# Stratified split the dataset into training, validation, and test sets
df = pd.read_csv("/home/norakami/age-prediction/csv_dataset/age_dataset.csv")

# 80% train, 10% validation, 10% test
df_train, df_temp = train_test_split(df, train_size=0.8, stratify=df.age, random_state=config['seed'])
df_test, df_valid = train_test_split(df_temp, train_size=0.5, stratify=df_temp.age, random_state=config['seed']) 

directory = "./csv_dataset"
if not os.path.exists(directory):
    os.makedirs(directory)

df_train.to_csv(f"{directory}/train_set.csv", index=False)
df_valid.to_csv(f"{directory}/valid_set.csv", index=False)
df_test.to_csv(f"{directory}/test_set.csv", index=False)

print("All datasets have been saved successfully!")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])

def preprocess_image(image):
    # Convert to OpenCV image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Deblurring 
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Convert to PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


class CustomDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = os.path.join(self.root_dir, self.data.iloc[idx, 1], self.data.iloc[idx, 0])
        image = Image.open(img_dir).convert('RGB')  
        image = preprocess_image(image)
        image = self.transform(image)
        age = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        return image, age
    

root_dir = "/home/norakami/age-prediction/dataset"
train_csv = "/home/norakami/age-prediction/csv_dataset/train_set.csv"
valid_csv = "/home/norakami/age-prediction/csv_dataset/valid_set.csv"
test_csv = "/home/norakami/age-prediction/csv_dataset/test_set.csv"

train_set = CustomDataset(root_dir, train_csv, transform)
valid_set = CustomDataset(root_dir, valid_csv, transform)
test_set = CustomDataset(root_dir, test_csv, transform)

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'])
test_loader = DataLoader(test_set, batch_size=config['batch_size'])

print("Data loaders have been created successfully!")

if __name__ == "__main__":
    print("Train set size:", len(train_set))
    print("Validation set size:", len(valid_set))
    print("Test set size:", len(test_set))
    print("Number of batches in train loader:", len(train_loader))
    print("Number of batches in validation loader:", len(valid_loader))
    print("Number of batches in test loader:", len(test_loader))