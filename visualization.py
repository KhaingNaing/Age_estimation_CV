import os 
import random 
import matplotlib.pyplot as plt
from PIL import Image
from helper import get_all_files

# Function to show 9 random images from the dataset
def show_random_images(dataset_folder_path, save_file, image_details=None, num_images=9):
    # Get all files in the folder
    file_details = image_details if image_details else get_all_files(dataset_folder_path)
    # Randomly select 'num_images' files if not specified
    selected_file_detailes = random.sample(file_details, num_images)
    # Display the images
    for i, (filename, age) in enumerate(selected_file_detailes):
        image = Image.open(os.path.join(dataset_folder_path, f"{age}/{filename}"))
        plt.subplot(3, 3, i+1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"age: {age}")
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()

# Function to show 9 pics of blurry and sharp images and save them in a file 
def save_blurry_sharp_pics(dataset_path, blurry_images, sharp_images):
    blur_file = "/home/norakami/age-prediction/pics/blurry_images.png"
    show_random_images(dataset_path, blur_file, image_details=blurry_images)

    sharp_file = "/home/norakami/age-prediction/pics/sharp_images.png"
    show_random_images(dataset_path, sharp_file, image_details=sharp_images)

# Function to show 5 blurry and sharp images (plot image by image detail)
def show_blurry_sharp_pics(dataset_path, blurry_images, sharp_images, samples=5):
    fig, ax = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(samples):
        filename, age = blurry_images[i]
        blurry_image = Image.open(os.path.join(dataset_path, f"{age}/{filename}"))
        filename, age = sharp_images[i]
        sharp_image = Image.open(os.path.join(dataset_path, f"{age}/{filename}"))
        ax[0, i].imshow(blurry_image)
        ax[0, i].axis("off")
        ax[0, i].set_title("Blurry")
        ax[1, i].imshow(sharp_image)
        ax[1, i].axis("off")
        ax[1, i].set_title("Sharp")
    plt.tight_layout()
    plt.show()

# Function to show 5 side by side images (plot image by image tensor)
def side_by_side_images(image1s, image2s, title1, title2, samples=5):
    fig, ax = plt.subplots(2, samples, figsize=(10, 5))
    for i in range(samples):
        ax[0, i].imshow(image1s[i])
        ax[0, i].axis("off")
        ax[0, i].set_title(title1)
        ax[1, i].imshow(image2s[i])
        ax[1, i].axis("off")
        ax[1, i].set_title(title2)
    plt.tight_layout()
    plt.show()
    

# if __name__ == "__main__":
#     dataset_folder_path = "/home/norakami/age-prediction/dataset/"
#     save_simple_images = "/home/norakami/age-prediction/pics/sample_images.png"
#     show_random_images(9, dataset_folder_path, save_simple_images)