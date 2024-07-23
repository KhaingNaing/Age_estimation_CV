import os 
import random 
import matplotlib.pyplot as plt
from PIL import Image
from helper import get_all_files

def show_random_images(num_images, dataset_folder_path, save_file):
    # Get all files in the folder
    file_details = get_all_files(dataset_folder_path)
    # Randomly select 'num_images' files
    selected_file_detailes = random.sample(file_details, num_images)
    # # Plot the images
    # fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    # for i, (filename, age) in enumerate(selected_file_detailes):
    #     image = Image.open(os.path.join(dataset_folder_path, f"{age}/{filename}"))
    #     axes[i].imshow(image)
    #     axes[i].axis('off')
    #     axes[i].set_title(f"file: {filename}\nage: {age}")
    # plt.show()

    plt.figure(figsize=(8, 8))
    for i, (filename, age) in enumerate(selected_file_detailes, 1):
        image = Image.open(os.path.join(dataset_folder_path, f"{age}/{filename}"))
        plt.subplot(3, 3, i)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"file: {filename}\nage: {age}")
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()

if __name__ == "__main__":
    dataset_folder_path = "/home/norakami/age-prediction/dataset/"
    save_simple_images = "/home/norakami/age-prediction/pics/sample_images.png"
    show_random_images(9, dataset_folder_path, save_simple_images)