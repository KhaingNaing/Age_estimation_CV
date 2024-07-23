import os 
import cv2
import numpy as np
from helper import get_all_files

# Function to check if an image is blurry
def is_blurry(image, threshold=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Return True if the variance is below the threshold
    return laplacian_var < threshold

# Function to check if an image has low resolution
def is_low_resolution(image, threshold=(224, 224)):
    # Get the image dimensions
    h, w = image.shape[:2]
    # Return True if the dimensions are below the threshold
    return h < threshold[0] or w < threshold[1]

def get_blurry_sharp_files(dataset_folder_path, threshold=100):
    # Get all files in the folder
    file_details = get_all_files(dataset_folder_path)
    # Initialize a list to store the blurry files
    blurry_files = []
    sharp_files = []
    # Iterate over the files
    for filename, age in file_details:
        # Load the image
        image = cv2.imread(os.path.join(dataset_folder_path, f"{age}/{filename}"))
        # Check if the image is blurry
        if is_blurry(image, threshold):
            blurry_files.append((filename, age))
        else:
            sharp_files.append((filename, age))
    return blurry_files, sharp_files
