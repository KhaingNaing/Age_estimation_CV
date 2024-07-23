import csv 
import os 
import pandas as pd
from helper import get_all_files, get_files_extension

# Create a csv file with the following columns: "image_id", "age"
def create_csv(dataset_folder_path, save_path):
    # get all images and their age values
    file_details = get_all_files(dataset_folder_path)

    header = ["image_id", "age"]
    with open(save_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for image_name, age in file_details:
            writer.writerow([image_name, age])

if __name__ == "__main__":
    folder_path = "/home/norakami/age-prediction/dataset/"
    csv_file_path = "/home/norakami/age-prediction/csv_dataset/age_dataset.csv"

    extensions = get_files_extension(folder_path)
    print("All file extensions", extensions)

    create_csv(folder_path, csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(f"Dataframe length: ",len(df), "\n")
    print("Top 5 rows in Dataframe:")
    print(df.head(5))
    print("Dataframe info:")
    print(df.info())