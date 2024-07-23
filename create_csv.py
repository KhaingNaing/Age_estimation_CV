import csv 
import os 
import pandas as pd

# Find all files' extensions in the current directory
def get_files_extension(dataset_folder_path):
    age_folders = os.listdir(dataset_folder_path)
    for age_folder in age_folders:
        if os.path.isdir(dataset_folder_path + age_folder):
            sub_folder_path = os.path.join(dataset_folder_path, age_folder)
            extensions = [os.path.splitext(filename)[1] for filename in os.listdir(sub_folder_path)]
    return set(extensions)

# Create a csv file with the following columns: "image_id", "age"
def create_csv(dataset_folder_path, save_path):
    # Get all subfolders in the dataset folder
    sub_folders = os.listdir(dataset_folder_path)
    # Get all subfolders that are directories
    age_folders = [sub_folder for sub_folder in sub_folders if os.path.isdir(os.path.join(dataset_folder_path, sub_folder))]
    header = ["image_id", "age"]
    with open(save_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for age_folder in age_folders:
            img_files = os.listdir(os.path.join(dataset_folder_path, age_folder))
            for img in img_files:
                data = [img, age_folder]
                writer.writerow(data)

if __name__ == "__main__":
    folder_path = "/home/norakami/age-prediction/dataset/"
    csv_file_path = "/home/norakami/age-prediction/csv_dataset/age_dataset.csv"

    extensions = get_files_extension(folder_path)
    print(extensions)

    create_csv(folder_path, csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(f"Dataframe length: ",len(df), "\n")
    print(f"Age Values in Dataframe: ",df["age"].unique(), "\n")
    print("Top 5 rows in Dataframe:")
    print(df.head(5))