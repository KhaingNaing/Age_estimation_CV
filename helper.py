import os

# Get all the files in the folder and their associated age values
def get_all_files(dataset_path):
    age_folders = os.listdir(dataset_path)
    file_details = []
    for age in age_folders:
        folder_path = os.path.join(dataset_path, age)
        if os.path.isdir(folder_path):
            folder_files = os.listdir(folder_path)
            folder_files = [file for file in folder_files if file and not file.startswith('.')]

            for file_name in folder_files:
                file_details.append((file_name, age))

    return file_details

# Find all files' extensions in the current directory
def get_files_extension(dataset_folder_path):
    file_details = get_all_files(dataset_folder_path)
    files = [file[0] for file in file_details]
    extensions = [os.path.splitext(filename)[1] for filename in files]
    return set(extensions)


