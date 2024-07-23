import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Plot the distribution of the age values
def plot_age_distribution(df, save_path=None):
    # Set style and palette
    sns.set_style("whitegrid")
    sns.set_palette("inferno")

    # Create the histogram with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(df["age"], kde=True, color="g", stat="density")

    # Customize the plot
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Distribution of Age Values in the Dataset")

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()

def get_age_count_table(df, save_path="/home/norakami/age-prediction/csv_dataset/age_count_table.csv"):
    # get the age value counts
    age_counts = df["age"].value_counts()
    # convert to a dataframe
    age_counts = age_counts.reset_index()
    age_counts.columns = ["age", "count"]
    age_counts.sort_values(by="age", inplace=True)

    # save to csv
    age_counts.to_csv(save_path, index=False)
    print("Age count table saved to age_count_table.csv")


# Create datagenerator obj to load images, this can avoid loading all images at once and save memory
def create_datagenerator(df, dest_folder, batch_size=32, target_size=(224, 224), class_mode="raw"):
    datagen = ImageDataGenerator(
        # image rescaling (normalizes the pixel values to be between 0 and 1)
        rescale=1./255
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=dest_folder,
        x_col="image_id",
        y_col="age",
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        seed=42
    )

    return train_generator

if __name__ == "__main__":
    fig_save_path = "/home/norakami/age-prediction/figs/age_distribution.png"
    df = pd.read_csv("/home/norakami/age-prediction/csv_dataset/age_dataset.csv")

    get_age_count_table(df)
    plot_age_distribution(df, fig_save_path)


