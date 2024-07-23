import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import config

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

if __name__ == "__main__":
    fig_save_path = "/home/norakami/age-prediction/figs"
    csv_path = config["csv_path"]

    for file in ["train_set.csv", "valid_set.csv", "test_set.csv"]:
        df = pd.read_csv(f"{csv_path}/{file}")
        plot_age_distribution(df, f"{fig_save_path}/{file.split('.')[0]}_age_distribution.png")


