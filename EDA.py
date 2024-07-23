import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of the age values
def plot_age_distribution(df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_palette("inferno")
    sns.distplot(df["age"], kde=True)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Distribution of Age Values in the Dataset")
    plt.show()
    plt.savefig(save_path)

if __name__ == "__main__":
    fig_save_path = "/home/norakami/age-prediction/figs/age_distribution.png"
    df = pd.read_csv("csv_dataset/age_dataset.csv")
    print(df.head(5))
    plot_age_distribution(df, fig_save_path)


