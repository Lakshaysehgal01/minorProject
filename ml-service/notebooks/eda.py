"""Exploratory data analysis for the synthetic microgrid dataset.


Run from the `ml-service/` directory: python notebooks/eda.py
"""
import os


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = "data/synthetic_energy_data.csv"
PLOTS_DIR = "notebooks/plots"




def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


    print("=== Missing value check ===")
    print(df.isnull().sum())


    print("\n=== df.describe() ===")
    print(df.describe())


    # --- Line plot: avg solar_generation_kwh and consumption_kwh by hour ---
    hourly_avg = df.groupby("hour")[["solar_generation_kwh", "consumption_kwh"]].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg.index, hourly_avg["solar_generation_kwh"], marker="o", label="Avg solar generation (kWh)")
    plt.plot(hourly_avg.index, hourly_avg["consumption_kwh"], marker="o", label="Avg consumption (kWh)")
    plt.xlabel("Hour of day")
    plt.ylabel("kWh")
    plt.title("Average Solar Generation and Consumption by Hour")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hourly_profile.png"))
    plt.close()


    # --- Correlation heatmap of numeric features ---
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()


    # --- Boxplot of consumption_kwh by hour ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="hour", y="consumption_kwh")
    plt.xlabel("Hour of day")
    plt.ylabel("Consumption (kWh)")
    plt.title("Consumption Distribution by Hour")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "consumption_boxplot_by_hour.png"))
    plt.close()


    print(f"\nSaved plots to {PLOTS_DIR}/")




if __name__ == "__main__":
    main()



