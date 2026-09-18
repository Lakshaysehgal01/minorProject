"""Exploratory data analysis for the synthetic microgrid dataset.


Run from the `ml-service/` directory: python notebooks/eda.py
"""
import os


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = "data/energy_data.csv"
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


    # --- Line plot: monthly-average solar generation, consumption, and temperature ---
    monthly_avg = df.groupby("month")[
        ["solar_generation_kwh", "consumption_kwh", "temperature"]
    ].mean()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(monthly_avg.index, monthly_avg["solar_generation_kwh"], marker="o", label="Avg solar generation (kWh)")
    ax1.plot(monthly_avg.index, monthly_avg["consumption_kwh"], marker="o", label="Avg consumption (kWh)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("kWh")
    ax1.set_xticks(range(1, 13))
    ax2 = ax1.twinx()
    ax2.plot(monthly_avg.index, monthly_avg["temperature"], marker="s", color="gray", linestyle="--", label="Avg temperature (C)")
    ax2.set_ylabel("Temperature (C)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    plt.title("Monthly Seasonal Profile: Solar Generation, Consumption, and Temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "monthly_seasonal_profile.png"))
    plt.close()


    # --- Boxplot of consumption_kwh by household consumption profile tier ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="consumption_profile", y="consumption_kwh", order=["small", "medium", "large"])
    plt.xlabel("Consumption profile")
    plt.ylabel("Consumption (kWh)")
    plt.title("Consumption Distribution by Household Profile Tier")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "consumption_boxplot_by_profile.png"))
    plt.close()


    print(f"\nSaved plots to {PLOTS_DIR}/")




if __name__ == "__main__":
    main()



