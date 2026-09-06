"""Combine every model's metrics into one comparison table + bar chart per target.


Run from the `ml-service/` directory: python results/model_comparison.py


Reads only `models/metrics/*.json` (written by train.py's train_and_compare
and every models/train_*.py script) - never re-runs any training. Run this
last, after train.py, train_lightgbm.py, train_prophet.py, train_lstm.py, and
train_ensemble.py have all produced their metrics JSON files.


METRICS_DIR/SOLAR_TARGET/CONSUMPTION_TARGET are duplicated here (matching
train.py's values) rather than imported, so this script stays free of
train.py's heavier ML dependencies (xgboost, sklearn, etc.) - it only needs
pandas/matplotlib to combine already-computed results.
"""
import glob
import json
import os


import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = "models/metrics"
RESULTS_DIR = "results"
SOLAR_TARGET = "solar_generation_kwh"
CONSUMPTION_TARGET = "consumption_kwh"




def load_all_metrics():
    records = []
    for path in glob.glob(os.path.join(METRICS_DIR, "*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return pd.DataFrame(records).rename(
        columns={"model": "Model", "target": "Target", "rmse": "RMSE", "mse": "MSE", "mae": "MAE", "r2": "R2"}
    )




def collapse_household_benchmark_rows(df): # pyright: ignore[reportUnknownParameterType]
    """For the bar chart only: collapse the per-household Prophet/SARIMAX-
    fallback rows into one median-R2 representative row, so the chart isn't
    dominated by 3 near-duplicate bars. The full CSV keeps all rows as-is.
    """
    is_household_row = df["Model"].str.match(r"^(Prophet|SARIMAX_fallback)_hh")
    household_rows = df[is_household_row]
    other_rows = df[~is_household_row]
    if household_rows.empty:
        return df


    backend = "SARIMAX fallback" if household_rows["Model"].str.startswith("SARIMAX_fallback").any() else "Prophet"
    median_r2 = household_rows["R2"].median()
    median_row = household_rows.iloc[[(household_rows["R2"] - median_r2).abs().argsort().iloc[0]]].copy()
    median_row["Model"] = f"{backend} (median of {len(household_rows)} households)"
    return pd.concat([other_rows, median_row], ignore_index=True)




def plot_r2_bar(df, target, out_path):
    display_df = collapse_household_benchmark_rows(df).sort_values("R2", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(display_df["Model"], display_df["R2"])
    plt.xlabel("R2")
    plt.title(f"Model R2 comparison - {target}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()




def honesty_check(df, target):
    """Automated, not manually-written: compares the ensemble's R2 against the
    best non-ensemble model and prints a plain verdict either way.
    """
    if "Ensemble" not in df["Model"].values:
        print(f"[{target}] No Ensemble row found - skipping honesty check.")
        return
    ensemble_r2 = df.loc[df["Model"] == "Ensemble", "R2"].iloc[0]
    others = df[df["Model"] != "Ensemble"]
    best_idx = others["R2"].idxmax()
    best_other_r2 = others.loc[best_idx, "R2"]
    best_other_name = others.loc[best_idx, "Model"]
    diff = ensemble_r2 - best_other_r2
    if diff > 0:
        print(
            f"[{target}] Ensemble (R2={ensemble_r2:.4f}) IMPROVES on best "
            f"individual model {best_other_name} (R2={best_other_r2:.4f}) by +{diff:.4f}"
        )
    else:
        print(
            f"[{target}] Ensemble (R2={ensemble_r2:.4f}) does NOT beat best "
            f"individual model {best_other_name} (R2={best_other_r2:.4f}) - "
            "reporting honestly rather than forcing the result."
        )




def main():
    all_df = load_all_metrics()
    if all_df.empty:
        print(f"No metrics found in {METRICS_DIR}/ - run the training scripts first.")
        return


    os.makedirs(RESULTS_DIR, exist_ok=True)


    for target, out_prefix in [(SOLAR_TARGET, "solar"), (CONSUMPTION_TARGET, "consumption")]:
        target_df = (
            all_df[all_df["Target"] == target][["Model", "RMSE", "MSE", "MAE", "R2"]]
            .sort_values("R2", ascending=False)
            .reset_index(drop=True)
        )
        if target_df.empty:
            print(f"No metrics found for target {target} - skipping.")
            continue


        print(f"\n=== Full model comparison: {target} ===")
        print(target_df.to_string(index=False))


        target_df.to_csv(f"{RESULTS_DIR}/{out_prefix}_model_comparison.csv", index=False)
        plot_r2_bar(target_df, target, f"{RESULTS_DIR}/{out_prefix}_r2_comparison.png")
        honesty_check(target_df, target)


    print(f"\nSaved comparison CSVs and R2 bar charts to {RESULTS_DIR}/")




if __name__ == "__main__":
    main()



