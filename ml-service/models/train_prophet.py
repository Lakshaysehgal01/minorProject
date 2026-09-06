"""Train Prophet models as a per-household classical time-series benchmark.


Run from the `ml-service/` directory: python models/train_prophet.py


Prophet is univariate per series, so unlike the pooled tabular/LSTM models it
can't cover all 25 households at once. Instead we pick one representative
household per consumption-profile tier (small/medium/large) and report one
row per household, per target (6 rows total) - not averaged, since averaging
RMSE/R2 across tiers of very different scale would hide real per-tier
differences and doesn't correspond to any single deployable model.


Deliberately univariate (ds/y only, no add_regressor for temperature or
solar_irradiance): handing Prophet the same causal drivers XGBoost gets would
make it trivially near-perfect for solar and defeat the point of the
comparison. This way Prophet represents "seasonality decomposition of the raw
series" as an honest contrast to "gradient boosting on causal features".


With only 1 year of hourly history, Prophet's yearly seasonality term is fit
on a single observed cycle (not validated across multiple years) - treat it
as a demonstration, not a rigorously validated seasonal model.


If Prophet fails to install or its first fit fails (e.g. the corporate
network blocks cmdstanpy's one-time Stan compiler download), this script
falls back to a SARIMAX(order=(1,0,1), seasonal_order=(1,0,1,24)) model -
daily seasonality only, since a 8760-hour yearly seasonal term is both
computationally impractical and statistically under-identified with just one
year of training data. That row is labeled "SARIMAX_fallback_..." so it's
clear a substitute was used.
"""
import os
import sys


import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import (  # noqa: E402
    CONSUMPTION_TARGET,
    DATA_PATH,
    SOLAR_TARGET,
    evaluate,
    time_based_split,
    write_metrics,
)


# First household_id (ascending) in each consumption_profile tier - deterministic,
# not randomly picked. Verified against the current CSV; the assertion in main()
# re-derives this at runtime so a silent drift (e.g. data regenerated with a
# different seed) fails loudly instead of quietly reporting the wrong households.
PROPHET_HOUSEHOLDS = {"small": 4, "medium": 0, "large": 3}




def get_backend():
    """Returns "prophet" or "statsmodels", after actually testing Prophet works."""
    try:
        from prophet import Prophet


        # Smoke-test fit forces cmdstanpy's first-use Stan compile to happen now,
        # not mid-loop on the first real household - and a plain ImportError catch
        # wouldn't catch a fit()-time failure (e.g. blocked Stan compiler download).
        smoke_df = pd.DataFrame(
            {"ds": pd.date_range("2026-01-01", periods=48, freq="h"), "y": range(48)}
        )
        Prophet(daily_seasonality=True).fit(smoke_df)
        return "prophet"
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(
            f"[train_prophet] Prophet unavailable ({exc!r}); "
            "falling back to statsmodels SARIMAX."
        )
        return "statsmodels"




def _fit_predict_prophet(train_df, test_df, target):
    from prophet import Prophet


    train_prophet_df = train_df[["timestamp", target]].rename(
        columns={"timestamp": "ds", target: "y"}
    )
    test_prophet_df = test_df[["timestamp"]].rename(columns={"timestamp": "ds"})


    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(train_prophet_df)
    forecast = model.predict(test_prophet_df)
    return forecast["yhat"].to_numpy()




def _fit_predict_sarimax(train_df, test_df, target):
    from statsmodels.tsa.statespace.sarimax import SARIMAX


    train_series = train_df.set_index("timestamp")[target]
    model = SARIMAX(
        train_series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.forecast(steps=len(test_df)).to_numpy()




def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


    actual_mapping = {
        tier: int(sorted(df.loc[df["consumption_profile"] == tier, "household_id"].unique())[0])
        for tier in ["small", "medium", "large"]
    }
    assert actual_mapping == PROPHET_HOUSEHOLDS, (
        f"Prophet household mapping drifted: expected {PROPHET_HOUSEHOLDS}, "
        f"got {actual_mapping}. Update the PROPHET_HOUSEHOLDS constant if "
        "generate_data.py's seed/tier logic changed."
    )


    backend = get_backend()
    print(f"[train_prophet] Using backend: {backend}")


    for tier, household_id in PROPHET_HOUSEHOLDS.items():
        hh_df = (
            df[df["household_id"] == household_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        for target in [SOLAR_TARGET, CONSUMPTION_TARGET]:
            train_df, test_df = time_based_split(hh_df)


            if backend == "prophet":
                preds = _fit_predict_prophet(train_df, test_df, target)
                model_name = f"Prophet_hh{household_id}_{tier}"
            else:
                preds = _fit_predict_sarimax(train_df, test_df, target)
                model_name = f"SARIMAX_fallback_hh{household_id}_{tier}"


            metrics = evaluate(test_df[target].to_numpy(), preds)
            write_metrics(target, model_name, metrics)
            print(
                f"{model_name} ({target}): RMSE={metrics['RMSE']:.4f} "
                f"MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}"
            )




if __name__ == "__main__":
    main()



