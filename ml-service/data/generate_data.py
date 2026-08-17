"""Generate synthetic hourly smart-microgrid data for 10 households over 60 days."""
import numpy as np
import pandas as pd


SEED = 42
N_HOUSEHOLDS = 10
N_DAYS = 60
START_DATE = "2026-01-01"


rng = np.random.default_rng(SEED)




def build_timestamps():
    return pd.date_range(start=START_DATE, periods=N_DAYS * 24, freq="h")




def generate():
    timestamps = build_timestamps()
    n_hours = len(timestamps)


    # Per-household fixed attributes (drawn once, broadcast across all rows).
    panel_capacity_by_household = rng.uniform(3.0, 8.0, size=N_HOUSEHOLDS)
    baseline_load_by_household = rng.uniform(0.3, 0.8, size=N_HOUSEHOLDS)


    rows = []
    for household_id in range(N_HOUSEHOLDS):
        hour = timestamps.hour.to_numpy()
        day_of_week = timestamps.dayofweek.to_numpy()  # Monday=0 ... Sunday=6
        day_index = np.arange(n_hours) // 24


        # Temperature: daily sinusoid (cooler at night, peak mid-afternoon) + mild
        # seasonal drift over the 60-day window + noise.
        daily_cycle = -8 * np.cos(2 * np.pi * (hour - 3) / 24)
        seasonal_drift = 3 * np.sin(2 * np.pi * day_index / N_DAYS)
        temperature = 18 + daily_cycle + seasonal_drift + rng.normal(0, 1.0, n_hours)


        # Solar irradiance: bell curve peaking at 1pm, ~0 outside ~6am-7pm, scaled 0-1.
        bell = np.exp(-((hour - 13) ** 2) / (2 * 3.0 ** 2))
        daylight_mask = (hour >= 6) & (hour <= 19)
        solar_irradiance = np.clip(
            bell * daylight_mask + rng.normal(0, 0.02, n_hours), 0, 1
        )


        panel_capacity_kw = np.full(n_hours, panel_capacity_by_household[household_id])


        # Solar generation scaled by panel capacity and irradiance, plus small noise,
        # clipped at 0 so nights stay exactly 0.
        solar_noise = rng.normal(0, 0.05, n_hours)
        solar_generation_kwh = np.clip(
            panel_capacity_kw * solar_irradiance + solar_noise, 0, None
        )
        solar_generation_kwh = np.where(daylight_mask, solar_generation_kwh, 0.0)


        # Consumption: baseline + morning peak (~8am) + evening peak (~7pm),
        # weekends get an extra daytime bump, plus per-household offset + noise.
        morning_peak = 1.2 * np.exp(-((hour - 8) ** 2) / (2 * 1.5 ** 2))
        evening_peak = 1.8 * np.exp(-((hour - 19) ** 2) / (2 * 2.0 ** 2))
        is_weekend = day_of_week >= 5
        weekend_daytime_bump = np.where(
            is_weekend, 0.6 * np.exp(-((hour - 14) ** 2) / (2 * 4.0 ** 2)), 0.0
        )
        consumption_noise = rng.normal(0, 0.15, n_hours)
        consumption_kwh = np.clip(
            baseline_load_by_household[household_id]
            + morning_peak
            + evening_peak
            + weekend_daytime_bump
            + consumption_noise,
            0,
            None,
        )


        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "household_id": household_id,
                    "temperature": temperature.round(2),
                    "solar_irradiance": solar_irradiance.round(4),
                    "panel_capacity_kw": panel_capacity_kw.round(2),
                    "solar_generation_kwh": solar_generation_kwh.round(4),
                    "consumption_kwh": consumption_kwh.round(4),
                }
            )
        )


    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["timestamp", "household_id"]).reset_index(drop=True)
    return df




if __name__ == "__main__":
    df = generate()
    out_path = "data/synthetic_energy_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows for {N_HOUSEHOLDS} households over {N_DAYS} days.")
    print(f"Saved to {out_path}")
    print(df.head())



