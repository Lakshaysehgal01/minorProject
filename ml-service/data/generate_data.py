"""Generate synthetic hourly smart-microgrid data for 25 households over 365 days."""
import numpy as np
import pandas as pd


SEED = 42
N_HOUSEHOLDS = 25
N_DAYS = 365
START_DATE = "2026-01-01"


rng = np.random.default_rng(SEED)


# Consumption profile tiers: 25 households split 10 small / 10 medium / 5 large.
# "Medium" (multiplier 1.0) exactly reproduces the original single-tier constants.
TIER_COUNTS = {"small": 10, "medium": 10, "large": 5}
TIER_ENCODED = {"small": 0, "medium": 1, "large": 2}
TIER_BASELINE_LOW = {"small": 0.2, "medium": 0.4, "large": 0.7}
TIER_BASELINE_HIGH = {"small": 0.4, "medium": 0.7, "large": 1.2}
TIER_PEAK_MULTIPLIER = {"small": 0.7, "medium": 1.0, "large": 1.5}




def build_timestamps():
    return pd.date_range(start=START_DATE, periods=N_DAYS * 24, freq="h")




def assign_household_tiers():
    """Shuffle tier labels across household ids so id isn't a trivial proxy for tier."""
    tiers = np.array(
        sum(([tier] * count for tier, count in TIER_COUNTS.items()), [])
    )
    return rng.permutation(tiers)




def generate():
    timestamps = build_timestamps()
    n_hours = len(timestamps)


    # Calendar-derived arrays are identical across households - compute once.
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()  # Monday=0 ... Sunday=6
    day_of_year = timestamps.dayofyear.to_numpy()
    month = timestamps.month.to_numpy()
    is_weekend = day_of_week >= 5


    # Annual seasonal driver: +1 at day 172 (Jun 21 solstice), -1 at day 355 (Dec 21).
    # Shared by temperature and solar irradiance so both peak/trough in the same season.
    seasonal_factor = np.cos(2 * np.pi * (day_of_year - 172) / 365)


    # Temperature: daily sinusoid (cooler at night, peak mid-afternoon) + a genuine
    # annual seasonal swing (+-10C) + noise.
    daily_cycle = -8 * np.cos(2 * np.pi * (hour - 3) / 24)
    seasonal_drift = 10 * seasonal_factor
    temperature_base = 18 + daily_cycle + seasonal_drift


    # Solar irradiance: bell curve peaking at 1pm, with both the daylight window and
    # the peak intensity varying seasonally (shorter/dimmer in winter, longer/brighter
    # in summer). sunrise in [5.0, 8.0], sunset in [16.0, 21.0] - 8h margin, so the
    # window can never invert. peak_intensity_factor in [0.6, 1.0].
    sunrise_hour = 6.5 - 1.5 * seasonal_factor
    sunset_hour = 18.5 + 2.5 * seasonal_factor
    daylight_mask = (hour >= sunrise_hour) & (hour <= sunset_hour)
    peak_intensity_factor = 0.8 + 0.2 * seasonal_factor
    bell_base = peak_intensity_factor * np.exp(-((hour - 13) ** 2) / (2 * 3.0 ** 2))


    # Per-household fixed attributes (drawn once, broadcast across all rows).
    panel_capacity_by_household = rng.uniform(2.0, 12.0, size=N_HOUSEHOLDS)
    household_tiers = assign_household_tiers()
    baseline_low = np.array([TIER_BASELINE_LOW[t] for t in household_tiers])
    baseline_high = np.array([TIER_BASELINE_HIGH[t] for t in household_tiers])
    baseline_load_by_household = rng.uniform(baseline_low, baseline_high)
    peak_multiplier_by_household = np.array(
        [TIER_PEAK_MULTIPLIER[t] for t in household_tiers]
    )


    rows = []
    for household_id in range(N_HOUSEHOLDS):
        tier = household_tiers[household_id]
        peak_multiplier = peak_multiplier_by_household[household_id]


        temperature = temperature_base + rng.normal(0, 1.0, n_hours)
        solar_irradiance = np.clip(
            bell_base * daylight_mask + rng.normal(0, 0.02, n_hours), 0, 1
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
        # weekends get an extra daytime bump, all scaled by the household's
        # consumption-profile tier, plus noise.
        morning_peak = peak_multiplier * 1.2 * np.exp(-((hour - 8) ** 2) / (2 * 1.5 ** 2))
        evening_peak = peak_multiplier * 1.8 * np.exp(-((hour - 19) ** 2) / (2 * 2.0 ** 2))
        weekend_daytime_bump = np.where(
            is_weekend,
            peak_multiplier * 0.6 * np.exp(-((hour - 14) ** 2) / (2 * 4.0 ** 2)),
            0.0,
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
                    "day_of_year": day_of_year,
                    "month": month,
                    "household_id": household_id,
                    "consumption_profile": tier,
                    "consumption_profile_encoded": TIER_ENCODED[tier],
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



