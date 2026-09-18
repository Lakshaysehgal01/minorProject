"""Generate synthetic hourly smart-microgrid data for 25 households over 365 days.


Solar generation is driven by pvlib panel physics fed with REAL Delhi 2024
weather (via fetch_weather.py); consumption is a behavioral model driven by
that same real weather plus an Indian public-holiday calendar. Both add
genuine outlier events (equipment faults, blackouts) - see README.md's
Dataset section for the full rationale (this replaces an earlier version
whose smooth, noise-free sinusoids made every model's R2 unrealistically
close to 1.0).
"""
import numpy as np
import pandas as pd
import pvlib
from holidays import India as IndiaHolidays


from fetch_weather import LATITUDE, LONGITUDE, TIMEZONE, WEATHER_YEAR, fetch_delhi_weather


SEED = 42
N_HOUSEHOLDS = 25
N_DAYS = 365
START_DATE = "2026-01-01"
ALTITUDE = 216  # meters, Delhi


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




def load_weather_for_timeline(n_hours):
    """Real WEATHER_YEAR Delhi weather, dropping Feb 29 (if present) so it's
    exactly `n_hours` long and lines up hour-for-hour, in chronological
    order, with the nominal (non-leap) N_DAYS-day timeline in build_timestamps().
    """
    weather = fetch_delhi_weather()
    weather = weather[~((weather["time"].dt.month == 2) & (weather["time"].dt.day == 29))]
    weather = weather.sort_values("time").reset_index(drop=True)
    if len(weather) != n_hours:
        raise ValueError(f"expected {n_hours} weather rows after dropping Feb 29, got {len(weather)}")
    return weather




def build_soiling_multiplier(day_of_year):
    """Dust accumulation on panels (very relevant in Delhi) - a slow decline
    reset by a random "cleaning" every ~15-30 days, with a random phase per
    household. Unexplained variance uncorrelated with instantaneous weather.
    """
    decay_rate = rng.uniform(0.001, 0.004)
    cleaning_interval = rng.integers(15, 31)
    phase = rng.integers(0, cleaning_interval)
    days_since_cleaning = (day_of_year - 1 + phase) % cleaning_interval
    return np.clip(1 - decay_rate * days_since_cleaning, 0.85, 1.0)




def apply_solar_outliers(generation_kwh, daylight_mask, day_of_year):
    """Extreme-but-present anomalies (no missing-data gaps): rare inverter
    faults (zero output), rare measurement-glitch spikes, and a couple of
    full-day equipment-outage days per household per year.
    """
    generation_kwh = generation_kwh.copy()
    n = len(generation_kwh)


    fault_zero_mask = daylight_mask & (rng.random(n) < 0.003)
    generation_kwh[fault_zero_mask] = 0.0


    spike_mask = daylight_mask & (rng.random(n) < 0.002)
    generation_kwh[spike_mask] *= rng.uniform(1.3, 1.8, size=spike_mask.sum())


    n_outage_days = rng.integers(1, 4)
    outage_days = rng.choice(np.arange(1, N_DAYS + 1), size=n_outage_days, replace=False)
    generation_kwh[np.isin(day_of_year, outage_days)] = 0.0


    return generation_kwh




def build_daily_ar1_noise(sigma=0.15, phi=0.6):
    """Per-household AR(1) daily "mood"/occupancy multiplier - correlated
    day-to-day (unlike i.i.d. hourly noise), so a day is more/less busy as a
    whole rather than every hour rolling an independent die.
    """
    daily = np.zeros(N_DAYS)
    for d in range(1, N_DAYS):
        daily[d] = phi * daily[d - 1] + rng.normal(0, sigma)
    return np.repeat(daily, 24)




def apply_consumption_outliers(consumption_kwh, temperature):
    """Extreme-but-present anomalies: blackout blocks biased toward the
    hottest hours (realistic grid strain during Delhi heat waves) and rare
    single-hour appliance-surge spikes.
    """
    consumption_kwh = consumption_kwh.copy()
    n = len(consumption_kwh)


    hot_threshold = np.percentile(temperature, 80)
    hot_hours = np.where(temperature > hot_threshold)[0]
    n_outages = min(rng.integers(2, 6), len(hot_hours))
    outage_starts = rng.choice(hot_hours, size=n_outages, replace=False)
    for start in outage_starts:
        end = min(start + rng.integers(2, 7), n)
        consumption_kwh[start:end] = 0.0


    spike_mask = rng.random(n) < 0.001
    consumption_kwh[spike_mask] += rng.uniform(1.0, 2.5, size=spike_mask.sum())


    return consumption_kwh




def generate():
    timestamps = build_timestamps()
    n_hours = len(timestamps)
    weather = load_weather_for_timeline(n_hours)


    # Calendar-derived arrays are identical across households - compute once.
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()  # Monday=0 ... Sunday=6
    day_of_year = timestamps.dayofyear.to_numpy()
    month = timestamps.month.to_numpy()
    is_weekend = day_of_week >= 5


    # Real Indian public holidays (computed for the real weather year so
    # festivals land on their true dates), reprojected onto the nominal
    # timeline the same (month, day) way as the weather itself.
    india_holidays = IndiaHolidays(years=WEATHER_YEAR)
    is_holiday = weather["time"].dt.date.map(lambda d: d in india_holidays).to_numpy()


    # Real hourly weather shared by all households (before per-household sensor jitter).
    temperature_shared = weather["temperature_2m"].to_numpy()
    cloud_cover_shared = weather["cloud_cover"].to_numpy()
    humidity_shared = weather["relative_humidity_2m"].to_numpy()
    wind_speed_shared = weather["wind_speed_10m"].to_numpy()
    ghi = np.clip(weather["shortwave_radiation"].to_numpy(), 0, None)
    dni = np.clip(weather["direct_normal_irradiance"].to_numpy(), 0, None)
    dhi = np.clip(weather["diffuse_radiation"].to_numpy(), 0, None)


    # Real solar geometry for the real weather timestamps (the DNI/GHI/DHI
    # values were measured against these actual sun positions, not the
    # nominal 2026 dates).
    location = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE, altitude=ALTITUDE)
    weather_times_local = weather["time"].dt.tz_localize(TIMEZONE)
    solar_position = location.get_solarposition(weather_times_local)
    solar_zenith = solar_position["apparent_zenith"].to_numpy()
    solar_azimuth = solar_position["azimuth"].to_numpy()
    daylight_mask = ghi > 5


    # Per-household fixed attributes (drawn once, broadcast across all rows).
    panel_capacity_by_household = rng.uniform(2.0, 12.0, size=N_HOUSEHOLDS)
    panel_tilt_by_household = rng.uniform(25.0, 31.0, size=N_HOUSEHOLDS)  # centered on Delhi's latitude
    panel_azimuth_by_household = rng.uniform(170.0, 190.0, size=N_HOUSEHOLDS)  # centered on true south
    temp_sensor_bias_by_household = rng.normal(0, 0.5, size=N_HOUSEHOLDS)  # fixed calibration quirk
    ac_factor_by_household = rng.uniform(0.03, 0.08, size=N_HOUSEHOLDS)  # cooling-load sensitivity


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


        # Per-household sensor readings: real shared weather + a fixed
        # calibration bias + small sensor noise.
        temperature = (
            temperature_shared + temp_sensor_bias_by_household[household_id] + rng.normal(0, 0.3, n_hours)
        )
        cloud_cover = np.clip(cloud_cover_shared + rng.normal(0, 2.0, n_hours), 0, 100)
        humidity = np.clip(humidity_shared + rng.normal(0, 2.0, n_hours), 0, 100)
        wind_speed = np.clip(wind_speed_shared + rng.normal(0, 0.3, n_hours), 0, None)


        panel_capacity_kw = np.full(n_hours, panel_capacity_by_household[household_id])


        # --- Solar generation: real irradiance -> pvlib panel physics ---
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=panel_tilt_by_household[household_id],
            surface_azimuth=panel_azimuth_by_household[household_id],
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
        )
        poa_global = np.clip(np.asarray(poa["poa_global"]), 0, None)
        cell_temperature = pvlib.temperature.faiman(poa_global, temperature, wind_speed)


        soiling = build_soiling_multiplier(day_of_year)
        degradation = 1 - 0.005 * (day_of_year / N_DAYS)  # ~0.5%/year panel degradation
        pdc0 = panel_capacity_by_household[household_id] * 1000 * soiling * degradation
        pdc = pvlib.pvsystem.pvwatts_dc(poa_global, cell_temperature, pdc0=pdc0, gamma_pdc=-0.004)


        # DC:AC ratio ~1.11 (inverter undersized vs panels) so sunny-day
        # output visibly clips instead of scaling linearly forever.
        inverter_pdc0 = panel_capacity_by_household[household_id] * 1000 * 0.9
        pac = np.clip(pvlib.inverter.pvwatts(pdc, pdc0=inverter_pdc0), 0, None)


        solar_generation_kwh = pac / 1000.0  # hourly W -> kWh
        solar_generation_kwh = np.where(daylight_mask, solar_generation_kwh, 0.0)
        solar_generation_kwh = apply_solar_outliers(solar_generation_kwh, daylight_mask, day_of_year)


        # Reference-only column (not fed to models - see train.py's
        # SOLAR_FEATURES) kept for EDA/debugging continuity.
        solar_irradiance = np.clip(poa_global / 1000.0, 0, 1.3)


        # --- Consumption: behavioral shape + real-weather-driven AC load ---
        morning_peak = peak_multiplier * 1.2 * np.exp(-((hour - 8) ** 2) / (2 * 1.5 ** 2))
        evening_peak = peak_multiplier * 1.8 * np.exp(-((hour - 19) ** 2) / (2 * 2.0 ** 2))
        weekend_daytime_bump = np.where(
            is_weekend,
            peak_multiplier * 0.6 * np.exp(-((hour - 14) ** 2) / (2 * 4.0 ** 2)),
            0.0,
        )
        holiday_evening_bump = np.where(
            is_holiday,
            peak_multiplier * 1.0 * np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2)),
            0.0,
        )
        # Humidity raises perceived heat (a rough heat-index adjustment)
        # before computing cooling-degree load - real AC use responds to
        # "feels like", not just dry-bulb temperature.
        feels_like = temperature + 0.03 * np.maximum(humidity - 40, 0)
        cooling_degree = np.maximum(feels_like - 24.0, 0)
        ac_term = ac_factor_by_household[household_id] * peak_multiplier * cooling_degree


        daily_mood = build_daily_ar1_noise()
        consumption_noise = rng.normal(0, 0.15, n_hours)


        consumption_kwh = np.clip(
            baseline_load_by_household[household_id]
            + morning_peak
            + evening_peak
            + weekend_daytime_bump
            + holiday_evening_bump
            + ac_term
            + daily_mood
            + consumption_noise,
            0,
            None,
        )
        consumption_kwh = apply_consumption_outliers(consumption_kwh, temperature)


        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "day_of_year": day_of_year,
                    "month": month,
                    "is_holiday": is_holiday.astype(int),
                    "household_id": household_id,
                    "consumption_profile": tier,
                    "consumption_profile_encoded": TIER_ENCODED[tier],
                    "temperature": temperature.round(2),
                    "cloud_cover": cloud_cover.round(1),
                    "humidity": humidity.round(1),
                    "wind_speed": wind_speed.round(2),
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
    out_path = "data/energy_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows for {N_HOUSEHOLDS} households over {N_DAYS} days.")
    print(f"Saved to {out_path}")
    print(df.head())



