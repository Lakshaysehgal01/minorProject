"""Fetch and cache one real year of hourly Delhi weather from Open-Meteo.


Run from the `ml-service/` directory: python data/fetch_weather.py
(generate_data.py also calls this automatically if the cache is missing.)


Uses the free, keyless Open-Meteo Historical Weather API
(https://archive-api.open-meteo.com/v1/archive) - no account, no API key,
no meaningful rate limit for a one-time fetch. This is what makes the
synthetic dataset's weather genuinely noisy (real cloudy days, the real
2024 Delhi heat wave, real monsoon cloud cover) instead of a smooth,
noise-free sinusoid.


NOTE: on a network with TLS-inspecting corporate proxies, this request can
fail with `SSLCertVerificationError` because the proxy's certificate isn't
in Python's trust store. If that happens, run this script once from an
unrestricted network (home network, or the separate training machine) and
copy the resulting `data/weather_cache/delhi_weather_2024.csv` file over -
generate_data.py only needs the cached CSV to exist, it never re-fetches
once that file is present.
"""
import os


import pandas as pd
import requests


LATITUDE = 28.6139
LONGITUDE = 77.2090
TIMEZONE = "Asia/Kolkata"
WEATHER_YEAR = 2024  # most recent complete calendar year at generator design time


CACHE_DIR = "data/weather_cache"
CACHE_PATH = f"{CACHE_DIR}/delhi_weather_{WEATHER_YEAR}.csv"


HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "precipitation",
]




def fetch_delhi_weather(year=WEATHER_YEAR, cache_path=CACHE_PATH):
    """Return a DataFrame of real hourly Delhi weather for `year`, using
    (and populating) a local CSV cache so this never needs to hit the
    network more than once.
    """
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["time"])


    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": TIMEZONE,
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "time"})
    df["time"] = pd.to_datetime(df["time"])


    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Fetched {len(df)} hourly rows of {year} Delhi weather -> {cache_path}")
    return df




if __name__ == "__main__":
    fetch_delhi_weather()



