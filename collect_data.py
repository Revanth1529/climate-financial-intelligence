import yfinance as yf
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime

# ============ STOCK DATA ============
print("Fetching stock data...")

stocks = {
    "NTPC": "NTPC.NS",
    "Tata Power": "TATAPOWER.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Reliance": "RELIANCE.NS",
    "Power Grid": "POWERGRID.NS"
}

all_stocks = []

for name, ticker in stocks.items():
    df = yf.download(ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=True)
    df = df[['Close']].copy()
    df.columns = [name]
    all_stocks.append(df)

stock_df = pd.concat(all_stocks, axis=1)
stock_df.index = pd.to_datetime(stock_df.index)
stock_df.to_csv("stock_data.csv")
print("Stock data saved to stock_data.csv ✅")

# ============ TEMPERATURE DATA ============
print("Fetching temperature data...")

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867)
}

all_temps = []

for city, (lat, lon) in cities.items():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2020-01-01",
        "end_date": datetime.today().strftime('%Y-%m-%d'),
        "daily": "temperature_2m_max",
        "timezone": "Asia/Kolkata"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    temp_max = daily.Variables(0).ValuesAsNumpy()

    import numpy as np
    dates = pd.date_range(
        start=pd.Timestamp(daily.Time(), unit="s", tz="Asia/Kolkata"),
        end=pd.Timestamp(daily.TimeEnd(), unit="s", tz="Asia/Kolkata"),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )

    temp_df = pd.DataFrame({"Date": dates, city: temp_max})
    temp_df.set_index("Date", inplace=True)
    all_temps.append(temp_df)

temp_final = pd.concat(all_temps, axis=1)
temp_final.index = temp_final.index.tz_localize(None)
temp_final.to_csv("temperature_data.csv")
print("Temperature data saved to temperature_data.csv ✅")