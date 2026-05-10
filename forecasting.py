import pandas as pd
import numpy as np
from prophet import Prophet
import json
import warnings
warnings.filterwarnings('ignore')

# ============ LOAD DATA ============
print("Loading processed data...")
df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)
df = df.dropna()

stock_cols = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]

forecast_results = {}

# ============ PROPHET FORECASTING ============
for stock in stock_cols:
    print(f"\n📈 Forecasting {stock}...")

    # Prophet needs columns named 'ds' and 'y'
    prophet_df = df[[stock, "avg_temp"]].reset_index()
    prophet_df.columns = ["ds", "y", "avg_temp"]

    # Remove timezone if any
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    # Initialize Prophet with temperature as extra regressor
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )

    # Add temperature as additional regressor
    model.add_regressor("avg_temp")

    # Fit model
    model.fit(prophet_df)

    # Create future dataframe — forecast 90 days ahead
    future = model.make_future_dataframe(periods=90)

    # Fill future temperature with average of last 30 days
    last_avg_temp = prophet_df["avg_temp"].tail(30).mean()
    future["avg_temp"] = last_avg_temp

    # Predict
    forecast = model.predict(future)

    # Get last 90 days forecast
    future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(90)

    forecast_results[stock] = {
        "dates": future_forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "predicted": future_forecast["yhat"].round(2).tolist(),
        "lower": future_forecast["yhat_lower"].round(2).tolist(),
        "upper": future_forecast["yhat_upper"].round(2).tolist()
    }

    print(f"✅ {stock} forecast complete!")
    print(f"   Current price: ₹{prophet_df['y'].iloc[-1]:.2f}")
    print(f"   90-day forecast: ₹{future_forecast['yhat'].iloc[-1]:.2f}")

# ============ SAVE FORECAST RESULTS ============
with open("forecast_results.json", "w") as f:
    json.dump(forecast_results, f, indent=2)

print("\n✅ Forecast results saved to forecast_results.json")