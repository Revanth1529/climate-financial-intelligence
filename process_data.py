import pandas as pd
import numpy as np

# ============ LOAD DATA ============
print("Loading data...")

stock_df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
temp_df = pd.read_csv("temperature_data.csv", index_col=0, parse_dates=True)

# ============ FLATTEN MULTI-LEVEL COLUMNS IF ANY ============
if isinstance(stock_df.columns, pd.MultiIndex):
    stock_df.columns = stock_df.columns.get_level_values(0)

# ============ ALIGN BOTH DATASETS BY DATE ============
print("Merging datasets...")

# Remove timezone info if any
stock_df.index = stock_df.index.tz_localize(None)
temp_df.index = temp_df.index.tz_localize(None)

# Merge on common dates
merged = pd.merge(stock_df, temp_df, left_index=True, right_index=True, how='inner')

print(f"Merged dataset shape: {merged.shape}")
print(f"Date range: {merged.index.min()} to {merged.index.max()}")

# ============ COMPUTE AVERAGE TEMPERATURE ============
city_cols = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]
merged["avg_temp"] = merged[city_cols].mean(axis=1)

# ============ COMPUTE COOLING DEGREE DAYS (CDD) ============
# CDD = max(0, temp - 18)
# 18°C is base comfort temperature
BASE_TEMP = 18
merged["CDD"] = merged["avg_temp"].apply(lambda x: max(0, x - BASE_TEMP))

print("CDD computed ✅")

# ============ DETECT HEAT SHOCK EVENTS ============
# Heat shock = day where avg temp > 40°C
HEAT_THRESHOLD = 40
merged["heat_shock"] = merged["Delhi"] > HEAT_THRESHOLD

heat_shock_days = merged[merged["heat_shock"]].shape[0]
print(f"Heat shock days detected: {heat_shock_days} ✅")

# ============ COMPUTE STOCK DAILY RETURNS ============
stock_cols = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]

for col in stock_cols:
    merged[f"{col}_return"] = merged[col].pct_change() * 100

print("Daily returns computed ✅")

# ============ COMPUTE STOCK VOLATILITY (7 DAY ROLLING) ============
for col in stock_cols:
    merged[f"{col}_volatility"] = merged[f"{col}_return"].rolling(7).std()

print("Volatility computed ✅")

# ============ SAVE PROCESSED DATA ============
merged.to_csv("processed_data.csv")
print("Processed data saved to processed_data.csv ✅")
print("\nSample data:")
print(merged[["avg_temp", "CDD", "heat_shock", "NTPC", "NTPC_return"]].head(10))