import pandas as pd
import numpy as np
from scipy import stats
import json

# ============ LOAD PROCESSED DATA ============
print("Loading processed data...")
df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)

# Drop NaN rows
df = df.dropna()

stock_cols = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]
return_cols = [f"{col}_return" for col in stock_cols]
volatility_cols = [f"{col}_volatility" for col in stock_cols]

# ============ CORRELATION ANALYSIS ============
print("\n📊 CORRELATION — Temperature vs Stock Returns:")
correlations = {}

for col in return_cols:
    corr, pvalue = stats.pearsonr(df["avg_temp"], df[col])
    stock_name = col.replace("_return", "")
    correlations[stock_name] = round(corr, 4)
    significance = "✅ Significant" if pvalue < 0.05 else "❌ Not Significant"
    print(f"{stock_name}: correlation={round(corr,4)}, p-value={round(pvalue,4)} {significance}")

# ============ REGRESSION ANALYSIS ============
print("\n📊 REGRESSION — Temperature vs Stock Volatility:")
regression_results = {}

for col in volatility_cols:
    stock_name = col.replace("_volatility", "")
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["avg_temp"], df[col])
    regression_results[stock_name] = {
        "slope": round(slope, 4),
        "r_squared": round(r_value**2, 4)
    }
    print(f"{stock_name}: slope={round(slope,4)}, R²={round(r_value**2,4)}")

# ============ HEAT SENSITIVITY INDEX ============
print("\n📊 HEAT SENSITIVITY INDEX:")
heat_sensitivity = {}

for col in return_cols:
    stock_name = col.replace("_return", "")
    
    # Average return on heat shock days vs normal days
    heat_days = df[df["heat_shock"]][col].mean()
    normal_days = df[~df["heat_shock"]][col].mean()
    sensitivity = round(heat_days - normal_days, 4)
    heat_sensitivity[stock_name] = sensitivity
    direction = "📈 UP" if sensitivity > 0 else "📉 DOWN"
    print(f"{stock_name}: {sensitivity}% {direction} on heat shock days")

# ============ TEMPERATURE TIPPING POINT ============
print("\n📊 TEMPERATURE TIPPING POINT:")
temp_ranges = range(30, 46)
avg_volatility = []

for temp in temp_ranges:
    subset = df[df["avg_temp"] >= temp]
    if len(subset) > 10:
        vol = subset[[f"{col}_volatility" for col in stock_cols]].mean().mean()
        avg_volatility.append((temp, round(vol, 4)))

# Find tipping point — biggest jump in volatility
tipping_point = None
max_jump = 0

for i in range(1, len(avg_volatility)):
    jump = avg_volatility[i][1] - avg_volatility[i-1][1]
    if jump > max_jump:
        max_jump = jump
        tipping_point = avg_volatility[i][0]

print(f"Temperature Tipping Point: {tipping_point}°C")
print(f"(Above this temperature, stock volatility increases significantly)")

# ============ SAVE RESULTS ============
results = {
    "correlations": correlations,
    "regression": regression_results,
    "heat_sensitivity": heat_sensitivity,
    "tipping_point": tipping_point,
    "temp_volatility": avg_volatility
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Analysis results saved to analysis_results.json")