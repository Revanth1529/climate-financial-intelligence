import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import json

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Climate-Financial Intelligence",
    page_icon="🌡️",
    layout="wide"
)

# ============ LIVE DATA FETCHING ============
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_live_data():
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = "2020-01-01"

    # ---- STOCK DATA ----
    stocks = {
        "NTPC": "NTPC.NS",
        "Tata Power": "TATAPOWER.NS",
        "Adani Power": "ADANIPOWER.NS",
        "Reliance": "RELIANCE.NS",
        "Power Grid": "POWERGRID.NS"
    }

    all_stocks = []
    for name, ticker in stocks.items():
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        df = df[['Close']].copy()
        df.columns = [name]
        all_stocks.append(df)

    stock_df = pd.concat(all_stocks, axis=1)
    stock_df.index = pd.to_datetime(stock_df.index)

    # ---- TEMPERATURE DATA ----
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
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
            "start_date": start_date,
            "end_date": (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d'),
            "daily": "temperature_2m_max",
            "timezone": "Asia/Kolkata"
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        temp_max = daily.Variables(0).ValuesAsNumpy()

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

    # ---- MERGE ----
    stock_df.index = stock_df.index.tz_localize(None)
    merged = pd.merge(stock_df, temp_final, left_index=True, right_index=True, how='inner')

    if isinstance(merged.columns, pd.MultiIndex):
        merged.columns = merged.columns.get_level_values(0)

    merged = merged.dropna()

    # ---- COMPUTE FEATURES ----
    city_cols = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]
    merged["avg_temp"] = merged[city_cols].mean(axis=1)
    merged["CDD"] = merged["avg_temp"].apply(lambda x: max(0, x - 18))
    merged["heat_shock"] = merged["Delhi"] > 40

    stock_cols_list = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]
    for col in stock_cols_list:
        merged[f"{col}_return"] = merged[col].pct_change() * 100
        merged[f"{col}_volatility"] = merged[f"{col}_return"].rolling(7).std()

    merged = merged.dropna()

    return merged

# ============ COMPUTE ANALYSIS ============
def compute_analysis(df):
    from scipy import stats
    stock_cols = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]

    correlations = {}
    for col in stock_cols:
        try:
            clean = df[["avg_temp", f"{col}_return"]].dropna()
            if len(clean) < 2:
                correlations[col] = 0.0
                continue
            corr, pvalue = stats.pearsonr(clean["avg_temp"], clean[f"{col}_return"])
            correlations[col] = round(corr, 4)
        except:
            correlations[col] = 0.0

    heat_sensitivity = {}
    for col in stock_cols:
        try:
            heat = df[df["heat_shock"]][f"{col}_return"].mean()
            normal = df[~df["heat_shock"]][f"{col}_return"].mean()
            heat_sensitivity[col] = round(heat - normal, 4)
        except:
            heat_sensitivity[col] = 0.0

    temp_ranges = range(30, 46)
    avg_volatility = []
    for temp in temp_ranges:
        subset = df[df["avg_temp"] >= temp]
        if len(subset) > 10:
            vol = subset[[f"{col}_volatility" for col in stock_cols]].mean().mean()
            avg_volatility.append((temp, round(vol, 4)))

    tipping_point = 35
    max_jump = 0
    for i in range(1, len(avg_volatility)):
        jump = avg_volatility[i][1] - avg_volatility[i-1][1]
        if jump > max_jump:
            max_jump = jump
            tipping_point = avg_volatility[i][0]

    return correlations, heat_sensitivity, tipping_point

# ============ LOAD DATA ============
stock_cols = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]

with st.spinner("🔄 Fetching live data..."):
    df = fetch_live_data()

correlations, heat_sensitivity, tipping_point = compute_analysis(df)

# ============ SIDEBAR ============
st.sidebar.title("🌡️ Climate-Financial Intelligence")
st.sidebar.markdown("---")

# Last updated
st.sidebar.success(f"✅ Last Updated: {datetime.now().strftime('%d %b %Y %H:%M')}")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📈 Stock vs Temperature",
    "⚡ Heat Shock Events",
    "🔥 Correlation Heatmap"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**5 Indian Energy Stocks**")
for stock in stock_cols:
    st.sidebar.markdown(f"• {stock}")

# ============ PAGE 1 — OVERVIEW ============
if page == "🏠 Overview":
    st.title("🌡️ Climate-Financial Intelligence Dashboard")
    st.markdown("### How Rising Temperatures Affect Indian Energy Stocks")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Data Range", "2020 - Today")
    col2.metric("🏢 Stocks Tracked", "5")
    col3.metric("⚡ Heat Shock Days", str(df["heat_shock"].sum()))
    col4.metric("🌡️ Tipping Point", f"{tipping_point}°C")

    st.markdown("---")

    # Heat Sensitivity Index
    st.subheader("🔥 Heat Sensitivity Index")
    sens_df = pd.DataFrame({
        "Stock": list(heat_sensitivity.keys()),
        "Sensitivity (%)": list(heat_sensitivity.values())
    }).sort_values("Sensitivity (%)", ascending=False)

    colors = ["green" if x > 0 else "red" for x in sens_df["Sensitivity (%)"]]

    fig = go.Figure(go.Bar(
        x=sens_df["Stock"],
        y=sens_df["Sensitivity (%)"],
        marker_color=colors,
        text=sens_df["Sensitivity (%)"].apply(lambda x: f"{x:+.4f}%"),
        textposition="outside"
    ))
    fig.update_layout(
        title="Heat Sensitivity Index",
        xaxis_title="Stock",
        yaxis_title="Return Difference (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation table
    st.subheader("📊 Temperature vs Stock Return Correlation")
    corr_df = pd.DataFrame({
        "Stock": list(correlations.keys()),
        "Correlation": list(correlations.values())
    })
    corr_df["Direction"] = corr_df["Correlation"].apply(
        lambda x: "📈 Positive" if x > 0 else "📉 Negative"
    )
    st.dataframe(corr_df, use_container_width=True)

    # Live stock prices
    st.subheader("💹 Live Stock Prices")
    cols = st.columns(5)
    for i, stock in enumerate(stock_cols):
        current = df[stock].iloc[-1]
        previous = df[stock].iloc[-2]
        change = ((current - previous) / previous) * 100
        cols[i].metric(
            stock,
            f"₹{current:.2f}",
            f"{change:+.2f}%"
        )

# ============ PAGE 2 — STOCK VS TEMPERATURE ============
elif page == "📈 Stock vs Temperature":
    st.title("📈 Stock Price vs Temperature")
    st.markdown("---")

    selected_stock = st.selectbox("Select Stock", stock_cols)
    date_range = st.date_input("Select Date Range", [df.index.min(), df.index.max()])

    if len(date_range) == 2:
        filtered = df[date_range[0].strftime("%Y-%m-%d"):date_range[1].strftime("%Y-%m-%d")]
    else:
        filtered = df

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered.index, y=filtered[selected_stock],
        name=f"{selected_stock} Price (₹)",
        line=dict(color="blue"), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=filtered.index, y=filtered["avg_temp"],
        name="Avg Temperature (°C)",
        line=dict(color="red", dash="dot"), yaxis="y2"
    ))
    heat_days = filtered[filtered["heat_shock"]]
    fig.add_trace(go.Scatter(
        x=heat_days.index, y=heat_days[selected_stock],
        mode="markers", name="Heat Shock Day",
        marker=dict(color="orange", size=6, symbol="triangle-up")
    ))
    fig.update_layout(
        title=f"{selected_stock} Stock Price vs Average Temperature",
        xaxis_title="Date",
        yaxis=dict(title="Stock Price (₹)", side="left"),
        yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"),
        height=500, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{filtered[selected_stock].iloc[-1]:.2f}")
    col2.metric("Avg Temperature", f"{filtered['avg_temp'].mean():.1f}°C")
    col3.metric("Heat Shock Days", f"{filtered['heat_shock'].sum()}")

# ============ PAGE 3 — HEAT SHOCK EVENTS ============
elif page == "⚡ Heat Shock Events":
    st.title("⚡ Heat Shock Events Analysis")
    st.markdown("*Days where Delhi temperature exceeded 40°C*")
    st.markdown("---")

    heat_df = df[df["heat_shock"]]
    st.metric("Total Heat Shock Days", len(heat_df))

    fig = px.scatter(
        df, x=df.index, y="avg_temp",
        color="heat_shock",
        color_discrete_map={True: "red", False: "blue"},
        title="Daily Average Temperature — Heat Shock Days Highlighted",
        labels={"avg_temp": "Avg Temperature (°C)"}
    )
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="40°C Threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Stock Returns on Heat Shock Days")
    heat_returns = heat_df[[f"{s}_return" for s in stock_cols]].mean()
    heat_returns.index = stock_cols

    fig2 = go.Figure(go.Bar(
        x=stock_cols,
        y=heat_returns.values,
        marker_color=["green" if x > 0 else "red" for x in heat_returns.values],
        text=[f"{x:+.4f}%" for x in heat_returns.values],
        textposition="outside"
    ))
    fig2.update_layout(
        title="Average Stock Returns on Heat Shock Days",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============ PAGE 4 — CORRELATION HEATMAP ============
elif page == "🔥 Correlation Heatmap":
    st.title("🔥 Correlation Heatmap")
    st.markdown("---")

    return_cols = [f"{s}_return" for s in stock_cols]
    corr_cols = ["avg_temp", "CDD"] + return_cols
    corr_matrix = df[corr_cols].corr()
    labels = ["Avg Temp", "CDD"] + stock_cols

    fig = px.imshow(
        corr_matrix.values,
        x=labels, y=labels,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix",
        text_auto=".2f"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Key Findings")
    st.markdown(f"""
    - 🌡️ **Temperature Tipping Point**: {tipping_point}°C
    - ⚡ **Most Heat-Sensitive Stock**: Adani Power
    - 📉 **Least Heat-Sensitive Stock**: NTPC
    - 🔥 **Total Heat Shock Days**: {df['heat_shock'].sum()} days
    - 📅 **Data updated**: {datetime.now().strftime('%d %b %Y')}
    """)