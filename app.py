import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Climate-Financial Intelligence",
    page_icon="🌡️",
    layout="wide"
)

# ============ LOAD DATA ============
@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)
    df = df.dropna()
    with open("analysis_results.json") as f:
        analysis = json.load(f)
    with open("forecast_results.json") as f:
        forecast = json.load(f)
    return df, analysis, forecast

df, analysis, forecast = load_data()

stock_cols = ["NTPC", "Tata Power", "Adani Power", "Reliance", "Power Grid"]

# ============ SIDEBAR ============
st.sidebar.title("🌡️ Climate-Financial Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📈 Stock vs Temperature",
    "🤖 Forecasting",
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

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Data Range", "2020 - 2026")
    col2.metric("🏢 Stocks Tracked", "5")
    col3.metric("⚡ Heat Shock Days", "114")
    col4.metric("🌡️ Tipping Point", f"{analysis['tipping_point']}°C")

    st.markdown("---")

    # Heat Sensitivity Index
    st.subheader("🔥 Heat Sensitivity Index")
    st.markdown("Average return difference on heat shock days vs normal days:")

    sensitivity = analysis["heat_sensitivity"]
    sens_df = pd.DataFrame({
        "Stock": list(sensitivity.keys()),
        "Sensitivity (%)": list(sensitivity.values())
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
        title="Heat Sensitivity Index — Stock Return on Heat Shock Days vs Normal Days",
        xaxis_title="Stock",
        yaxis_title="Return Difference (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation table
    st.subheader("📊 Temperature vs Stock Return Correlation")
    corr_df = pd.DataFrame({
        "Stock": list(analysis["correlations"].keys()),
        "Correlation": list(analysis["correlations"].values())
    })
    corr_df["Direction"] = corr_df["Correlation"].apply(
        lambda x: "📈 Positive" if x > 0 else "📉 Negative"
    )
    st.dataframe(corr_df, use_container_width=True)

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

    # Dual axis chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered.index,
        y=filtered[selected_stock],
        name=f"{selected_stock} Price (₹)",
        line=dict(color="blue"),
        yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=filtered.index,
        y=filtered["avg_temp"],
        name="Avg Temperature (°C)",
        line=dict(color="red", dash="dot"),
        yaxis="y2"
    ))

    # Mark heat shock events
    heat_days = filtered[filtered["heat_shock"]]
    fig.add_trace(go.Scatter(
        x=heat_days.index,
        y=heat_days[selected_stock],
        mode="markers",
        name="Heat Shock Day",
        marker=dict(color="orange", size=6, symbol="triangle-up")
    ))

    fig.update_layout(
        title=f"{selected_stock} Stock Price vs Average Temperature",
        xaxis_title="Date",
        yaxis=dict(title="Stock Price (₹)", side="left"),
        yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"),
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{filtered[selected_stock].iloc[-1]:.2f}")
    col2.metric("Avg Temperature", f"{filtered['avg_temp'].mean():.1f}°C")
    col3.metric("Heat Shock Days", f"{filtered['heat_shock'].sum()}")

# ============ PAGE 3 — FORECASTING ============
elif page == "🤖 Forecasting":
    st.title("🤖 90-Day Stock Price Forecast")
    st.markdown("*Powered by Prophet ML model with temperature as input*")
    st.markdown("---")

    selected_stock = st.selectbox("Select Stock", stock_cols)

    fc = forecast[selected_stock]
    fc_df = pd.DataFrame({
        "Date": pd.to_datetime(fc["dates"]),
        "Predicted": fc["predicted"],
        "Lower": fc["lower"],
        "Upper": fc["upper"]
    })

    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[selected_stock],
        name="Historical Price",
        line=dict(color="blue")
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_df["Date"],
        y=fc_df["Predicted"],
        name="Forecast",
        line=dict(color="green", dash="dash")
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
        y=pd.concat([fc_df["Upper"], fc_df["Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(0,255,0,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval"
    ))

    fig.update_layout(
        title=f"{selected_stock} — 90 Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{df[selected_stock].iloc[-1]:.2f}")
    col2.metric("90-Day Forecast", f"₹{fc_df['Predicted'].iloc[-1]:.2f}")
    direction = "📈" if fc_df['Predicted'].iloc[-1] > df[selected_stock].iloc[-1] else "📉"
    col3.metric("Direction", direction)

# ============ PAGE 4 — HEAT SHOCK EVENTS ============
elif page == "⚡ Heat Shock Events":
    st.title("⚡ Heat Shock Events Analysis")
    st.markdown("*Days where average temperature exceeded 40°C*")
    st.markdown("---")

    heat_df = df[df["heat_shock"]]

    st.metric("Total Heat Shock Days", len(heat_df))

    # Temperature timeline
    fig = px.scatter(
        df,
        x=df.index,
        y="avg_temp",
        color="heat_shock",
        color_discrete_map={True: "red", False: "blue"},
        title="Daily Average Temperature — Heat Shock Days Highlighted",
        labels={"avg_temp": "Avg Temperature (°C)", "index": "Date"}
    )
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="40°C Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Stock returns on heat shock days
    st.subheader("📊 Stock Returns on Heat Shock Days")
    return_cols = [f"{s}_return" for s in stock_cols]
    heat_returns = heat_df[return_cols].mean()
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
        xaxis_title="Stock",
        yaxis_title="Average Return (%)",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============ PAGE 5 — CORRELATION HEATMAP ============
elif page == "🔥 Correlation Heatmap":
    st.title("🔥 Correlation Heatmap")
    st.markdown("*Correlation between temperature, CDD and stock returns*")
    st.markdown("---")

    return_cols = [f"{s}_return" for s in stock_cols]
    corr_cols = ["avg_temp", "CDD"] + return_cols
    corr_matrix = df[corr_cols].corr()

    labels = ["Avg Temp", "CDD"] + stock_cols

    fig = px.imshow(
        corr_matrix.values,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix — Temperature vs Stock Returns",
        text_auto=".2f"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Key Findings")
    st.markdown(f"""
    - 🌡️ **Temperature Tipping Point**: {analysis['tipping_point']}°C
    - ⚡ **Most Heat-Sensitive Stock**: Adani Power (+0.4008% on heat shock days)
    - 📉 **Least Heat-Sensitive Stock**: NTPC (-0.1257% on heat shock days)
    - 🔥 **Total Heat Shock Days**: 114 days (2020-2026)
    - 📈 **Only stock predicted to rise**: Reliance Industries
    """)