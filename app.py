import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page Config ---
st.set_page_config(
    page_title="Nairobi Air Quality",
    page_icon="🌍",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    df_hourly = pd.read_csv("data/df_hourly.csv", index_col="timestamp", parse_dates=True)
    predictions = pd.read_csv("data/predictions.csv", index_col="timestamp", parse_dates=True)
    mae_scores = pd.read_csv("data/mae_scores.csv")
    return df_hourly, predictions, mae_scores

df_hourly, predictions, mae_scores = load_data()

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png", width=100)
st.sidebar.title("Nairobi Air Quality")
st.sidebar.markdown("PM2.5 Time Series Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "📊 Exploratory Analysis",
    "🔬 Model Analysis",
    "📈 Model Results"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data source:** sensors.AFRICA")
st.sidebar.markdown("**Period:** December 2025")
st.sidebar.markdown("**Location:** Nairobi, Kenya")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🌍 Nairobi PM2.5 Air Quality Analysis")
    st.markdown("#### Time series forecasting of particulate matter concentrations using classical statistical models")
    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Readings", "30,884")
    col2.metric("Hourly Averages", "743")
    col3.metric("EPA Exceedances", "35 hours")
    col4.metric("Exceedance Rate", "4.71%")

    st.markdown("---")

    # EPA exceedance plot
    st.subheader("PM2.5 Levels vs EPA Threshold")
    epa_threshold = 35.4
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_hourly.index, df_hourly["pm2.5"], color="steelblue", linewidth=0.8, label="PM2.5")
    ax.axhline(y=epa_threshold, color="red", linestyle="--", linewidth=1.5, label="EPA Threshold (35.4 µg/m³)")
    ax.fill_between(df_hourly.index, df_hourly["pm2.5"], epa_threshold,
                    where=df_hourly["pm2.5"] > epa_threshold,
                    color="red", alpha=0.3, label="Exceeds Threshold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.info("📌 Nearly all exceedances occurred in the final 3 days of December — driven by festive season activity, open burning, and fireworks.")

    st.markdown("---")
    # Dataset description
    st.subheader("About the Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Source:** sensors.AFRICA Air Quality Archive  
        **Portal:** openAfrica (open.africa)  
        **Period:** December 2025  
        **City:** Nairobi, Kenya  
        """)
    with col2:
        st.markdown("""
        **Sensor types:** PMS5003, DHT22  
        **Measurements:** PM1, PM2.5, PM10, Temperature, Humidity  
        **Resampled to:** Hourly averages  
        **Target variable:** PM2.5 (µg/m³)  
        """)

# ============================================================
# PAGE 2 — EXPLORATORY ANALYSIS
# ============================================================
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    # Raw time series
    st.subheader("Hourly PM2.5 — December 2025")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_hourly.index, df_hourly["pm2.5"], color="steelblue", linewidth=0.8)
    ax.axhline(y=35.4, color="red", linestyle="--", linewidth=1, label="EPA Threshold")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Seasonal decomposition
    st.subheader("Seasonal Decomposition (period = 24 hours)")
    decomposition = seasonal_decompose(df_hourly["pm2.5"], model="additive", period=24)
    fig = decomposition.plot()
    fig.set_size_inches(14, 8)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("📌 A clear 24-hour seasonal cycle is visible — PM2.5 peaks in the evening and drops overnight. The trend dips in mid-December before rising sharply towards New Year.")

    st.markdown("---")

    # ACF and PACF
    st.subheader("ACF and PACF")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df_hourly["pm2.5"], ax=axes[0], lags=40)
    axes[0].set_title("ACF")
    axes[0].set_xlabel("Lag [hours]")
    plot_pacf(df_hourly["pm2.5"], ax=axes[1], lags=40)
    axes[1].set_title("PACF")
    axes[1].set_xlabel("Lag [hours]")
    plt.tight_layout()
    st.pyplot(fig)

    st.info("📌 PACF cuts off after lag 2 → AR order p = 2. ACF shows gradual decay with seasonal spikes at lag 24 → confirms daily seasonality.")

# ============================================================
# PAGE 3 — MODEL ANALYSIS
# ============================================================
elif page == "🔬 Model Analysis":
    st.title("🔬 Model Analysis")
    st.markdown("---")

    # ADF Test
    st.subheader("Stationarity — ADF Test")
    col1, col2, col3 = st.columns(3)
    col1.metric("ADF Statistic", "-3.95")
    col2.metric("p-value", "0.0017")
    col3.metric("Result", "Stationary ✅")
    st.success("p-value < 0.05 — series is stationary. No differencing required (d = 0).")

    st.markdown("---")

    # Model equations
    st.subheader("Model Equations")

    st.markdown("**Linear Regression**")
    st.code("pm2.5 = 4.73 + (0.79 × L1)")

    st.markdown("**AR(2,0,0)**")
    st.code("pm2.5 = 22.54 + (0.9742 × L1) + (-0.2329 × L2)")

    st.markdown("**ARMA(2,0,2)**")
    st.code("pm2.5 = 22.55 + (1.4734 × L1) + (-0.6000 × L2) + (-0.5097 × e1) + (-0.0908 × e2)")

    st.markdown("**SARIMA(2,0,2)(1,0,1,24)**")
    st.code("pm2.5 = (1.6340 × L1) + (-0.6391 × L2) + (-0.7972 × e1) + (-0.1231 × e2) + (0.9984 × SL24) + (-0.9489 × Se1)")

    st.markdown("---")

    # Model fit comparison
    st.subheader("Model Fit — AIC / BIC / Log Likelihood")
    df_fit = pd.DataFrame({
        "Model": ["AR(2,0,0)", "ARMA(2,0,2)", "SARIMA(2,0,2)(1,0,1,24)"],
        "AIC": [3336.17, 3333.16, 3168.08],
        "BIC": [3353.72, 3359.48, 3198.79],
        "Log Likelihood": [-1664.08, -1660.58, -1577.04]
    })
    st.dataframe(df_fit, use_container_width=True)
    st.info("📌 SARIMA dominates on all three metrics — the seasonal component significantly improves model fit.")

# ============================================================
# PAGE 4 — MODEL RESULTS
# ============================================================
elif page == "📈 Model Results":
    st.title("📈 Model Results")
    st.markdown("---")

    # MAE comparison
    st.subheader("Model Comparison — MAE")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["gray", "steelblue", "orange", "green", "red"]
    bars = ax.bar(mae_scores["model"], mae_scores["mae"], color=colors)
    ax.axhline(y=5.36, color="gray", linestyle="--")
    ax.set_ylabel("MAE (µg/m³)")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()
    st.pyplot(fig)

    st.success("🏆 SARIMA achieves the lowest MAE of 3.59 — outperforming all other models by capturing the daily seasonal cycle.")

    st.markdown("---")

    # Actual vs Predicted plots
    st.subheader("Actual vs Predicted — Test Period")

    model_cols = {
        "Linear Regression": "linear_regression",
        "AR(2,0,0)": "ar",
        "ARMA(2,0,2)": "arma",
        "SARIMA(2,0,2)(1,0,1,24)": "sarima"
    }

    selected_model = st.selectbox("Select model", list(model_cols.keys()))
    col = model_cols[selected_model]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(predictions.index, predictions["actual"], label="Actual", color="steelblue")
    ax.plot(predictions.index, predictions[col], label="Predicted", color="red")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Date")
    ax.set_title(f"{selected_model} — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # MAE table
    st.subheader("MAE Summary")
    st.dataframe(mae_scores, use_container_width=True)