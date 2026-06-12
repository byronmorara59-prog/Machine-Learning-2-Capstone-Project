import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

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
    "🔬 Stationarity & ACF/PACF",
    "📈 Model Analysis",
    "🏆 Model Comparison"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data source:** sensors.AFRICA")
st.sidebar.markdown("**Period:** December 2025")
st.sidebar.markdown("**Location:** Nairobi, Kenya")
st.sidebar.markdown("**Train/Test split:** 80/20")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🌍 Nairobi PM2.5 Air Quality Analysis")
    st.markdown("#### Time series forecasting of PM2.5 concentrations using classical statistical models")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Readings", "30,884")
    col2.metric("Hourly Averages", "743")
    col3.metric("EPA Exceedances", "35 hours")
    col4.metric("Exceedance Rate", "4.71%")

    st.markdown("---")
    st.subheader("About the project")
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
        **Target variable:** PM2.5 (µg/m³)
        **Resampled to:** Hourly averages
        **Best model MAE:** 3.59 µg/m³
        """)

    st.markdown("---")
    st.subheader("PM2.5 levels vs EPA threshold — December 2025")
    epa_threshold = 35.4
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_hourly.index, df_hourly["pm2.5"], color="steelblue", linewidth=0.8, label="PM2.5")
    ax.axhline(y=epa_threshold, color="red", linestyle="--", linewidth=1.5,
               label="EPA Threshold (35.4 µg/m³)")
    ax.fill_between(df_hourly.index, df_hourly["pm2.5"], epa_threshold,
                    where=df_hourly["pm2.5"] > epa_threshold,
                    color="red", alpha=0.3, label="Exceeds Threshold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Nearly all exceedances occurred in the final 3 days of December — driven by festive season activity, open burning, and fireworks.")

# ============================================================
# PAGE 2 — EXPLORATORY ANALYSIS
# ============================================================
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    st.subheader("Raw PM2.5 time series — December 2025")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_hourly.index, df_hourly["pm2.5"], color="steelblue", linewidth=0.8)
    ax.axhline(y=35.4, color="red", linestyle="--", linewidth=1, label="EPA Threshold")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Hourly averages resampled from raw readings. No gaps confirmed after resampling.")

    st.markdown("---")
    st.subheader("Weekly rolling average")
    fig, ax = plt.subplots(figsize=(14, 4))
    df_hourly["pm2.5"].rolling(168).mean().plot(
        ax=ax, color="steelblue", ylabel="PM2.5 (µg/m³)")
    ax.set_xlabel("Date")
    ax.set_title("")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("168-hour (7-day) rolling mean. Shows a clear dip in mid-December and a sharp rise towards New Year.")

    st.markdown("---")
    st.subheader("Seasonal decomposition — period = 24 hours")
    decomposition = seasonal_decompose(df_hourly["pm2.5"], model="additive", period=24)
    fig = decomposition.plot()
    fig.set_size_inches(14, 8)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 A clear 24-hour daily cycle is visible in the seasonal component (±5 µg/m³). The trend dips mid-December before rising sharply towards New Year. Residuals are mostly random with larger spikes at the end of December.")

# ============================================================
# PAGE 3 — STATIONARITY & ACF/PACF
# ============================================================
elif page == "🔬 Stationarity & ACF/PACF":
    st.title("🔬 Stationarity & ACF/PACF Analysis")
    st.markdown("---")

    st.subheader("Augmented Dickey-Fuller (ADF) test")
    st.markdown("Tests whether the series is stationary. Null hypothesis: the series has a unit root (non-stationary). We reject the null when p-value < 0.05.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ADF Statistic", "-3.95")
    col2.metric("p-value", "0.0017")
    col3.metric("Critical Value (1%)", "-3.44")
    col4.metric("Critical Value (5%)", "-2.87")
    col5.metric("Result", "Stationary ✅")
    st.success("p-value of 0.0017 is well below 0.05. ADF statistic of -3.95 clears all three critical value thresholds. The series is stationary — d = 0, no differencing required.")

    st.markdown("---")
    st.subheader("ACF plot — determines MA order (q)")
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_acf(df_hourly["pm2.5"], ax=ax, lags=40)
    ax.set_xlabel("Lag [hours]")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("ACF cuts off after lag 2 before seasonal spikes reappear around lag 24 — confirms q = 2 and daily seasonality (s = 24).")

    st.markdown("---")
    st.subheader("PACF plot — determines AR order (p)")
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_pacf(df_hourly["pm2.5"], ax=ax, lags=40)
    ax.set_xlabel("Lag [hours]")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("PACF cuts off sharply after lag 2 — all subsequent lags fall inside the confidence zone. Confirms p = 2.")

    st.markdown("---")
    st.subheader("Order selection summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("AR order (p)", "2", "from PACF cutoff")
    col2.metric("Differencing (d)", "0", "series is stationary")
    col3.metric("MA order (q)", "2", "from ACF cutoff")

# ============================================================
# PAGE 4 — MODEL ANALYSIS
# ============================================================
elif page == "📈 Model Analysis":
    st.title("📈 Model Analysis")
    st.markdown("---")

    st.subheader("Baseline")
    st.markdown("Predict the mean PM2.5 (22.59 µg/m³) for every hour — no intelligence, no patterns.")
    col1, col2 = st.columns(2)
    col1.metric("Mean PM2.5", "22.59 µg/m³")
    col2.metric("Baseline MAE", "5.36 µg/m³")
    st.warning("Every model must beat MAE = 5.36 to be considered useful.")

    st.markdown("---")
    st.subheader("Model equations")
    st.markdown("**Linear Regression**")
    st.code("pm2.5 = 4.73 + (0.79 × pm2.5_L1)")
    st.markdown("**AR(2,0,0)**")
    st.code("pm2.5 = 22.54 + (0.9742 × L1) + (-0.2329 × L2)")
    st.markdown("**ARMA(2,0,2)**")
    st.code("pm2.5 = 22.55 + (1.4734 × L1) + (-0.6000 × L2) + (-0.5097 × e1) + (-0.0908 × e2)")
    st.markdown("**SARIMA(2,0,2)(1,0,1,24)**")
    st.code("pm2.5 = (1.6340 × L1) + (-0.6391 × L2) + (-0.7972 × e1) + (-0.1231 × e2) + (0.9984 × SL24) + (-0.9489 × Se1)")
    st.info("📌 SL24 = 0.9984 — what the PM2.5 was at this exact hour yesterday is nearly a perfect predictor of today's reading.")

    st.markdown("---")
    st.subheader("Model fit — AIC / BIC / Log Likelihood")
    df_fit = pd.DataFrame({
        "Model": ["AR(2,0,0)", "ARMA(2,0,2)", "SARIMA(2,0,2)(1,0,1,24)"],
        "AIC": [3336.17, 3333.16, 3168.08],
        "BIC": [3353.72, 3359.48, 3198.79],
        "Log Likelihood": [-1664.08, -1660.58, -1577.04]
    })
    st.dataframe(df_fit, use_container_width=True, hide_index=True)
    st.info("📌 SARIMA dominates on all three metrics — AIC drops by 165 points vs AR, driven entirely by the seasonal component.")

    st.markdown("---")
    st.subheader("Actual vs predicted — test period (Dec 26 – Jan 1)")
    model_map = {
        "Linear Regression": "linear_regression",
        "AR(2,0,0)": "ar",
        "ARMA(2,0,2)": "arma",
        "SARIMA(2,0,2)(1,0,1,24)": "sarima"
    }
    selected = st.selectbox("Select model to view", list(model_map.keys()))
    col = model_map[selected]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(predictions.index, predictions["actual"], label="Actual", color="steelblue")
    ax.plot(predictions.index, predictions[col], label="Predicted", color="red")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Date")
    ax.set_title(f"{selected} — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# PAGE 5 — MODEL COMPARISON
# ============================================================
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")
    st.markdown("---")

    st.subheader("MAE leaderboard")
    st.dataframe(mae_scores.rename(columns={"model": "Model", "mae": "MAE (µg/m³)"}),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("MAE comparison chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#9E9E9E", "#1565C0", "#E65100", "#2E7D32", "#B71C1C"]
    bars = ax.bar(mae_scores["model"], mae_scores["mae"], color=colors)
    ax.axhline(y=5.36, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("MAE (µg/m³)")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()
    st.pyplot(fig)
    st.success("🏆 SARIMA achieves the lowest MAE of 3.59 — a 33% improvement over the baseline.")

    st.markdown("---")
    st.subheader("Key findings")
    st.markdown("""
    - Every model beat the baseline — PM2.5 has a strong, learnable temporal structure
    - The MA component in ARMA added almost nothing — AR and ARMA are virtually identical (3.84 vs 3.85)
    - SARIMA clearly outperforms all models — the 24-hour seasonal cycle is the dominant pattern
    - Walk-forward validation is essential — naive multi-step AR forecasting gave MAE of 7.17 (worse than baseline)
    - 4.71% of hours exceeded the EPA threshold, nearly all in the final 3 days of December
    """)