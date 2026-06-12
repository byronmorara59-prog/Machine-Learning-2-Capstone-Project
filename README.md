# Time Series Analysis of PM2.5 Air Quality in Nairobi



## Project Description

This project analyses PM2.5 (fine particulate matter) air quality readings from Nairobi, Kenya, using low-cost sensor data from the sensors.AFRICA network — a Code for Africa initiative. The dataset contains over 30,000 readings from December 2025, resampled to 743 hourly averages. Four time series models were built and evaluated using walk-forward validation — Linear Regression, AR, ARMA, and SARIMA — with results deployed in a Streamlit dashboard.



## Problem Statement

Can classical time series models reliably forecast hourly PM2.5 levels in Nairobi, and what does the temporal structure of the data reveal about air quality patterns in the city? The EPA 24-hour standard for PM2.5 is 35.4 µg/m³ — analysis shows 4.71% of hours exceeded this threshold, with exceedances concentrated in the final 3 days of December, driven by festive season activity.



## Data

| Field | Description |
|---|---|
| **Source** | sensors.AFRICA Air Quality Archive — openAfrica (open.africa) |
| **Period** | December 2025 |
| **Location** | Nairobi, Kenya (lat: -1.311, lon: 36.817) |
| **Sensor types** | PMS5003 (particulate matter), DHT22 (temperature, humidity) |
| **Raw readings** | 30,884 |
| **Hourly averages** | 743 |
| **Target variable** | PM2.5 (µg/m³) — `value_type == "P2"` |

### Wrangling steps — `wrangle()` function

1. Load with `sep=";"` and `on_bad_lines="skip"` — handles semicolon delimiter and corrupted rows
2. Filter `value_type == "P2"` — keeps PM2.5 readings only
3. Parse timestamps with `format='ISO8601'` — convert UTC to `Africa/Nairobi` (+03:00)
4. Convert to float using `pd.to_numeric(errors="coerce")` — corrupted strings become NaN
5. Drop NaNs and remove sensor errors (`pm2.5 > 500`)
6. Resample to hourly with `resample("1h").mean().ffill()`
7. Add lag feature `pm2.5_L1 = pm2.5.shift(1)`

---

## Exploratory Analysis

### Stationarity — ADF test

| Metric | Value |
|---|---|
| ADF Statistic | -3.95 |
| p-value | 0.0017 |
| Critical value (1%) | -3.44 |
| Critical value (5%) | -2.87 |
| Result | **Stationary — d = 0, no differencing required** |

### ACF / PACF results

- PACF cuts off sharply after lag 2 → **p = 2**
- ACF shows gradual decay with seasonal spikes at lag 24 → **q = 2**
- Seasonal period confirmed as **s = 24 hours** (one full day)

### Seasonal decomposition

- Clear additive daily cycle (±5 µg/m³) repeating every 24 hours
- Trend dips mid-December (short rains) then rises sharply towards New Year
- Residuals mostly random with larger spikes at end of December
- Confirms `model="additive"` and `s=24` for SARIMA

### EPA exceedance analysis

| Metric | Value |
|---|---|
| Total hours monitored | 743 |
| Hours exceeding EPA threshold (35.4 µg/m³) | 35 |
| Exceedance rate | 4.71% |

> Nearly all exceedances occurred Dec 29 – Jan 1, driven by festive season activity, open burning, and fireworks.

---

## Methodology

1. **Data wrangling** — Reusable `wrangle()` function handling all cleaning, filtering, and feature creation
2. **EDA and decomposition** — Time series plots, rolling averages, correlation check, ADF test, ACF/PACF, seasonal decomposition, EPA exceedance analysis
3. **Train/test split** — Chronological 80/20 split. Training: Dec 1–25 (594 hours). Test: Dec 26–Jan 1 (149 hours)
4. **Baseline model** — Predict mean PM2.5 (22.59 µg/m³) for every hour. MAE = 5.36
5. **Model fitting and walk-forward validation** — Each model evaluated via one-step-ahead walk-forward validation, feeding actual values back into the training window before each step
6. **Streamlit dashboard** — Results deployed as an interactive app with four pages: Overview, EDA, Model Analysis, and Model Results

---

## Models

### Baseline
> Predict mean PM2.5 = 22.59 µg/m³ for every hour. **MAE = 5.36**

### Model 1 — Linear Regression
```
pm2.5 = 4.73 + (0.79 × pm2.5_L1)
```
Single lag-1 feature. Coefficient of 0.79 confirms strong autocorrelation at lag-1. Training MAE: 3.13 · **Test MAE: 3.98**

### Model 2 — AR(2,0,0)
```
pm2.5 = 22.54 + (0.9742 × L1) + (-0.2329 × L2)
```
Pure autoregressive model using 2 past values. All coefficients statistically significant (p = 0.000). **Walk-forward MAE: 3.84**

| Metric | Value |
|---|---|
| AIC | 3336.17 |
| BIC | 3353.72 |
| Log Likelihood | -1664.08 |
| Prob(Q) — Ljung-Box | 0.82 ✓ |
| Prob(H) — Heteroskedasticity | 0.59 ✓ |

### Model 3 — ARMA(2,0,2)
```
pm2.5 = 22.55 + (1.4734 × L1) + (-0.6000 × L2) + (-0.5097 × e1) + (-0.0908 × e2)
```
Adds MA component to AR. `ma.L2` not statistically significant (p = 0.219). AIC improves over AR but BIC worsens — extra complexity not fully justified. **Walk-forward MAE: 3.85**

| Metric | Value |
|---|---|
| AIC | 3333.16 |
| BIC | 3359.48 |
| Log Likelihood | -1660.58 |

### Model 4 — SARIMA(2,0,2)(1,0,1,24) ✓ Best model
```
pm2.5 = (1.6340 × L1) + (-0.6391 × L2) + (-0.7972 × e1) + (-0.1231 × e2) + (0.9984 × SL24) + (-0.9489 × Se1)
```
Adds seasonal terms with period s=24. SL24 = 0.9984 — PM2.5 at the same hour yesterday is nearly a perfect predictor. Dominates all fit metrics. **Walk-forward MAE: 3.59**

| Metric | Value |
|---|---|
| AIC | 3168.08 |
| BIC | 3198.79 |
| Log Likelihood | -1577.04 |

---

## Results

| Model | MAE (µg/m³) | vs Baseline |
|---|---|---|
| Baseline | 5.36 | — |
| Linear Regression | 3.98 | ↓ 25.7% |
| AR(2,0,0) | 3.84 | ↓ 28.4% |
| ARMA(2,0,2) | 3.85 | ↓ 28.2% |
| **SARIMA(2,0,2)(1,0,1,24)** | **3.59** | **↓ 33.0%** |

---

## Key Findings

- Every model beat the baseline — PM2.5 has a strong, learnable temporal structure
- The MA component in ARMA added almost nothing — AR and ARMA are virtually identical (3.84 vs 3.85)
- SARIMA clearly outperforms all models. The SL24 coefficient of 0.9984 shows that PM2.5 at the same hour yesterday is nearly a perfect predictor of today's reading
- Walk-forward validation is essential — naive multi-step AR forecasting gave MAE of 7.17 (worse than baseline), while walk-forward gave 3.84
- 4.71% of hours exceeded the EPA threshold, nearly all in the final 3 days of December — festive season activity is a significant driver of poor air quality in Nairobi

---

## Tools and Libraries

| Category | Libraries |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Modelling | `statsmodels` (ARIMA, SARIMAX, ADF, ACF/PACF), `sklearn` (LinearRegression, MAE) |
| Visualisation | `matplotlib`, `seaborn`, `plotly.express` |
| Deployment | `streamlit` |
| Environment | Python 3, VS Code, Jupyter notebooks |

---

## Limitations and Next Steps

**Limitations**
- One month of data only — a longer dataset would reveal annual seasonality
- Single sensor site — multi-site averaging would reduce noise from faulty sensors
- No hyperparameter grid search over p, q, P, Q orders
- LSTM not implemented due to environment constraints


