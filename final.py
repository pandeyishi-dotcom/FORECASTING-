# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Optional libraries
HAS_YFIN = False
HAS_PMD = False
try:
    import yfinance as yf
    HAS_YFIN = True
except:
    pass

try:
    from pmdarima import auto_arima
    HAS_PMD = True
except:
    pass

from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Forecasting", layout="wide")

# ---------------------------
# Helper functions
# ---------------------------

def fetch_data(ticker, start, end):
    if not HAS_YFIN:
        return None, "yfinance not installed"
    df = yf.download(ticker, start=start, end=end, interval="1mo", progress=False)
    if df.empty:
        return None, "No data downloaded"
    if "Adj Close" in df.columns:
        s = df["Adj Close"].dropna()
    else:
        s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "Price"
    return s, None

def fit_model(series, use_auto, order, seasonal):
    if use_auto:
        if not HAS_PMD:
            raise RuntimeError("pmdarima not installed (auto_arima not available).")
        step = auto_arima(
            series,
            seasonal=seasonal,
            m=12 if seasonal else 1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True
        )
        model = ARIMA(series, order=step.order, seasonal_order=step.seasonal_order)
        res = model.fit()
        return res, step.order, step.seasonal_order
    else:
        model = ARIMA(series, order=order)
        res = model.fit()
        return res, order, None

def forecast(res, series, steps=12):
    last = series.index.max()
    start = (last + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
    idx = pd.date_range(start=start, periods=steps, freq="MS")
    fc = res.get_forecast(steps)
    mean = pd.Series(fc.predicted_mean, index=idx)
    conf = fc.conf_int()
    lower = pd.Series(conf.iloc[:,0].values, index=idx)
    upper = pd.Series(conf.iloc[:,1].values, index=idx)
    return mean, lower, upper

def metrics(actual, fitted):
    df = pd.concat([actual, fitted], axis=1).dropna()
    df.columns = ["Actual", "Fitted"]
    if df.empty:
        return None
    rmse = sqrt(mean_squared_error(df["Actual"], df["Fitted"]))
    mae = mean_absolute_error(df["Actual"], df["Fitted"])
    mask = df["Actual"] != 0
    mape = (np.abs((df["Actual"][mask]-df["Fitted"][mask])/df["Actual"][mask]).mean()*100) if mask.sum() else None
    r2 = r2_score(df["Actual"], df["Fitted"])
    return rmse, mae, mape, r2

# ---------------------------
# UI
# ---------------------------

st.title("📈 ARIMA Forecasting (No Matplotlib — Plotly Only)")

left, right = st.columns([1,2])

with left:
    source = st.radio("Select data source", ["Yahoo Finance", "Upload CSV"])

    ticker = st.text_input("Ticker", value="RELIANCE.NS")

    project = st.selectbox("Project", [
        "Project 1: 2010–2018 ➜ 2019 Forecast",
        "Project 2: 2021–2025 ➜ 2026 Forecast"
    ])

    seasonal = st.checkbox("Use Seasonal ARIMA (SARIMA)?", value=False)

    if HAS_PMD:
        use_auto = st.checkbox("Use auto_arima?", value=True)
    else:
        st.warning("pmdarima not installed — auto_arima disabled.")
        use_auto = False

    if not use_auto:
        p = st.number_input("p", 0, 10, 1)
        d = st.number_input("d", 0, 2, 1)
        q = st.number_input("q", 0, 10, 1)
        manual_order = (p, d, q)

    run_btn = st.button("Run Forecast")

# ---------------------------
# Project date ranges
# ---------------------------
if "Project 1" in project:
    start, end = "2010-01-01", "2018-12-31"
else:
    start, end = "2021-01-01", "2025-12-31"

# ---------------------------
# RUN
# ---------------------------
if run_btn:

    # Load data
    if source == "Yahoo Finance":
        if not HAS_YFIN:
            st.error("yfinance not installed — upload CSV instead.")
            st.stop()

        series, err = fetch_data(ticker, start, end)
        if series is None:
            st.error(err)
            st.stop()

    else:
        file = st.file_uploader("Upload CSV with Date & Price columns", type="csv")
        if file is None:
            st.warning("Upload a CSV file.")
            st.stop()
        df = pd.read_csv(file)
        # detect date + price
        date_col = None
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
        if date_col is None:
            st.error("CSV must contain a Date column.")
            st.stop()
        price_col = None
        for c in df.columns:
            if any(x in c.lower() for x in ["price","close","adj"]):
                price_col = c
                break
        if price_col is None:
            st.error("CSV must contain a Price/Close/Adj column.")
            st.stop()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        series = df.set_index(date_col)[price_col]
        # resample monthly
        series = series.resample("MS").last().dropna()
        series.index = series.index.to_period("M").to_timestamp()
        series.name = "Price"

    st.success(f"Loaded {len(series)} monthly records.")

    # -----------------------
    # Price Chart
    # -----------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Price"))
    fig.update_layout(title="Monthly Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Fit ARIMA
    # -----------------------
    try:
        if use_auto:
            res, order, sorder = fit_model(series, use_auto=True, order=None, seasonal=seasonal)
        else:
            res, order, sorder = fit_model(series, use_auto=False, order=manual_order, seasonal=False)
    except Exception as e:
        st.error(f"ARIMA failed: {e}")
        st.stop()

    st.write(f"**Model order chosen:** {order}")
    if sorder:
        st.write(f"**Seasonal order:** {sorder}")

    # -----------------------
    # In-sample fitted
    # -----------------------
    fitted = res.predict(start=series.index.min(), end=series.index.max())
    fitted = pd.Series(fitted, index=series.index)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual"))
    fig2.add_trace(go.Scatter(x=fitted.index, y=fitted.values, name="Fitted", line=dict(dash="dash")))
    fig2.update_layout(title="Actual vs Fitted", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------
    # Forecast next 12 months
    # -----------------------
    pred, lower, upper = forecast(res, series, steps=12)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=series.index, y=series.values, name="Training Data"))
    fig3.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast", line=dict(color="crimson")))
    fig3.add_trace(go.Scatter(x=pred.index, y=upper.values, name="Upper CI", line=dict(color="lightgrey")))
    fig3.add_trace(go.Scatter(x=pred.index, y=lower.values, name="Lower CI", fill="tonexty",
                              fillcolor="rgba(200,200,200,0.3)", line=dict(color="lightgrey"), showlegend=False))
    fig3.update_layout(title="12-Month Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------
    # Metrics
    # -----------------------
    met = metrics(series, fitted)
    if met:
        rmse, mae, mape, r2 = met
        st.subheader("In-Sample Metrics")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"MAPE (%): {mape}")
        st.write(f"R²: {r2}")
    else:
        st.info("Metrics unavailable — no fitted data.")

    st.success("Forecast complete.")
