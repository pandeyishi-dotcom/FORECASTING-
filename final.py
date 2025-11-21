# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12,5)

# Try optional libraries
try:
    import yfinance as yf
    HAS_YFIN = True
except Exception:
    HAS_YFIN = False

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def download_monthly_safe(ticker, start, end):
    if not HAS_YFIN:
        return None, "yfinance not installed"
    df = yf.download(ticker, start=start, end=end, interval="1mo", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None, "No data returned"
    # Prefer Adj Close if available else Close
    if 'Adj Close' in df.columns:
        s = df['Adj Close'].dropna().copy()
        col = 'Adj Close'
    else:
        s = df['Close'].dropna().copy()
        col = 'Close'
    s.index = pd.DatetimeIndex(s.index).to_period('M').to_timestamp()  # align to month start
    s.name = 'Price'
    return s, col

def fit_arima_with_auto(series, seasonal=False, m=12, max_p=5, max_q=5, trace=False):
    # Use pmdarima if available; otherwise raise
    if not HAS_PMDARIMA:
        raise RuntimeError("pmdarima not installed; switch to manual order.")
    step = auto_arima(series, start_p=0, start_q=0,
                      max_p=max_p, max_q=max_q, seasonal=seasonal, m=m if seasonal else 1,
                      trace=trace, error_action='ignore', suppress_warnings=True, stepwise=True)
    order = step.order
    seasonal_order = step.seasonal_order if seasonal else (0,0,0,0)
    model = ARIMA(series, order=order, seasonal_order=seasonal_order if seasonal else (0,0,0,0))
    res = model.fit()
    return step, res

def fit_arima_manual(series, order):
    model = ARIMA(series, order=order)
    res = model.fit()
    return None, res

def forecast_series(res, series_index, steps=12):
    last = pd.to_datetime(series_index.max())
    start = (last + pd.offsets.MonthBegin(1)).to_period('M').to_timestamp()
    forecast_index = pd.date_range(start=start, periods=steps, freq='MS')
    fc = res.get_forecast(steps=steps)
    pred = pd.Series(fc.predicted_mean, index=forecast_index)
    conf = fc.conf_int(alpha=0.05)
    lower = pd.Series(conf.iloc[:,0].values, index=forecast_index)
    upper = pd.Series(conf.iloc[:,1].values, index=forecast_index)
    return pred, lower, upper

def compute_metrics(actual, predicted):
    df = pd.concat([actual.rename('Actual'), predicted.rename('Pred')], axis=1).dropna()
    if df.empty:
        return {}
    y_true = df['Actual'].values
    y_pred = df['Pred'].values
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100) if mask.sum() else np.nan
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2': r2, 'N': len(df)}

def to_csv_bytes(df):
    b = df.to_csv(index=True).encode('utf-8')
    return b

def download_link_bytes(b, filename='forecast.csv'):
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("ARIMA Forecasting — RELIANCE (two projects)")

st.markdown("""
This app runs ARIMA forecasting for two pre-defined projects:
- **Project 1**: monthly data 2010–2018 → forecast for 2019  
- **Project 2**: monthly data 2021–2025 → forecast for 2026  

You can fetch data from Yahoo Finance (if `yfinance` is installed) or upload your own CSV with Date & Price.
If `pmdarima` is installed, auto_arima is available — otherwise provide manual (p,d,q).
""")

col1, col2 = st.columns([1,2])

with col1:
    # Data source selection
    st.header("Data source")
    data_source = st.selectbox("Choose data source", ("Yahoo Finance (if available)", "Upload CSV (Date + Price)"))
    ticker = st.text_input("Ticker (for Yahoo)", value="RELIANCE.NS")
    seasonal = st.checkbox("Use seasonal SARIMA (m=12)?", value=False)
    st.markdown("If pmdarima is not installed, manual (p,d,q) input will be used.")

    st.markdown("---")
    st.header("ARIMA options")
    if HAS_PMDARIMA:
        use_auto = st.checkbox("Use auto_arima (pmdarima)", value=True)
    else:
        st.info("pmdarima not installed — auto_arima disabled; enter manual order below.")
        use_auto = False

    if not use_auto:
        st.markdown("Enter manual ARIMA order (p,d,q):")
        p = st.number_input("p", min_value=0, max_value=10, value=1)
        d = st.number_input("d", min_value=0, max_value=2, value=1)
        q = st.number_input("q", min_value=0, max_value=10, value=1)
        manual_order = (int(p), int(d), int(q))
    else:
        max_p = st.number_input("max_p (auto_arima)", min_value=0, max_value=10, value=5)
        max_q = st.number_input("max_q (auto_arima)", min_value=0, max_value=10, value=5)

    st.markdown("---")
    st.header("Project")
    project = st.selectbox("Choose Project", ("Project 1: 2010-2018 → Forecast 2019", "Project 2: 2021-2025 → Forecast 2026"))
    if project.startswith("Project 1"):
        start = "2010-01-01"
        end = "2018-12-31"
        forecast_label = "2019"
    else:
        start = "2021-01-01"
        end = "2025-12-31"
        forecast_label = "2026"

    run_button = st.button("Run ARIMA & Plot")

with col2:
    st.header("Output")
    out = st.empty()
    info = st.empty()

# ---------------------------
# Run logic
# ---------------------------
if run_button:
    # Step 1: Load series
    series = None
    used_col = None
    if data_source.startswith("Yahoo"):
        if not HAS_YFIN:
            st.error("yfinance not installed. Choose 'Upload CSV' or install yfinance (pip install yfinance).")
            st.stop()
        out.info(f"Downloading {ticker} monthly from {start} to {end}...")
        s, used_col = download_monthly_safe(ticker, start, end)
        if s is None:
            st.error(f"Failed to download data: {used_col}")
            st.stop()
        series = s
    else:
        uploaded = st.file_uploader("Upload CSV file (Date + Price column)", type=['csv'])
        if uploaded is None:
            st.warning("Upload a CSV file to continue.")
            st.stop()
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            st.stop()
        # Detect date column and price column
        cols_low = [c.lower() for c in df_in.columns]
        date_col = None
        for c in df_in.columns:
            if c.lower() in ('date','ds','timestamp','time'):
                date_col = c
                break
        if date_col is None:
            # fallback: find column with parseable values
            for c in df_in.columns:
                try:
                    pd.to_datetime(df_in[c].dropna().iloc[:10])
                    date_col = c
                    break
                except Exception:
                    continue
        if date_col is None:
            st.error("No date-like column found in CSV.")
            st.stop()
        # find numeric price col (prefer 'price' 'adj' 'close')
        price_col = None
        for name in df_in.columns:
            if any(x in name.lower() for x in ('adj','close','price')):
                if pd.api.types.is_numeric_dtype(df_in[name]):
                    price_col = name
                    break
        if price_col is None:
            # fallback to first numeric column excluding date
            for name in df_in.columns:
                if name != date_col and pd.api.types.is_numeric_dtype(df_in[name]):
                    price_col = name
                    break
        if price_col is None:
            st.error("No numeric price column found in CSV.")
            st.stop()
        # Build series
        df_in[date_col] = pd.to_datetime(df_in[date_col])
        df_in = df_in.sort_values(date_col)
        s = df_in.set_index(date_col)[price_col].copy().dropna()
        # Resample to month start if daily
        if pd.infer_freq(s.index) is None:
            s = s.resample('MS').last().dropna()
        s.index = pd.DatetimeIndex(s.index).to_period('M').to_timestamp()
        s.name = 'Price'
        series = s
        used_col = price_col

    out.info(f"Series ready: {len(series)} monthly points. Using column: {used_col}")
    st.write(f"Data range: {series.index.min().date()} — {series.index.max().date()}")

    # Plot price
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, linewidth=1.5)
    ax.set_title(f"Price ({start} to {end})")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    # Fit ARIMA
    try:
        if use_auto:
            with st.spinner("Running auto_arima (this may take a bit)..."):
                step, res = fit_arima_with_auto(series, seasonal=seasonal, m=12, max_p=max_p, max_q=max_q, trace=False)
            st.success(f"auto_arima selected order: {step.order}; seasonal_order: {step.seasonal_order}")
        else:
            with st.spinner("Fitting ARIMA with manual order..."):
                step, res = fit_arima_manual(series, order=manual_order)
            st.success(f"Fitted manual ARIMA order: {manual_order}")
    except Exception as e:
        st.error("ARIMA fit failed: " + str(e))
        st.stop()

    # In-sample fitted
    try:
        fitted_vals = res.predict(start=series.index.min(), end=series.index.max())
        fitted_series = pd.Series(fitted_vals, index=series.index)
    except Exception:
        fitted_series = pd.Series(dtype=float, index=series.index)

    # Plot actual vs fitted
    fig2, ax2 = plt.subplots()
    ax2.plot(series.index, series.values, label='Actual', linewidth=1.5)
    if not fitted_series.empty:
        ax2.plot(fitted_series.index, fitted_series.values, label='ARIMA fitted', linestyle='--')
    ax2.set_title("Actual vs ARIMA fitted (in-sample)")
    ax2.legend(); ax2.grid(alpha=0.25)
    st.pyplot(fig2)

    # Forecast 12 months
    try:
        pred, lower, upper = forecast_series(res, series.index, steps=12)
    except Exception as e:
        st.error("Forecast failed: " + str(e))
        st.stop()

    # Plot forecast appended
    fig3, ax3 = plt.subplots()
    ax3.plot(series.index, series.values, label='Actual (train)', linewidth=1.3)
    ax3.plot(pred.index, pred.values, label=f'Forecast {forecast_label}', color='crimson')
    ax3.fill_between(pred.index, lower.values, upper.values, color='lightgrey', alpha=0.4, label='95% CI')
    ax3.set_title(f"Forecast for {forecast_label}")
    ax3.legend(); ax3.grid(alpha=0.25)
    st.pyplot(fig3)

    # Metrics: compare fitted vs actual (in-sample)
    metrics = compute_metrics(series, fitted_series)
    st.write("### In-sample metrics (Actual vs Fitted)")
    if metrics:
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        st.table(df_metrics)
    else:
        st.info("Could not compute in-sample metrics (no overlap).")

    # Download forecast CSV
    out_df = pd.DataFrame({'Forecast': pred, 'CI_lower': lower, 'CI_upper': upper})
    b = to_csv_bytes(out_df)
    st.markdown(download_link_bytes(b, filename=f"{ticker}_forecast_{forecast_label}.csv"), unsafe_allow_html=True)

    st.success("Completed. Scroll up to view plots and metrics.")

# Footer notes
st.markdown("---")
st.markdown("**Notes:** If `yfinance` or `pmdarima` are not installed, the app will still work: upload CSV for data and enter manual (p,d,q) for ARIMA. To enable automated fetching and auto_arima, install packages locally: `pip install yfinance pmdarima`.")
