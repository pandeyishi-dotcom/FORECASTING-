# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64

st.set_page_config(page_title="ARIMA Forecast — RELIANCE", layout="wide")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def download_monthly(ticker: str, start: str, end: str):
    # Use auto_adjust=False to preserve Adj Close presence; we'll pick column fallback later
    df = yf.download(ticker, start=start, end=end, interval="1mo", progress=False, auto_adjust=False)
    if df.empty:
        return df
    # Some yfinance monthly downloads might not include 'Adj Close'
    if 'Adj Close' in df.columns:
        series = df['Adj Close'].dropna().copy()
        price_col = 'Adj Close'
    else:
        series = df['Close'].dropna().copy()
        price_col = 'Close'
    # Align to month start for nicer plotting frequency indexing
    series.index = pd.DatetimeIndex(series.index).to_period('M').to_timestamp()
    series.name = 'Price'
    return series, price_col

def fit_arima(series, seasonal=False, m=12, max_p=5, max_q=5, trace=False):
    # auto_arima to pick order
    stepwise = auto_arima(series, start_p=0, start_q=0,
                          max_p=max_p, max_q=max_q, seasonal=seasonal, m=m if seasonal else 1,
                          trace=trace, error_action='ignore', suppress_warnings=True, stepwise=True)
    order = stepwise.order
    seasonal_order = stepwise.seasonal_order if seasonal else (0,0,0,0)
    # Fit statsmodels ARIMA with chosen order
    model = ARIMA(series, order=order, seasonal_order=seasonal_order if seasonal else (0,0,0,0))
    res = model.fit()
    return stepwise, res

def forecast_and_ci(res, steps):
    fc = res.get_forecast(steps=steps)
    mean = pd.Series(fc.predicted_mean, index=pd.date_range(start=res.model.endog_names, periods=steps))
    # note: we won't use res.model.endog_names as an index; we'll build forecast index outside
    conf = fc.conf_int(alpha=0.05)
    return fc, conf

def series_forecast(res, series_index, steps):
    # Build forecast index as next months starting after last series_index month
    last = pd.to_datetime(series_index.max())
    start = (last + pd.offsets.MonthBegin(1)).to_period('M').to_timestamp()
    forecast_index = pd.date_range(start=start, periods=steps, freq='MS')
    fc = res.get_forecast(steps=steps)
    pred = pd.Series(fc.predicted_mean, index=forecast_index)
    conf = fc.conf_int(alpha=0.05)
    conf_lower = pd.Series(conf.iloc[:,0].values, index=forecast_index)
    conf_upper = pd.Series(conf.iloc[:,1].values, index=forecast_index)
    return pred, conf_lower, conf_upper

def compute_metrics(actual, predicted):
    # align
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

def df_to_csv_bytes(df):
    csv = df.to_csv(index=True)
    return csv.encode('utf-8')

def download_link(df, filename="forecast.csv", text="Download CSV"):
    b = df_to_csv_bytes(df)
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# -------------------------
# UI Layout
# -------------------------
st.title("ARIMA Forecasting — RELIANCE.NS")
st.markdown("This app fetches monthly data from Yahoo Finance, fits an ARIMA (auto_arima) model, "
            "shows fitted vs actuals, and produces a 12-month forecast with 95% confidence intervals.")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="RELIANCE.NS")
    proj = st.selectbox("Project", ["Project 1: 2010-2018 → Forecast 2019", "Project 2: 2021-2025 → Forecast 2026"])
    seasonal = st.checkbox("Use seasonal (SARIMA, m=12)", value=False)
    auto_arima_trace = st.checkbox("Show auto_arima trace", value=False)
    run_btn = st.button("Run / Refit ARIMA")

    st.markdown("---")
    st.write("Advanced")
    max_p = st.number_input("max_p (auto_arima)", min_value=0, max_value=10, value=5)
    max_q = st.number_input("max_q (auto_arima)", min_value=0, max_value=10, value=5)
    st.markdown("Note: `auto_arima` does stepwise selection. Increase max_p/max_q to search more (slower).")

with col2:
    st.header("Output")
    out_area = st.empty()

# -------------------------
# Decide date ranges based on project
# -------------------------
if proj.startswith("Project 1"):
    start = "2010-01-01"
    end = "2018-12-31"
    forecast_label = "Forecast 2019 (12 months)"
else:
    start = "2021-01-01"
    end = "2025-12-31"
    forecast_label = "Forecast 2026 (12 months)"

# -------------------------
# Main run
# -------------------------
if run_btn:
    with st.spinner("Downloading data..."):
        data = download_monthly(ticker, start, end)
    if not data:
        st.error("No data returned from Yahoo Finance. Check ticker or date range.")
        st.stop()
    # data is (series, price_col)
    if isinstance(data, tuple):
        series, used_col = data
    else:
        st.error("Unexpected data shape returned from yfinance.")
        st.stop()

    st.success(f"Downloaded {len(series)} monthly observations ({used_col}). Range: {series.index.min().date()} — {series.index.max().date()}")

    # show price chart
    fig_price = px.line(x=series.index, y=series.values, labels={'x':'Date', 'y':'Price'}, title=f"{ticker} Monthly Price ({start} to {end})")
    st.plotly_chart(fig_price, use_container_width=True)

    # Fit ARIMA
    with st.spinner("Selecting ARIMA order (auto_arima) and fitting model... This may take some time..."):
        try:
            stepwise, res = fit_arima(series, seasonal=seasonal, m=12, max_p=max_p, max_q=max_q, trace=auto_arima_trace)
        except Exception as e:
            st.error(f"ARIMA fit failed: {e}")
            st.stop()

    st.write("### Model selection")
    st.write(f"Auto ARIMA selected order: **{stepwise.order}**  seasonal_order: **{stepwise.seasonal_order}**")
    with st.expander("Model summary (statsmodels)"):
        st.text(res.summary().as_text())

    # In-sample fit (fitted values)
    try:
        fitted = res.predict(start=series.index.min(), end=series.index.max())
        fitted = pd.Series(fitted, index=series.index)
    except Exception as e:
        st.warning("Could not compute in-sample fitted values: " + str(e))
        fitted = pd.Series(index=series.index, dtype=float)

    # Plot actual vs fitted
    fig_fitted = go.Figure()
    fig_fitted.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Actual'))
    if not fitted.empty:
        fig_fitted.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name='ARIMA fitted', line=dict(dash='dash')))
    fig_fitted.update_layout(title=f"{ticker} — Actual vs ARIMA Fitted (in-sample)", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_fitted, use_container_width=True)

    # Forecast next 12 months
    steps = 12
    try:
        pred, conf_lower, conf_upper = series_forecast(res, series.index, steps)
    except Exception as e:
        st.error("Forecast failed: " + str(e))
        st.stop()

    # Plot forecast appended to series
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Actual (train)'))
    fig_fc.add_trace(go.Scatter(x=pred.index, y=pred.values, mode='lines', name='Forecast', line=dict(color='firebrick')))
    fig_fc.add_trace(go.Scatter(x=pred.index, y=conf_upper.values, mode='lines', name='95% CI upper', line=dict(color='lightgrey'), showlegend=False))
    fig_fc.add_trace(go.Scatter(x=pred.index, y=conf_lower.values, mode='lines', name='95% CI lower', line=dict(color='lightgrey'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', showlegend=False))
    fig_fc.update_layout(title=f"{ticker} — {forecast_label}", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_fc, use_container_width=True)

    # Metrics: compare fitted (in-sample) with actuals
    metrics = compute_metrics(series, fitted)
    st.write("### In-sample metrics (Actual vs Fitted)")
    if metrics:
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    else:
        st.info("Could not compute metrics (empty overlap).")

    # Export forecast csv
    out_df = pd.DataFrame({
        'Forecast': pred,
        'CI_lower': conf_lower,
        'CI_upper': conf_upper
    })
    st.markdown("### Download forecast")
    csv_bytes = df_to_csv_bytes(out_df)
    st.download_button(label="Download forecast CSV", data=csv_bytes, file_name=f"{ticker}_forecast_{start[:4]}_{end[:4]}.csv", mime="text/csv")

    st.success("Done. Scroll up to view charts and model summary.")
else:
    st.info("Choose project settings and click **Run / Refit ARIMA** on the left.")

# -------------------------
# Footer / Notes
# -------------------------
st.markdown("---")
st.markdown("""
**Notes**
- This app defaults to `RELIANCE.NS` monthly data from Yahoo Finance (Adj Close if available, otherwise Close).
- `auto_arima` is used to select `(p,d,q)` automatically. Toggle seasonal if you expect yearly seasonality.
- Forecast horizon is 12 months (you can adjust source code to change steps).
- If you run the app repeatedly and want faster experiments, reduce `max_p`/`max_q` in Advanced settings.
""")
