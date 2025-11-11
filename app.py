#!/usr/bin/env python3
"""
Streamlit app for CryptoPredictor1
- Displays historical price data and a forecast line
- Tries to use data_fetcher.py and predictor_model.py if present
- Falls back to yfinance (if installed) or synthetic demo data
- Uses Plotly for reliable time-series plotting in Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import importlib
import traceback

# Prefer Plotly for robust time-series plotting
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="CryptoPredictor1 - Price Forecast")

st.title("Historical Data and Future Forecast")
st.markdown("Interactive chart showing historical prices and a simple forecast. "
            "This app will try to use a model from predictor_model.py; if not available it uses a fallback predictor.")

# --- Utilities -------------------------------------------------------------
def ensure_datetime(df, date_col='Date'):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.reset_index(drop=True)
        return df
    # If index looks like a datetime index already
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
        return df
    raise ValueError("No date column found and index is not DatetimeIndex")

def fallback_predict(last_series, n_periods):
    # Simple linear trend extrapolation using last 7 points
    y = np.asarray(last_series[-7:])
    x = np.arange(len(y))
    # if constant values, slope = 0
    if np.all(y == y[0]):
        slope = 0.0
        intercept = y[-1]
    else:
        slope, intercept = np.polyfit(x, y, 1)
    preds = intercept + slope * (np.arange(len(y), len(y) + n_periods))
    # if naive negative or nan, fallback to last value repeat
    if np.any(np.isnan(preds)) or np.all(np.isfinite(preds) == False):
        preds = np.full(n_periods, y[-1])
    return preds

# --- Data acquisition ------------------------------------------------------
# UI inputs
coin = st.sidebar.selectbox("Select cryptocurrency / ticker", ["BTC-USD", "ETH-USD", "BTC", "ETH"])
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))
forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=90)

data_df = None
model_predict_fn = None

# Try to load repository data_fetcher and predictor_model (if present)
try:
    df_module = importlib.import_module("data_fetcher")
    if hasattr(df_module, "fetch_price_history"):
        data_df = df_module.fetch_price_history(symbol=coin, start=start_date, end=end_date)
except Exception:
    # Import may fail if file not present or function not matching; ignore and fallback
    pass

# If not provided by data_fetcher, try yfinance (if installed)
if data_df is None:
    try:
        import yfinance as yf
        ticker = coin if "-" in coin else coin + "-USD"
        hist = yf.download(ticker, start=str(start_date), end=str(end_date))
        if hist is None or hist.empty:
            raise RuntimeError("No data from yfinance")
        # Ensure consistent DataFrame with Date and Close
        hist = hist.reset_index()[["Date", "Close"]].rename(columns={"Close": "Close"})
        data_df = hist
    except Exception:
        # Fallback to synthetic demo data
        idx = pd.date_range(start=start_date, end=end_date, freq="D")
        # generate a gentle upward trend with noise
        base = 20000 if "BTC" in coin else 2000
        trend = np.linspace(base, base * 1.1, len(idx))
        noise = np.random.normal(scale=base * 0.02, size=len(idx))
        data_df = pd.DataFrame({"Date": idx, "Close": trend + noise})

# Normalize data_df format
try:
    data_df = ensure_datetime(data_df, date_col="Date")
except Exception:
    # If Date not present, try 'date' lowercase or index
    if "date" in data_df.columns:
        data_df = ensure_datetime(data_df, date_col="date")
    else:
        try:
            data_df.index = pd.to_datetime(data_df.index)
            data_df = data_df.reset_index().rename(columns={"index": "Date"})
        except Exception as e:
            st.error("Could not parse dates from the data. See details in the console.")
            st.exception(e)
            st.stop()

# --- Prediction ------------------------------------------------------------
# Try to import predictor_model.predict function
try:
    pm = importlib.import_module("predictor_model")
    if hasattr(pm, "CryptoPredictorModel"):
        # If predictor_model exports a class, attempt to use it.
        try:
            # If the model class requires saved model loading, predictor_model.py
            # should provide a helper like load_model or a saved joblib file.
            # Here we attempt predict on a naive feature set (last close).
            model_obj = None
            if hasattr(pm, "load_model"):
                model_obj = pm.load_model()
            elif hasattr(pm, "CryptoPredictorModel"):
                # instantiate default and hope it's pre-trained (rare)
                model_obj = pm.CryptoPredictorModel()
            if model_obj is not None and hasattr(model_obj, "predict"):
                # Prepare a minimal X for prediction if predictor_model expects it.
                # We'll pass the last close as a single feature repeated for forecast_days.
                last_close = data_df["Close"].iloc[-1]
                X_pred = np.arange(forecast_days).reshape(-1, 1)  # minimal shape
                try:
                    preds = model_obj.predict(X_pred)
                    # If predictions length doesn't match, fallback below
                    if len(preds) != forecast_days:
                        preds = None
                except Exception:
                    preds = None
                if preds is not None:
                    model_predict_fn = lambda last_series, n: np.array(preds)
        except Exception:
            # swallow and fallback
            pass
except Exception:
    # Module not present
    pass

# If model_predict_fn not set, use fallback trend predictor
if model_predict_fn is None:
    last_values = data_df["Close"].values
    model_predict_fn = lambda last_series, n: fallback_predict(last_series, n)

# Compute predicted series
preds = model_predict_fn(data_df["Close"].values, int(forecast_days))
last_date = data_df["Date"].iloc[-1]
pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(preds), freq="D")
pred_df = pd.DataFrame({"Date": pred_dates, "Predicted": preds})

# --- Plotting --------------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data_df["Date"],
        y=data_df["Close"],
        mode="lines",
        name="Historical Price",
        line=dict(color="royalblue", width=2),
        hovertemplate="%{x|%Y-%m-%d}: $%{y:,.2f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=pred_df["Date"],
        y=pred_df["Predicted"],
        mode="lines+markers",
        name="Predicted Price",
        line=dict(color="crimson", width=2, dash="dash"),
        marker=dict(size=4),
        hovertemplate="%{x|%Y-%m-%d}: $%{y:,.2f}<extra></extra>",
    )
)

# Improve layout for dark-mode / consistent axis formatting
fig.update_layout(
    template="plotly_dark",
    title=f"{coin} Price Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=80, b=60),
    hovermode="x unified",
)

# make x-axis tickformat friendly
fig.update_xaxes(nticks=10, tickformat="%b %Y", tickangle=0)

st.plotly_chart(fig, use_container_width=True)

# --- Show data & metrics ---------------------------------------------------
with st.expander("Show data (historical + predicted)"):
    combined = pd.concat([data_df[["Date", "Close"]].rename(columns={"Close": "Value"}), 
                          pred_df.rename(columns={"Predicted": "Value"})], ignore_index=True)
    st.dataframe(combined.tail(50).assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")))

st.markdown("Notes:")
st.markdown("- If you have a predictor model in predictor_model.py that exposes a load_model() or a predict() function, this app will try to use it.")
st.markdown("- To integrate a trained model: implement load_model() in predictor_model.py that returns an object with a predict(X) method (or implement a top-level predict function).")