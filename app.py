import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_fetcher import fetch_and_prepare_data
from predictor_model import CryptoPricePredictor

st.set_page_config(page_title="Crypto Price Predictor (ML)", layout="wide")
st.title("₿ Crypto Price Predictor (Traditional ML)")
st.caption("Predict BTC/ETH prices using Random Forest Regressor.")

# --- Sidebar ---
st.sidebar.header("Configuration")
TICKER_OPTIONS = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(TICKER_OPTIONS.keys()))
ticker_symbol = TICKER_OPTIONS[selected_crypto]

# --- Prediction horizon new selector ---
st.sidebar.subheader("Prediction Horizon")
horizon_mode = st.sidebar.radio("Predict for (choose one):", ["Days", "Months", "Years"])
if horizon_mode == "Days":
    n_units = st.sidebar.slider("How many days?", 1, 30, 7)
    n_pred_days = n_units
elif horizon_mode == "Months":
    n_units = st.sidebar.slider("How many months?", 1, 24, 1)
    n_pred_days = n_units * 30
elif horizon_mode == "Years":
    n_units = st.sidebar.slider("How many years?", 1, 5, 1)
    n_pred_days = n_units * 365

DATA_LOOKBACK = st.sidebar.slider("Training Data Length (years)", 1, 5, 3)

@st.cache_data
def train_and_cache(ticker, lookback):
    X, y = fetch_and_prepare_data(ticker, lookback * 365)
    if X is None:
        return None, None, None, None
    predictor = CryptoPricePredictor()
    mse, r2 = predictor.train(X, y)
    last_data_point = X.iloc[[-1]]
    return X, y, predictor, last_data_point, mse, r2

if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching data & training model..."):
        X, y, predictor, last_data_point, mse, r2 = train_and_cache(
            ticker_symbol, DATA_LOOKBACK
        )
    if X is None:
        st.error(f"Could not fetch data for {ticker_symbol}.")
        st.stop()
    with st.spinner(f"Generating {n_pred_days} day forecast..."):
        preds = predictor.predict_next_day(last_data_point, n_days=n_pred_days)
    last_date = X.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred_days)
    forecast_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Close': preds
    }).set_index('Date')
    st.subheader(f"Forecast for {selected_crypto}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Next Predicted Price",
            value=f"${preds[0]:,.2f}"
        )
        st.info(f"R² Score on Test Set: **{r2:.4f}**")
    with col2:
        st.metric(
            label="Price at End of Horizon",
            value=f"${preds[-1]:,.2f}"
        )
        st.info(f"MSE on Test Set: **{mse:,.2f}**")
    st.subheader("Historical Data and Future Forecast")
    historical_close = X['Close']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_close.index, y=historical_close.values, mode='lines', name='Historical Price', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['Predicted_Close'].values, mode='lines+markers',
        name='Predicted Price', line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f'{selected_crypto} Price Forecast',
        xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Raw Prediction Data")
    st.dataframe(forecast_df)
else:
    st.info("Pick options and then click 'Run Prediction'.")
