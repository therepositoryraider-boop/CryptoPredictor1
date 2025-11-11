import yfinance as yf
import pandas as pd

def fetch_and_prepare_data(ticker="BTC-USD", days_to_look_back=365*3):
    """Fetches historical data and creates features for ML."""
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=days_to_look_back)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None, None

    df = data[['Close']].copy()
    df['Target'] = df['Close'].shift(-1)
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    df['SMA_10'] = df['Close'].rolling(window=10).mean().shift(1)
    df['SMA_30'] = df['Close'].rolling(window=30).mean().shift(1)
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['Month'] = df.index.month
    df.dropna(inplace=True)
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y