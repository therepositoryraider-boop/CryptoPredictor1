import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class CryptoPricePredictor:
    def __init__(self, model=None):
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = model
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train(self, X, y):
        test_size = int(len(X) * 0.2)
        self.X_train, self.X_test = X[:-test_size], X[-test_size:]
        self.y_train, self.y_test = y[:-test_size], y[-test_size:]
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2

    def predict_next_day(self, last_data_point, n_days=1):
        input_data = last_data_point.copy()
        future_predictions = []

        for i in range(n_days):
            next_price = self.model.predict(input_data)[0]
            future_predictions.append(next_price)
            new_input_data = input_data.iloc[0].copy()
            for j in range(5, 1, -1):
                new_input_data[f'Lag_{j}'] = new_input_data[f'Lag_{j-1}']
            new_input_data['Lag_1'] = next_price
            new_input_data['Close'] = next_price
            current_date = pd.to_datetime(input_data.index[0]) + pd.Timedelta(days=i + 1)
            new_input_data['DayOfWeek'] = current_date.dayofweek
            new_input_data['DayOfMonth'] = current_date.day
            new_input_data['Month'] = current_date.month
            input_data = pd.DataFrame([new_input_data.values], columns=new_input_data.index)
        return np.array(future_predictions)

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    @classmethod
    def load_model(cls, filename):
        predictor = cls()
        predictor.model = joblib.load(filename)
        return predictor