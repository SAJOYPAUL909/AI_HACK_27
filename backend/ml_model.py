import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("energy_optim.ml")

def create_lag_features(df, value_col='energy_kwh', lags=24):
    df = df.copy().sort_values('timestamp')
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df[value_col].shift(lag)
    df['hour'] = df['timestamp'].dt.hour
    df = df.dropna().reset_index(drop=True)
    return df

class Forecaster:
    def __init__(self, lags=24):
        self.lags = lags
        self.model = GradientBoostingRegressor()

    def train(self, df, value_col='energy_kwh'):
        Xy = create_lag_features(df, value_col, lags=self.lags)
        X = Xy[[c for c in Xy.columns if c.startswith('lag_')] + ['hour']]
        y = Xy[value_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        logger.info('Forecaster R2: %s', score)
        return float(score)

    def predict_next_n(self, df, n_steps=24, value_col='energy_kwh'):
        df_copy = df.copy().sort_values('timestamp').reset_index(drop=True)
        preds = []
        last = df_copy[value_col].tolist()
        for step in range(n_steps):
            lags = last[-self.lags:]
            if len(lags) < self.lags:
                lags = [np.mean(last)] * (self.lags - len(lags)) + lags
            hour = (df_copy['timestamp'].iloc[-1] + pd.Timedelta(hours=step + 1)).hour
            X = np.array(lags[::-1] + [hour]).reshape(1, -1)
            yhat = float(self.model.predict(X)[0])
            preds.append(yhat)
            last.append(yhat)
        start = df_copy['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
        ts = pd.date_range(start, periods=n_steps, freq='H')
        return pd.DataFrame({'timestamp': ts, 'predicted': preds})

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.02, random_state=42)

    def fit(self, df, value_col='energy_kwh'):
        X = df[[value_col]].values
        self.model.fit(X)

    def detect(self, df, value_col='energy_kwh'):
        X = df[[value_col]].values
        preds = self.model.predict(X)
        out = df.copy()
        out['anomaly'] = (preds == -1)
        return out
