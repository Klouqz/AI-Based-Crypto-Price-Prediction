import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from binance.client import Client
from sklearn.ensemble import RandomForestRegressor

class BinanceModel:
    def __init__(self, pair):
        self.client = Client("", "")
        self.df = self.get_data(pair)
        self.add_rsi()
        self.preprocess_data()
        self.model = RandomForestRegressor(n_estimators=100)

    def get_data(self, pair):
        klines = self.client.get_historical_klines(pair, Client.KLINE_INTERVAL_4HOUR, "730 days ago")
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                           'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                           'Taker buy quote asset volume', 'Ignore'])
        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        return df

    def add_rsi(self):
        delta = self.df['Close'].diff().dropna()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = gain[:14].mean()
        avg_loss = loss[:14].mean()

        gain_loss_ratio = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + gain_loss_ratio))

        for day in range(14, len(delta)):
            gain_day = gain[day]
            loss_day = loss[day]
            avg_gain = ((avg_gain * 13) + gain_day) / 14
            avg_loss = ((avg_loss * 13) + loss_day) / 14
            gain_loss_ratio = avg_gain / avg_loss
            rsi_day = 100 - (100 / (1 + gain_loss_ratio))
            rsi = np.append(rsi, rsi_day)

        rsi = np.insert(rsi, 0, [np.nan] * 14)
        self.df['RSI'] = pd.Series(rsi, index=self.df.index)

    def preprocess_data(self):
        self.df.dropna(subset=['RSI'], inplace=True)
        self.df.drop(columns=['Open time', 'Close time', 'Quote asset volume', 'Number of trades',
                              'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True)

    def train_model(self):
        X = self.df.drop(columns=['Date', 'Close'])
        y = self.df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        print("Model Score:", self.model.score(X_test, y_test))
    def predict(self, X):
        return self.model.predict(X)


def signal(prediction, current_close_price):
    if prediction >= current_close_price:
        return "Long"
    else:
        return "Short"


if __name__ == "__main__":
    pair = "BNBUSDT"
    bn_model = BinanceModel(pair)
    bn_model.train_model()
    try:
        bn_model.df = bn_model.get_data(pair)
        bn_model.add_rsi()
        bn_model.preprocess_data()

        X_latest = bn_model.df.drop(columns=['Date', 'Close']).iloc[-1]
        current_close_price = bn_model.df['Close'].iloc[-1]
        prediction = bn_model.predict([X_latest])[0]
        print("Predicted Close Price (Next 4 Hour):", prediction)
        print("Difference in Percentage:", ((prediction - current_close_price) / current_close_price) * 100)

        trade_signal = signal(prediction, current_close_price)
        print("Trade Signal:", trade_signal)


    except Exception as e:
        print("Error:", e)
        time.sleep(600)