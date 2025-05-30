import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

class LSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.last_sequence = None
        self.feature_columns = ['close', 'volume', 'open', 'high', 'low']
        self.close_idx = self.feature_columns.index('close')

    def load_and_preprocess(self, file_paths: dict):
        all_dfs = []
        for name, path in file_paths.items():
            df = pd.read_csv(path, index_col='datetime', parse_dates=True)
            df['company'] = name
            df = df[[f for f in self.feature_columns if f in df.columns] + ['company']]
            all_dfs.append(df)
        self.combined_df = pd.concat(all_dfs).sort_index()
        feature_data = self.combined_df[self.feature_columns].values
        self.scaled_data = self.scaler.fit_transform(feature_data)

    def build_sequences(self):
        X, y = [], []
        company_series = self.combined_df['company'].values
        for company in set(company_series):
            idx = (company_series == company)
            company_scaled = self.scaled_data[idx]
            for i in range(self.sequence_length, len(company_scaled)):
                X.append(company_scaled[i - self.sequence_length:i])
                y.append(company_scaled[i, self.close_idx])
        return np.array(X), np.array(y)

    def train_model(self, X, y, epochs=10, batch_size=64):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, model_path='saved_model/model.keras'):
        self.model.save(model_path)

    def save_scaler(self, scaler_path='saved_model/scaler.pkl'):
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path='saved_model/model.keras'):
        self.model = load_model(model_path)

    def load_scaler(self, scaler_path='saved_model/scaler.pkl'):
        self.scaler = joblib.load(scaler_path)

    def load_data_only(self, file_path):
        df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
        df = df[self.feature_columns]
        data = self.scaler.transform(df.values)
        self.last_sequence = np.array([data[-self.sequence_length:]])

    def predict_next(self):
        if self.model is None or self.last_sequence is None:
            raise ValueError("모델 또는 시퀀스가 로드되지 않았습니다.")
        prediction = self.model.predict(self.last_sequence, verbose=0)
        dummy = np.zeros((1, len(self.feature_columns)))
        dummy[0, self.close_idx] = prediction[0][0]
        restored = self.scaler.inverse_transform(dummy)
        return restored[0, self.close_idx]

    def append_prediction(self, predicted_close):
        dummy = np.zeros((1, len(self.feature_columns)))
        dummy[0, self.close_idx] = predicted_close
        scaled = self.scaler.transform(dummy)
        new_row = scaled[0]
        new_sequence = np.append(self.last_sequence[0, 1:, :], [new_row], axis=0)
        self.last_sequence = np.reshape(new_sequence, (1, self.sequence_length, len(self.feature_columns)))