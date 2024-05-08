import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers, Sequential
import yfinance as yf
from sklearn.metrics import r2_score

os.environ["KERAS_BACKEND"] = "tensorflow"

tickers = 'GC=F'
start = '2012-03-03'
end = '2022-09-09'

df = yf.download(tickers, start=start, end=end)

df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis = 1)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

#Selecting our features
selected_features = ['Close']
df = df[selected_features]

#Splitting data into training and testing 70/30
data_training = pd.DataFrame(df[0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df[int(len(df)*0.70): int(len(df))])

#Scaling down our training data
scaler = MinMaxScaler(feature_range=(0,1))

#converting to array, scaler.fit_transform auto gives us an array
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)

look_back = 100
def create_sequences(data, look_back):
    x, y = [], []

    for i in range(look_back, data.shape[0]):
        x.append(data[i-look_back: i])
        y.append(data[i, 0])

    return np.array(x), np.array(y)

x_train, y_train = create_sequences(data_training_array, look_back)
x_test, y_test = create_sequences(data_testing_array, look_back)

def build_lstm_model(input_shape):
    model = Sequential([
        layers.LSTM(units=40, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(units=60, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(units=80, return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(units=120),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    return model

input_shape = (x_train.shape[1], x_train.shape[2])

model = build_lstm_model(input_shape)

def train_model(model, x_train, y_train, epochs):
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
    )

    return model

epochs = 5

model = train_model(model, x_train, y_train, epochs)

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)

    r2score = r2_score(y_test, model.predict(x_test))

    return score, r2score

score, r2score = evaluate_model(model, x_test, y_test)

print('Score:', score)
print('R2 Score:', r2score)

def save_model(model, path, tickers):
    model.save(f'{path}/{tickers}.keras')

path = 'models/lstm'

save_model(model, path, tickers)