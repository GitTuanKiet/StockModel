import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
import yfinance as yf

os.environ["KERAS_BACKEND"] = "tensorflow"

df = yf.download('GC=F', start='2012-03-03', end='2022-09-09')

if df is None:
    print('Data is None')
    exit()

# Resetting our index to be numbers instead of the date
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis = 1)
# print(df.head())
# print(df.tail())

#Getting Our Moving Averages
#MA100
# ma100 = df.Close.rolling(100).mean()
# print(ma100)  
#MA200
# ma200 = df.Close.rolling(200).mean()
#print(ma200)

#Splitting data into training and testing 70/30
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
data_training = pd.DataFrame(df[0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df[int(len(df)*0.70): int(len(df))])

#Scaling down our training data
scaler = MinMaxScaler(feature_range=(0,1))
#converting to array, scaler.fit_transform auto gives us an array
data_training_array = scaler.fit_transform(data_training)
# print(data_training_array)

#Now we divide our data into an X AND Y train
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    #appending data in our x_train, (i-100 because it should start from 0)
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
#converting our arrays into numpy arrays so we can provide the data to our LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

#Machine Learning Model
# print(x_train.shape)
# print(y_train.shape)

input_shape = (x_train.shape[1], x_train.shape[2])

model = keras.Sequential([
    keras.layers.LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = input_shape),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units = 60, activation = 'relu', return_sequences = True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(units = 80, activation = 'relu', return_sequences = True),
    keras.layers.Dropout(0.4),
    keras.layers.LSTM(units = 120, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units = 1)
])

print(model.summary())

model.compile(
    optimizer='adam', 
    loss= "mean_squared_error"
)

epochs = 5

model.fit(
    x_train,
    y_train,
    epochs=epochs,
)

model.save('models/lstm/model.keras')
