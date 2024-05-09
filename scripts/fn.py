from os import path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def create_sequences(data, look_back, target_column=0):
    """_summary_

    Args:
        data (_type_): _description_
        look_back (_type_): _description_
        target_column (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    x, y = [], []

    for i in range(look_back, data.shape[0]):
        x.append(data[i-look_back: i])
        y.append(data[i, target_column])

    return np.array(x), np.array(y)

def scale_data(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)
    return data, scaler

def logger(message, file_path):
    """_summary_

    Args:
        message (_type_): _description_
    """
    print(message)
    message = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + message
    if path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(message + '\n')
    else:
        with open(file_path, 'w') as f:
            f.write(message + '\n')
