from os import path
import pandas as pd

import fn

class ProcessData:
    """_summary_"""
    def __init__(self, data_name, look_back=3):
        """_summary_

        Args:
            data_name (_type_): _description_
            look_back (int, optional): _description_. Defaults to 3.

        Raises:
            Exception: _description_
        """
        self.data_name = data_name
        raw_data_path = f'data/raw/{data_name}.csv'
        if not path.exists(raw_data_path):
            raise Exception('Raw data file does not exist')
        self.raw_data_path = raw_data_path
        self.look_back = look_back

    def preprocessing_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        df = pd.read_csv(self.raw_data_path, sep='\t')

        # Scaling our data
        data, scaler = fn.scale_data(df)

        self.scaler = scaler
        self.data = data

        # 70/30 split
        data_training = data[:int(data.shape[0]*0.7)]
        data_val = data[int(data.shape[0]*0.7):]

        x_train, y_train = fn.create_sequences(data_training, self.look_back)
        x_test, y_test = fn.create_sequences(data_val, self.look_back)

        return x_train, y_train, x_test, y_test, scaler

    def processed_data(self, predicts, y_test):
        """_summary_

        Args:
            predicts (_type_): _description_
            y_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        i = 0
        data_path = f'data/processed/{self.data_name}{i}.csv'
        while path.exists(data_path):
            i += 1
            data_path = f'data/processed/{self.data_name}{i}.csv'

        feat = self.data.shape[1]
        num = predicts.shape[0]

        # Xóa ở đầu
        # if predicts.shape[1] != feat:
        #     temp = num % feat
        #     predicts = predicts[:int(num-temp)]
        #     y_test = y_test.reshape(-1, 1)
        #     y_test = y_test[:int(num-temp)]

        # Xóa ở cuối
        if predicts.shape[1] != feat:
            temp = num % feat
            predicts = predicts[int(temp):]
            y_test = y_test.reshape(-1, 1)
            y_test = y_test[int(temp):]

        predicts = predicts.reshape(-1, feat)
        y_test = y_test.reshape(-1, feat)
        predicts = self.scaler.inverse_transform(predicts)
        y_test = self.scaler.inverse_transform(y_test)

        predicts = pd.DataFrame({
            'predict': predicts.flatten(),
            'true': y_test.flatten()
        })

        predicts.to_csv(data_path, index=True)

        return data_path