from keras import layers, Sequential

from fn import logger

class ModelTraining:
    def __init__(self, model_name, file_name):
        self.model_name = model_name
        self.file_name = file_name
        self.path = f'models/{self.model_name}/{self.file_name}.keras'

    def build(self):
        pass
    

class ModelLSTMTraining(ModelTraining):
    """_summary_

    Args:
        ModelTraining (_type_): _description_
    """
    def __init__(self, file_name):
        super().__init__('lstm', file_name)

    def build(
        self,
        train_array,
        val_array,
        epochs,
        batch_size,
    ):
        """_summary_

        Args:
            train_array (_type_): _description_
            val_array (_type_): _description_
            epochs (_type_): _description_
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_shape = (train_array.shape[1], train_array.shape[2])
        model = Sequential()

        model.add(layers.Input(shape=input_shape))
        model.add(layers.LSTM(units=40, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(units=60, return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.LSTM(units=80, return_sequences=True))
        model.add(layers.Dropout(0.4))
        model.add(layers.LSTM(units=120))
        model.add(layers.Dropout(0.6))
        model.add(layers.Dense(1))

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error'
        )

        model.fit(
            train_array,
            val_array,
            epochs=epochs,
            batch_size=batch_size
        )

        model.save(self.path)

        self.model = model
        
        logger(f'Model saved at {self.path}', 'logs/model_training.log')
        return self.path