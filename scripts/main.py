import os
from fn import logger
import data_preprocessing
import model_training
import model_evaluation

os.environ["KERAS_BACKEND"] = "tensorflow"

path = 'data/raw/test.csv'
look_back = 10

process_data = data_preprocessing.ProcessData('test', look_back)
x_train, y_train, x_test, y_test, scaler = process_data.preprocessing_data()

# Training our model
def build_lstm_model(x_train, y_train, name, epochs, batch_size):
    """_summary_

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        name (_type_): _description_
        epochs (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = model_training.ModelLSTMTraining(name)
    model_path = model.build(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size
    )
    
    return model_path

tickets = 'GC=F'
epochs = 5
batch_size = 64
model_path = build_lstm_model(x_train, y_train, tickets, epochs, batch_size)

# Evaluating our model
def eval_model(x_test, y_test, model_path):
    """_summary_

    Args:
        x_test (_type_): _description_
        y_test (_type_): _description_
        model_path (_type_): _description_
    """
    model_eval = model_evaluation.ModelEvaluation(model_path)

    model_eval.evaluate(x_test, y_test)

    model_eval.r2_score(x_test, y_test)


eval_model(x_test, y_test, model_path)

# Saving our predictions
from keras import models  # noqa: E402

model = models.load_model(model_path)
predicts = model.predict(x_test)
data_path = process_data.processed_data(predicts, y_test)

logger(f'Predictions saved at {data_path}', 'logs/model_evaluation.log')
