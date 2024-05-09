from os import path
from keras import models
from sklearn.metrics import r2_score

from fn import logger

class ModelEvaluation:
    """_summary_"""
    def __init__(self, model):
        if path.exists(model):
            self.model = models.load_model(model)
        else:
            raise Exception('Model does not exist')

    def evaluate(self, x_test, y_test):
        """_summary_

        Args:
            x_test (_type_): _description_
            y_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        score = self.model.evaluate(x_test, y_test, verbose=0)
        logger(f'Model score: {score}', 'logs/model_evaluation.log')
    
    def r2_score(self, x_test, y_test):
        """_summary_

        Args:
            x_test (_type_): _description_
            y_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        r2 = r2_score(self.model.predict(x_test), y_test)
        logger(f'R2 score: {r2}', 'logs/model_evaluation.log')