import unittest
import pandas as pd
from lightgbm import LGBMClassifier
from src.models.train_model import train_model
from src.models.predict_model import predict_model



class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        # Criar um DataFrame de teste
        df = pd.read_csv("C:\\Users\\danie\\FraudClassifier\\data\\processed\\result_df_train.csv")

        X_train = df.drop('fraude', axis=1)
        y_train = df.fraude

        df = pd.read_csv("C:\\Users\\danie\\FraudClassifier\\data\\processed\\result_df_test.csv")

        X_test = df.drop('fraude', axis=1)
        y_test = df.fraude

        # Parâmetros do modelo
        params = {
            'subsample': 0.9,
            'reg_lambda': 5,
            'reg_alpha': 0,
            'num_leaves': 50,
            'n_estimators': 150,
            'min_child_weight': 1,
            'min_child_samples': 10,
            'max_depth': 30,
            'learning_rate': 0.05,
            'colsample_bytree': 1.0,
            'boosting_type': 'goss'
        }

        # Criando o modelo LightGBM com os parâmetros especificados
        model = LGBMClassifier(**params)
        # Chamar a função `train_model`
        pipe = train_model(X_train, y_train, model)

        predict_model(X_test, y_test, pipe)


if __name__ == '__main__':
    unittest.main()
