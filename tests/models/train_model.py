import unittest
import pandas as pd
from lightgbm import LGBMClassifier
from src.models.train_model import train_model



class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        # Criar um DataFrame de teste
        df = pd.read_csv("C:\\Users\\danie\\FraudClassifier\\data\\processed\\result_df_train.csv")

        X_train = df.drop('fraude', axis=1)

        print(X_train)

        y_train = df.fraude

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


        return pipe

if __name__ == '__main__':
    unittest.main()
