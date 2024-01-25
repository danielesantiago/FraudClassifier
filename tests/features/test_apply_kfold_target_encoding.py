import unittest
import pandas as pd
from src.features.apply_kfold_target_encoding import apply_kfold_target_encoding



class TestApplyKfoldTargetEncoding(unittest.TestCase):

    def test_column_removal(self):
        # Criar um DataFrame de teste
        df = pd.read_excel("C:\\Users\\danie\\FraudClassifier\\data\\raw\\dados.xlsx")
        # Chamar a função `data_preparation`
        result_df_train, result_df_test = apply_kfold_target_encoding(df)

if __name__ == '__main__':
    unittest.main()
