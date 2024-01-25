import pandas as pd
from src.features.apply_kfold_target_encoding import apply_kfold_target_encoding

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    # Lê o conjunto de dados
    df = pd.read_excel("C:\\Users\\danie\\FraudClassifier\\data\\raw\\dados.xlsx")

    result_df_train, result_df_test = apply_kfold_target_encoding(df)

    # Salva o conjunto de dados processado
    result_df_train.to_csv("C:\\Users\\danie\\FraudClassifier\\data\\processed\\result_df_train.csv", index=False)
    result_df_test.to_csv("C:\\Users\\danie\\FraudClassifier\\data\\processed\\result_df_test.csv", index=False)


if __name__ == '__main__':
    main()
