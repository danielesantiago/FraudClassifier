import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from features import KFoldTargetEncoder


def load_data(input_path=RAW_DATA_PATH):
    """
    Carrega os dados e retorna um DataFrame
    """
    return pd.read_excel(input_path)


def split_df(df, test_size=0.2, random_state=42):
    """
    Divide um DataFrame em conjuntos de treino e teste
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)


def save_splits(
    df_train, df_test, train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH
):
    """
    Salva os conjuntos de treino e teste em arquivos CSV
    """
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)


def main():
    df = load_data()
    df_train, df_test = split_df(df)

    targetc = KFoldTargetEncoder()
    df_train = targetc.fit_transform(df_train)
    df_test = targetc.transform(df_test)

    save_splits(df_train, df_test)

    print(f"Dados divididos.\nTreino: {TRAIN_DATA_PATH}\nTeste: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
