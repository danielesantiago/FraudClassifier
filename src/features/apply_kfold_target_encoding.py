import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold



class KFoldTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.colnames = 'categoria_produto'
        self.targetName = 'fraude'
        self.n_fold = 5
        self.verbosity = True
        self.discardOriginal_col = False

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        assert (type(self.targetName) == str)
        assert (type(self.colnames) == str)
        assert (self.colnames in X.columns)
        assert (self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=42)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean())

        X[col_mean_name].fillna(mean_of_target, inplace=True)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values

        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)

        return X


def apply_kfold_target_encoding(df):
    """
    Aplica K-Fold Target Encoding à coluna 'categoria_produto' para reduzir a cardinalidade e
    preparar os dados para modelagem de machine learning.

    Parâmetros:
    df (pandas.DataFrame): DataFrame contendo a coluna 'categoria_produto' e 'fraude'.

    Retorna:
    pandas.DataFrame: DataFrame após aplicar o K-Fold Target Encoding.
    """

    # Copiando o DataFrame para evitar a modificação do original
    df_copy = df.copy()

    # Contabilizando itens por categoria e fraudes por categoria
    item_por_categoria = _contar_itens_por_categoria(df_copy)
    fraude_por_categoria = _contar_fraudes_por_categoria(df_copy)

    # Mesclando as contagens e calculando a porcentagem cumulativa de fraudes
    df_item_fraude = _mesclar_e_calcular_cumulativo(item_por_categoria, fraude_por_categoria, df_copy)

    # Agrupando categorias com poucas ocorrências em 'Outros'
    df_copy = _agrupar_categorias_menos_frequentes(df_copy, df_item_fraude)

    # Dividindo o DataFrame em conjuntos de treino e teste
    df_train, df_test = split_df(df_copy)

    # Aplicando K-Fold Target Encoding
    df_train, df_test = _aplicar_kfold_target_encoding(df_train, df_test)

    return df_train, df_test


def _contar_itens_por_categoria(df):
    return df['categoria_produto'].value_counts().reset_index().rename(
        columns={"index": "categoria_produto", "categoria_produto": "categoria_produto"})


def _contar_fraudes_por_categoria(df):
    return df.groupby(['categoria_produto']).fraude.sum().reset_index()


def _mesclar_e_calcular_cumulativo(item_por_categoria, fraude_por_categoria, df):
    df_item_fraude = pd.merge(item_por_categoria, fraude_por_categoria,
                              on=['categoria_produto'], how="left")
    df_item_fraude['percent_cumsum_fraude'] = df_item_fraude['fraude'].cumsum() / df.fraude.sum() * 100
    return df_item_fraude


def _agrupar_categorias_menos_frequentes(df, df_item_fraude):
    produtos_categorias = df_item_fraude[1000:]
    lista_categorias_outros = produtos_categorias.categoria_produto.to_list()
    df.loc[df["categoria_produto"].isin(lista_categorias_outros), "categoria_produto"] = "Outros"
    return df


def _aplicar_kfold_target_encoding(df_train, df_test):
    target_encoder = KFoldTargetEncoder()
    df_train_encoded = target_encoder.fit_transform(df_train)
    df_test_encoded = target_encoder.transform(df_test)
    return df_train_encoded, df_test_encoded


def split_df(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_test
