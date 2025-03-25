import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica K-Fold Target Encoding para variáveis categóricas.
    """

    def __init__(self, colnames="categoria_produto", target_name="fraude", n_fold=5):
        self.colnames = colnames
        self.target_name = target_name
        self.n_fold = n_fold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean_target = X[self.target_name].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=42)

        col_mean_name = f"{self.colnames}_Kfold_Target_Enc"
        X[col_mean_name] = np.nan

        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            X.loc[X.index[val_idx], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.target_name].mean()
            )
        X[col_mean_name].fillna(mean_target, inplace=True)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Remove colunas desnecessárias.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            columns=[
                "data_compra",
                "produto",
                "score_fraude_modelo",
                "categoria_produto",
            ],
            errors="ignore"  # Ignora a ausência de colunas
        )


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Processa e transforma os dados.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Verifica e transforma a coluna 'entrega_doc_2'
        if "entrega_doc_2" in X_copy.columns:
            X_copy["is_missing"] = X_copy["entrega_doc_2"].isnull().astype(int)
            X_copy["entrega_doc_2"] = (
                X_copy["entrega_doc_2"].fillna("N").apply(lambda x: 1 if x == "Y" else 0)
            )
        
        # Verifica e transforma a coluna 'pais'
        if "pais" in X_copy.columns:
            X_copy["pais"] = X_copy["pais"].apply(
                lambda x: x if x in ["BR", "AR"] else "Outros"
            )
        
        # Verifica e transforma a coluna 'entrega_doc_3'
        if "entrega_doc_3" in X_copy.columns:
            X_copy["entrega_doc_3"] = X_copy["entrega_doc_3"].apply(
                lambda x: 1 if x == "Y" else 0
            )
        
        return X_copy



class ScoreImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores ausentes nas colunas de score.
    """

    def __init__(self):
        self.imputers = {}

    def fit(self, X, y=None):
        cols = [f"score_{i}" for i in range(2, 11)]
        for col in cols:
            imputer = SimpleImputer(strategy="median")
            imputer.fit(X[[col]])
            self.imputers[col] = imputer
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, imputer in self.imputers.items():
            X_copy[col] = imputer.transform(X_copy[[col]])
        return X_copy


class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica One-Hot Encoding em colunas categóricas.
    """

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.cols = [
            "score_1",
            "pais",
            "entrega_doc_1",
            "entrega_doc_2",
            "entrega_doc_3",
            "is_missing",
        ]

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):
        onehot_data = self.encoder.transform(X[self.cols])
        onehot_df = pd.DataFrame(
            onehot_data, columns=self.encoder.get_feature_names_out(self.cols)
        )
        return pd.concat(
            [X.reset_index(drop=True).drop(self.cols, axis=1), onehot_df], axis=1
        )


def preprocess_categoria_produto(df, min_threshold=1000):
    """
    Categoriza como outros as categorias de produtos menos frequentes
    """

    categoria_counts = df["categoria_produto"].value_counts()
    categorias_menos_frequentes = categoria_counts[
        categoria_counts < min_threshold
    ].index
    df["categoria_produto"] = df["categoria_produto"].replace(
        categorias_menos_frequentes, "Outros"
    )
    return df
