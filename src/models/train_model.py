import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Removendo colunas desnecessárias
        return X.drop(columns=['data_compra', 'produto', 'score_fraude_modelo', 'categoria_produto'], axis=1)


class DataProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Processamento de colunas específicas
        X_copy['is_missing'] = X_copy['entrega_doc_2'].isnull().astype(int)
        X_copy['entrega_doc_2'] = X_copy['entrega_doc_2'].fillna('N').apply(lambda x: 1 if x == 'Y' else 0)
        X_copy['pais'] = X_copy['pais'].apply(lambda x: x if x in ['BR', 'AR'] else 'Outros')
        X_copy['entrega_doc_3'] = X_copy['entrega_doc_3'].apply(lambda x: 1 if x == 'Y' else 0)

        return X_copy


class ScoreImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols = ['score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 'score_8', 'score_9', 'score_10']
        self.imputer = ColumnTransformer(
            transformers=[
                ('score_imputer', SimpleImputer(strategy='median'), self.cols)
            ],
            remainder='passthrough'  # Não modifica as outras colunas
        )

    def fit(self, X, y=None):
        self.imputer.fit(X[self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.cols] = self.imputer.transform(X_copy[self.cols])
        return X_copy


class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)  # Configurar para retornar uma matriz densa
        self.cols = ['score_1', 'pais', 'entrega_doc_1', 'entrega_doc_2', 'entrega_doc_3', 'is_missing']

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):
        # Codificando as colunas
        onehot_data = self.encoder.transform(X[self.cols])

        # Criando um DataFrame com os dados codificados
        onehot_df = pd.DataFrame(onehot_data, columns=self.encoder.get_feature_names_out(self.cols))

        # Resetando o índice do DataFrame original para alinhar com onehot_df
        X_reset = X.reset_index(drop=True)

        # Descartando as colunas originais e concatenando com as colunas codificadas
        return pd.concat([X_reset.drop(self.cols, axis=1), onehot_df], axis=1)


def create_pipeline(model):
    # Criando e retornando o pipeline
    return Pipeline([
        ("dropper", ColumnDropper()),
        ("processor", DataProcessor()),
        ("imputer", ScoreImputer()),
        ("onehot", OneHotFeatureEncoder()),
        ('classifier', model)
    ])


def train_model(X_train, y_train, model):
    # Treinando o modelo
    pipe = create_pipeline(model)
    pipe.fit(X_train, y_train)
    return pipe