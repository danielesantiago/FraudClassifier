import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from config import TRAIN_DATA_PATH, MODEL_PATH
from features import (
    ColumnDropper,
    DataProcessor,
    ScoreImputer,
    OneHotFeatureEncoder,
    preprocess_categoria_produto,
    KFoldTargetEncoder,
)


def get_pipeline():
    """
    Retorna o pipeline de treinamento.
    """
    model = LGBMClassifier(
        class_weight="balanced",
        random_state=1234,
        boosting_type="goss",
        colsample_bytree=1.0,
        learning_rate=0.05,
        max_depth=30,
        min_child_samples=10,
        min_child_weight=1,
        n_estimators=150,
        num_leaves=50,
        reg_alpha=0,
        reg_lambda=5,
        subsample=0.9,
    )

    pipeline = Pipeline(
        [
            ("kfold_encoder", KFoldTargetEncoder()), 
            ("drop_columns", ColumnDropper()),
            ("preprocessor", DataProcessor()),
            ("imputer", ScoreImputer()),
            ("encoder", OneHotFeatureEncoder()),
            ("model", model),
        ]
    )
    return pipeline


def train_model(train_date=TRAIN_DATA_PATH):
    df = pd.read_csv(train_date)

    df_train = preprocess_categoria_produto(df)

    X_train = df_train.drop(columns=["fraude"])
    y_train = df_train["fraude"]

    pipeline = get_pipeline()

    pipeline.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
