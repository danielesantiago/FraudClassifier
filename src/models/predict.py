import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from config import (
    MODEL_PATH,
    TEST_DATA_PATH,
    PREDICTIONS_PATH,
    PREDICTIONS_FILE,
    TRAIN_DATA_PATH,
    PREDICTIONS_FILE_TRAIN,
    PREDICTIONS_PATH_TRAIN,
)
from features import preprocess_categoria_produto


def load_model(model_path=MODEL_PATH):
    """
    Carrega o modelo treinado.
    """
    model = joblib.load(model_path)
    print("Modelo carregado com sucesso!")
    return model


def calculate_metrics(y_true, y_pred, y_proba, output_path=PREDICTIONS_FILE):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)

    # Salva as métricas em um arquivo
    with open(output_path, "w") as f:
        for metric, value in metrics.items():
            if isinstance(value, str):  # Para erros ao calcular ROC AUC
                f.write(f"{metric}: {value}\n")
            else:
                f.write(f"{metric}: {value:.4f}\n")

    print(f"Métricas salvas em: {output_path}")


def make_predictions(data_path=TEST_DATA_PATH, threshold=0.61):

    pipeline = load_model()

    data = pd.read_csv(data_path)

    data_preprocessed = preprocess_categoria_produto(data)

    X_test = data_preprocessed.drop(columns=["fraude"], errors="ignore")
    y_true = data_preprocessed.get("fraude", None)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("Probabilidades previstas calculadas com sucesso.")

    y_pred = (y_proba > threshold).astype(int)
    print(f"Predições feitas com sucesso. Threshold: {threshold}")

    # Salva as predições no dataframe
    data["predicted_fraude"] = y_pred
    data["predicted_proba"] = y_proba  # Salva as probabilidades
    if y_true is not None:
        data["true_fraude"] = y_true

    # Salva os resultados em arquivo
    data.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Resultados salvos em: {PREDICTIONS_PATH}")

    # Calcula métricas se y_true existir
    if y_true is not None:
        calculate_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)


def make_predictions_train(data_path=TRAIN_DATA_PATH, threshold=0.61):

    pipeline = load_model()

    data = pd.read_csv(data_path)

    data_preprocessed = preprocess_categoria_produto(data)

    X_test = data_preprocessed.drop(columns=["fraude"], errors="ignore")
    y_true = data_preprocessed.get("fraude", None)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("Probabilidades previstas calculadas com sucesso.")

    y_pred = (y_proba > threshold).astype(int)
    print(f"Predições feitas com sucesso. Threshold: {threshold}")

    # Salva as predições no dataframe
    data["predicted_fraude"] = y_pred
    data["predicted_proba"] = y_proba  # Salva as probabilidades
    if y_true is not None:
        data["true_fraude"] = y_true

    # Salva os resultados em arquivo
    data.to_csv(PREDICTIONS_PATH_TRAIN, index=False)
    print(f"Resultados salvos em: {PREDICTIONS_PATH_TRAIN}")

    # Calcula métricas se y_true existir
    if y_true is not None:
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            output_path=PREDICTIONS_FILE_TRAIN,
        )


if __name__ == "__main__":
    make_predictions()
