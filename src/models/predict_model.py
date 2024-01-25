import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score
from joblib import dump

def predict_model(X_test, y_test, model):
    # Previsões do modelo e probabilidades
    y_pred = model.predict(X_test)
    probs_test = model.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva

    # Calcula as métricas de desempenho
    loss = log_loss(y_test, probs_test)
    roc_auc = roc_auc_score(y_test, probs_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Imprime as métricas de desempenho
    print(f'Log Loss: {loss:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')

    dump(model, "C:\\Users\\danie\\FraudClassifier\\models\\lightgbm.joblib")
