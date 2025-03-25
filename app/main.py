from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="Fraud Classifier API")

# Caminho absoluto para o modelo (evita erro se rodar de fora da pasta)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")

# Carrega o modelo
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

# Pré-processamento simples
def preprocess_categoria_produto(df: pd.DataFrame, min_threshold: int = 1000) -> pd.DataFrame:
    """
    Categoriza como 'Outros' as categorias de produtos com frequência menor que o limiar.
    """
    categoria_counts = df["categoria_produto"].value_counts()
    categorias_menos_frequentes = categoria_counts[categoria_counts < min_threshold].index
    df["categoria_produto"] = df["categoria_produto"].replace(categorias_menos_frequentes, "Outros")
    return df

# Schema de entrada de um item
class DataItem(BaseModel):
    score_1: int
    score_2: float
    score_3: float
    score_4: float
    score_5: float
    score_6: float
    pais: str
    score_7: int
    categoria_produto: str
    score_8: float
    score_9: float
    score_10: float
    entrega_doc_1: int
    entrega_doc_2: Optional[str]
    entrega_doc_3: Optional[str]
    valor_compra: float

# Schema da requisição (lista de itens)
class PredictionRequest(BaseModel):
    data: List[DataItem]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Converte entrada em DataFrame
        df_input = pd.DataFrame([item.dict() for item in request.data])

        # Aplica pré-processamento
        df_processed = preprocess_categoria_produto(df_input.copy())

        # Remove coluna alvo se existir
        X = df_processed.drop(columns=["fraude"], errors="ignore")

        # Predição
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba > 0.61).astype(int)
            results = [
                {"prediction": int(pred), "probability": float(proba)}
                for pred, proba in zip(y_pred, y_proba)
            ]
        else:
            y_pred = model.predict(X)
            results = [{"prediction": int(pred)} for pred in y_pred]

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer predição: {e}")
