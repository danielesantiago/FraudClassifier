from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from typing import List, Optional, Any
import pandas as pd
import joblib
import os

# Importa a função de pré-processamento
from src.features import preprocess_categoria_produto
from src.models.predict import load_model

app = FastAPI(title="Fraud Classifier API")

# Caminho do modelo definido no treinamento
MODEL_PATH = os.path.join("FraudClassifier", "models", "model_pipeline.pkl")

# Carregar o modelo
try:
    model = load_model(model_path=MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")



class DataItem(BaseModel):
    score_1: int
    score_2: float
    score_3: float
    score_4: float
    score_5: float
    score_6: float
    pais: str
    score_7: int
    produto: str
    categoria_produto: str
    score_8: float
    score_9: float
    score_10: float
    entrega_doc_1: int
    entrega_doc_2: Optional[str]  # Pode ser string como "Y", "N", ou nulo
    entrega_doc_3: Optional[str]
    data_compra: datetime
    valor_compra: float
    score_fraude_modelo: int

class PredictionRequest(BaseModel):
    data: List[DataItem]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Converte a lista de dados para DataFrame
        df_input = pd.DataFrame([item.dict() for item in request.data])
        
        # Aplica o pré-processamento
        df_processed = preprocess_categoria_produto(df_input.copy())
        
        # Remove coluna alvo se existir
        if "fraude" in df_processed.columns:
            X = df_processed.drop(columns=["fraude"])
        else:
            X = df_processed

        # Realiza a predição: se o modelo tem predict_proba, use-o
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
        raise HTTPException(status_code=500, detail=str(e))
