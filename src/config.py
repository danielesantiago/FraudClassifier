from pathlib import Path

# Diretórios principais
PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)  # Define o diretório raiz do projeto com base na localização do arquivo atual
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"

# Caminhos para arquivos específicos
RAW_DATA_PATH = RAW_DIR / "dados.xlsx"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "model_pipeline.pkl"
PREDICTIONS_PATH = PROCESSED_DIR / "result_with_predictions.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "metrics.txt"
