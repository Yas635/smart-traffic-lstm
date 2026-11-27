from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "traffic_timeseries.csv"
RAW_DATA_PATH = DATA_DIR / "pems-12w.csv"
MODEL_PATH = MODEL_DIR / "lstm_model.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Seuil de congestion (km/h)
CONGESTION_THRESHOLD = 60.0

# Features utilisées par le modèle
FEATURES = ["speed_norm", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_rush_hour"]
TARGET = "congestion"

# Hyperparamètres LSTM
SEQUENCE_LENGTH = 12      # 1 heure (12 x 5 min)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 6

TRAIN_RATIO = 0.8

RANDOM_SEED = 42
