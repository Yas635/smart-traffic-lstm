import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
)

from config import (
    BATCH_SIZE, DATA_PATH, DROPOUT, EARLY_STOPPING_PATIENCE, EPOCHS,
    FEATURES, HIDDEN_SIZE, LEARNING_RATE, MODEL_PATH, NUM_LAYERS,
    RANDOM_SEED, SCALER_PATH, SEQUENCE_LENGTH, TARGET, TRAIN_RATIO,
)
from data_processing import load_and_enrich

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Dataset ──────────────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 car bidirectionnel
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.sigmoid(self.fc(out))


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class EarlyStopping:
    def __init__(self, patience: int, model_path):
        self.patience = patience
        self.model_path = model_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("📥 Chargement des données...")
    df = load_and_enrich(DATA_PATH)

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    # Split temporel (pas de shuffle pour préserver la causalité)
    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)

    train_loader = DataLoader(
        TrafficDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TrafficDataset(X_test_seq, y_test_seq), batch_size=BATCH_SIZE
    )

    model = LSTMModel(input_size=len(FEATURES))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, model_path=MODEL_PATH)

    print(f"🚀 Entraînement — {len(FEATURES)} features, seq={SEQUENCE_LENGTH}, "
          f"hidden={HIDDEN_SIZE}x{NUM_LAYERS} (bidirectionnel)")
    print(f"   Train: {len(X_train_seq):,}  |  Val: {len(X_test_seq):,}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        # -- train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb).squeeze(-1), yb).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if stopper.step(val_loss, model):
            print(f"⏹  Early stopping à l'epoch {epoch}")
            break

    stopper.restore(model)

    # ── Évaluation ────────────────────────────────────────────────────────────
    model.eval()
    preds_prob, true = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds_prob.extend(model(Xb).squeeze(-1).numpy())
            true.extend(yb.numpy())

    preds_prob = np.array(preds_prob)
    true = np.array(true)
    preds = (preds_prob > 0.5).astype(int)

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, zero_division=0)

    # Référence naïve : prédire que l'état de congestion ne change pas.
    # Le modèle n'a d'intérêt que s'il fait mieux que cette référence.
    baseline = y_test[SEQUENCE_LENGTH - 1:-1].astype(int)
    base_acc = accuracy_score(true, baseline)
    base_f1 = f1_score(true, baseline, zero_division=0)
    try:
        auc = roc_auc_score(true, preds_prob)
    except ValueError:
        auc = float("nan")

    print("\n" + "=" * 55)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Référence (répéter l'état précédent) : "
          f"accuracy={base_acc:.4f}  F1={base_f1:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(true, preds))
    print("\nRapport détaillé :")
    print(classification_report(true, preds, zero_division=0))
    print("=" * 55)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": len(FEATURES),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "features": FEATURES,
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "baseline_accuracy": float(base_acc),
        "baseline_f1": float(base_f1),
    }, MODEL_PATH)

    # Sauvegarde des stats de normalisation (min/max speed) pour predict
    speed_stats = {"vmin": float(df["speed"].min()), "vmax": float(df["speed"].max())}
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(speed_stats, fh)

    print(f"\n💾 Modèle → {MODEL_PATH}")
    print(f"💾 Scaler → {SCALER_PATH}")


def load_checkpoint() -> dict:
    """Charge le checkpoint ; repli pour d'anciens fichiers (scalaires numpy)."""
    try:
        return torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception:
        # Fichier local de confiance produit par train_model.py
        return torch.load(MODEL_PATH, map_location="cpu", weights_only=False)


def load_model() -> LSTMModel:
    checkpoint = load_checkpoint()
    model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    train()
