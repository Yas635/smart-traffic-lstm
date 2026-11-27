"""
CLI de prédiction : donne la probabilité de congestion pour une date/heure précise
ou pour les N prochains instants à partir des dernières données disponibles.

Usage:
    python predict.py                        # prédit le prochain instant
    python predict.py --hour 8 --minute 30  # prédit pour 08h30 du prochain jour disponible
    python predict.py --next 6              # prédit les 6 prochains pas (30 min)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    CONGESTION_THRESHOLD, DATA_PATH, FEATURES, MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH,
)
from data_processing import add_temporal_features
from train_model import LSTMModel, load_model


def build_feature_row(timestamp: pd.Timestamp, speed: float, speed_stats: dict) -> np.ndarray:
    vmin, vmax = speed_stats["vmin"], speed_stats["vmax"]
    speed_norm = (speed - vmin) / (vmax - vmin + 1e-8)

    h = timestamp.hour
    dow = timestamp.dayofweek
    row = {
        "speed_norm": speed_norm,
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": int(dow >= 5),
        "is_rush_hour": int(7 <= h <= 9 or 17 <= h <= 19),
    }
    return np.array([row[f] for f in FEATURES], dtype=np.float32)


def load_data_and_scaler():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "speed_norm" not in df.columns:
        from data_processing import load_and_enrich
        df = load_and_enrich()

    with open(SCALER_PATH, "rb") as fh:
        speed_stats = pickle.load(fh)

    return df, speed_stats


def predict_next(n_steps: int = 1):
    """Prédit les `n_steps` prochains instants en prolongeant la série."""
    df, speed_stats = load_data_and_scaler()
    model = load_model()

    # Séquence de départ : derniers SEQUENCE_LENGTH points
    seq = df[FEATURES].values[-SEQUENCE_LENGTH:].astype(np.float32)
    last_ts = df["timestamp"].iloc[-1]
    last_speed = float(df["speed"].iloc[-1])

    results = []
    for step in range(n_steps):
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = model(x).item()

        next_ts = last_ts + pd.Timedelta(minutes=5 * (step + 1))
        results.append({"timestamp": next_ts, "prob": prob, "congestion": int(prob > 0.5)})

        # Avancer la séquence (vitesse inchangée — prédiction conservative)
        new_row = build_feature_row(next_ts, last_speed, speed_stats)
        seq = np.vstack([seq[1:], new_row])

    return results


def predict_at(hour: int, minute: int):
    """Prédit la congestion pour une heure/minute donnée (aujourd'hui dans les données)."""
    df, speed_stats = load_data_and_scaler()
    model = load_model()

    target_min = hour * 60 + minute
    df["min_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    match = df[df["min_of_day"] == target_min]

    if match.empty:
        print(f"❌ Aucune donnée disponible pour {hour:02d}:{minute:02d}")
        sys.exit(1)

    # Prend le dernier jour disponible à cette heure
    idx = match.index[-1]
    if idx < SEQUENCE_LENGTH:
        print("❌ Pas assez d'historique avant ce point")
        sys.exit(1)

    seq = df[FEATURES].values[idx - SEQUENCE_LENGTH: idx].astype(np.float32)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = model(x).item()

    ts = df["timestamp"].iloc[idx]
    return [{"timestamp": ts, "prob": prob, "congestion": int(prob > 0.5)}]


def display(results: list):
    print("\n" + "─" * 50)
    for r in results:
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        prob = r["prob"] * 100
        icon = "🚨 CONGESTION" if r["congestion"] else "✅ Fluide    "
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"  {ts}  {bar}  {prob:5.1f}%  {icon}")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction de congestion trafic")
    parser.add_argument("--hour", type=int, default=None, help="Heure (0-23)")
    parser.add_argument("--minute", type=int, default=0, help="Minute (0-59)")
    parser.add_argument("--next", type=int, default=1, help="Nombre de pas futurs à prédire")
    args = parser.parse_args()

    if args.hour is not None:
        res = predict_at(args.hour, args.minute)
    else:
        res = predict_next(args.next)

    display(res)
