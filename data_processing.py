import numpy as np
import pandas as pd

from config import CONGESTION_THRESHOLD, DATA_PATH, RAW_DATA_PATH


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    h = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek

    # Encodage cyclique pour heure et jour (évite la discontinuité 23→0)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_rush_hour"] = h.apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)
    return df


def normalize_speed(df: pd.DataFrame) -> pd.DataFrame:
    vmin, vmax = df["speed"].min(), df["speed"].max()
    df["speed_norm"] = (df["speed"] - vmin) / (vmax - vmin + 1e-8)
    return df


def process_pems(raw_path=None, out_path=None) -> pd.DataFrame:
    raw_path = raw_path or RAW_DATA_PATH
    out_path = out_path or DATA_PATH

    print(f"📥 Lecture de {raw_path}...")
    raw = pd.read_csv(raw_path, header=None)

    speeds = raw.iloc[0].values.astype(float)

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=len(speeds), freq="5min"),
        "speed": speeds,
    })

    df["congestion"] = (df["speed"] < CONGESTION_THRESHOLD).astype(int)
    df = add_temporal_features(df)
    df = normalize_speed(df)

    cols = ["timestamp", "speed", "speed_norm",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "is_weekend", "is_rush_hour", "congestion"]
    df = df[cols]

    df.to_csv(out_path, index=False)

    n = len(df)
    nc = df["congestion"].sum()
    print(f"✅ {n:,} lignes sauvegardées — congestion: {nc:,} ({nc/n*100:.1f}%)")
    return df


def load_and_enrich(csv_path=None) -> pd.DataFrame:
    """Charge le CSV existant et ajoute les features manquantes si nécessaire."""
    csv_path = csv_path or DATA_PATH
    df = pd.read_csv(csv_path)

    if "speed_norm" not in df.columns:
        df = normalize_speed(df)
    if "hour_sin" not in df.columns:
        df = add_temporal_features(df)
        df["congestion"] = (df["speed"] < CONGESTION_THRESHOLD).astype(int)
        df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    process_pems()
