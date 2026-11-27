import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing import add_temporal_features, normalize_speed, process_pems  # noqa: E402


def test_temporal_features_are_cyclic():
    df = pd.DataFrame({"timestamp": ["2024-01-01 00:00", "2024-01-06 08:00"], "speed": [70, 40]})
    out = add_temporal_features(df)
    assert np.isclose(out.loc[0, "hour_sin"], 0.0)
    assert out.loc[0, "is_weekend"] == 0      # lundi
    assert out.loc[1, "is_weekend"] == 1      # samedi
    assert out.loc[1, "is_rush_hour"] == 1    # 8 h


def test_speed_is_normalised_between_0_and_1():
    out = normalize_speed(pd.DataFrame({"speed": [10.0, 40.0, 70.0]}))
    assert out["speed_norm"].min() >= 0 and out["speed_norm"].max() <= 1


def test_process_pems_labels_congestion(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame([[70, 30, 65, 20]]).to_csv(raw, header=False, index=False)
    out = process_pems(raw_path=raw, out_path=tmp_path / "out.csv")
    assert out["congestion"].tolist() == [0, 1, 0, 1]   # seuil 60 km/h
