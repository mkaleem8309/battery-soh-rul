import pandas as pd
import numpy as np
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in ("src", "outputs"):
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "per_cell_drivers.csv")

STRESSOR_COLUMNS = [
    "cumulative_time_above_40C",
    "discharge_depth",
    "c_rate",
    "internal_resistance_proxy",
]


def load_features():
    return pd.read_csv(FEATURES_PATH)


def compute_cell_driver_scores(cell_df: pd.DataFrame) -> dict:
    scores = {}
    for col in STRESSOR_COLUMNS:
        if col not in cell_df.columns or cell_df[col].nunique() < 2:
            scores[col] = 0.0
            continue
        corr = cell_df[col].corr(cell_df["soh_ground_truth"])
        if pd.isna(corr):
            scores[col] = 0.0
        else:
            scores[col] = abs(corr) if corr < 0 else 0.0

    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}
    else:
        scores = {k: 0.0 for k in scores}

    return scores


def build_per_cell_drivers(features_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, cell_df in features_df.groupby("cell_id"):
        scores = compute_cell_driver_scores(cell_df)
        if all(v == 0 for v in scores.values()):
            top_driver = "inconclusive"
        else:
            top_driver = max(scores, key=scores.get)
        rows.append({
            "cell_id": cell_id,
            "top_driver": top_driver,
            "driver_importance_scores": json.dumps(scores),
        })
    return pd.DataFrame(rows)


def main():
    features_df = load_features()
    result_df = build_per_cell_drivers(features_df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("=== Per-Cell Top Drivers ===")
    print(result_df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")
    print("Shape matches merge_driver_scores(): cell_id, top_driver, driver_importance_scores")


if __name__ == "__main__":
    main()