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
FLEET_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "outputs", "feature_importances.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "per_cell_drivers.csv")

STRESSOR_COLUMNS = [
    "cumulative_time_above_40C",
    "discharge_depth",
    "c_rate",
    "internal_resistance_proxy",
]

MIN_CORRELATION_STRENGTH = 0.35  # Threshold below which per-cell correlation is blended with fleet importances


def load_features():
    return pd.read_csv(FEATURES_PATH)


def load_fleet_importances():
    if os.path.exists(FLEET_IMPORTANCE_PATH):
        df = pd.read_csv(FLEET_IMPORTANCE_PATH)
        return dict(zip(df["Feature"], df["Importance"]))
    return {
        "cumulative_time_above_40C": 0.85,
        "cumulative_cycle_count": 0.10,
        "charge_time": 0.02,
        "c_rate": 0.015,
        "internal_resistance_proxy": 0.01,
        "discharge_depth": 0.005,
    }


def compute_cell_driver_scores(cell_df: pd.DataFrame, fleet_dict: dict = None) -> dict:
    if fleet_dict is None:
        fleet_dict = load_fleet_importances()

    raw_scores = {}
    valid_cols = []
    for col in STRESSOR_COLUMNS:
        if col not in cell_df.columns or cell_df[col].nunique() < 2:
            raw_scores[col] = 0.0
            continue
        valid_cols.append(col)
        corr = cell_df[col].corr(cell_df["soh_ground_truth"])
        if pd.isna(corr):
            raw_scores[col] = 0.0
        else:
            raw_scores[col] = abs(corr) if corr < 0 else 0.0

    max_corr = max(raw_scores.values()) if raw_scores else 0.0
    sum_local = sum(raw_scores.values())

    if sum_local > 0:
        local_norm = {k: v / sum_local for k, v in raw_scores.items()}
    else:
        local_norm = {k: 0.0 for k in STRESSOR_COLUMNS}

    if max_corr >= MIN_CORRELATION_STRENGTH:
        return {k: round(v, 4) for k, v in local_norm.items()}

    # Soft degradation fallback: blend local score with fleet-wide importances
    alpha = max_corr / MIN_CORRELATION_STRENGTH if MIN_CORRELATION_STRENGTH > 0 else 1.0

    fleet_raw = {k: (fleet_dict.get(k, 0.0) if k in valid_cols else 0.0) for k in STRESSOR_COLUMNS}
    sum_fleet = sum(fleet_raw.values())
    if sum_fleet > 0:
        fleet_norm = {k: v / sum_fleet for k, v in fleet_raw.items()}
    else:
        fleet_norm = {k: (1.0 / len(valid_cols) if k in valid_cols else 0.0) for k in STRESSOR_COLUMNS}

    blended = {k: alpha * local_norm[k] + (1 - alpha) * fleet_norm[k] for k in STRESSOR_COLUMNS}
    sum_blended = sum(blended.values())
    if sum_blended > 0:
        final_scores = {k: round(float(v / sum_blended), 4) for k, v in blended.items()}
    else:
        final_scores = {k: 0.0 for k in STRESSOR_COLUMNS}

    return final_scores


def build_per_cell_drivers(features_df: pd.DataFrame) -> pd.DataFrame:
    fleet_dict = load_fleet_importances()
    rows = []
    for cell_id, cell_df in features_df.groupby("cell_id"):
        scores = compute_cell_driver_scores(cell_df, fleet_dict)
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

    print("=== Per-Cell Top Drivers (With Soft Correlation Floor & Fleet Blend) ===")
    print(result_df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()