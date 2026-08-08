import pandas as pd
import numpy as np
import os

RATED_CAPACITY_AH = 2.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_battery_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "features.csv")


def make_mock_data():
    data = {
        "cycle_id": list(range(1, 11)),
        "cell_id": ["cell_01"] * 5 + ["cell_02"] * 5,
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
        "voltage_mean": [3.7, 3.68, 3.65, 3.6, 3.55, 3.72, 3.7, 3.66, 3.6, 3.5],
        "voltage_min": [3.2, 3.15, 3.1, 3.0, 2.9, 3.25, 3.2, 3.1, 3.0, 2.85],
        "voltage_max": [4.2, 4.18, 4.15, 4.1, 4.0, 4.2, 4.19, 4.15, 4.1, 4.0],
        "current_mean": [1.0, 1.1, 1.2, 1.4, 1.6, 0.8, 0.9, 1.0, 1.2, 1.5],
        "current_max": [2.0, 2.1, 2.3, 2.6, 3.0, 1.6, 1.7, 1.9, 2.2, 2.8],
        "temperature_mean": [30, 32, 35, 38, 42, 28, 29, 31, 34, 40],
        "temperature_max": [38, 41, 43, 46, 50, 33, 35, 38, 41, 47],
        "soc_min": [20, 18, 15, 12, 10, 25, 22, 18, 15, 10],
        "soc_max": [95, 94, 92, 90, 88, 96, 95, 93, 90, 87],
        "charge_time_min": [45, 46, 48, 50, 55, 40, 42, 44, 47, 52],
        "discharge_time_min": [60, 58, 55, 50, 45, 65, 62, 58, 53, 47],
        "soh_ground_truth": [99, 97, 95, 91, 87, 100, 98, 96, 93, 89],
    }
    return pd.DataFrame(data)


def load_input_data():
    if os.path.exists(INPUT_PATH):
        print(f"Found real data at {INPUT_PATH} — using it.")
        return pd.read_csv(INPUT_PATH)
    else:
        print(f"No file at {INPUT_PATH} yet — using mock data instead.")
        return make_mock_data()


def build_features(df):
    df = df.sort_values(["cell_id", "cycle_id"]).reset_index(drop=True)
    discharge_depth = df["soc_max"] - df["soc_min"]
    c_rate = df["current_mean"] / RATED_CAPACITY_AH
    current_swing = (df["current_max"] - df["current_mean"]).replace(0, np.nan)
    internal_resistance_proxy = ((df["voltage_max"] - df["voltage_min"]) / current_swing).fillna(0)
    over_40 = (df["temperature_max"] > 40).astype(int)
    cumulative_time_above_40C = over_40.groupby(df["cell_id"]).cumsum()
    cumulative_cycle_count = df.groupby("cell_id").cumcount() + 1

    return pd.DataFrame({
        "cycle_id": df["cycle_id"],
        "cell_id": df["cell_id"],
        "charge_time": df["charge_time_min"],
        "discharge_depth": discharge_depth,
        "c_rate": c_rate,
        "internal_resistance_proxy": internal_resistance_proxy,
        "cumulative_time_above_40C": cumulative_time_above_40C,
        "cumulative_cycle_count": cumulative_cycle_count,
        "soh_ground_truth": df["soh_ground_truth"],
    })


def main():
    raw_df = load_input_data()
    features_df = build_features(raw_df)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(features_df.head())
    print(features_df.describe())
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()