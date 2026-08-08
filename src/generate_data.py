"""
Synthetic battery telemetry generator — Person A.
Outputs data/synthetic_battery_data.csv matching team data contract:

cycle_id, cell_id, timestamp, voltage_mean, voltage_min, voltage_max,
current_mean, current_max, temperature_mean, temperature_max,
soc_min, soc_max, charge_time_min, discharge_time_min, soh_ground_truth
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_CELLS = 18
MIN_CYCLES = 300
MAX_CYCLES = 500

# usage profile -> (decay_rate, temp_bias, dod_bias, c_rate_bias)
PROFILES = {
    "mild":       dict(decay_k=0.00060, temp_bias=0.0, dod_bias=0.0, c_bias=0.0),
    "moderate":   dict(decay_k=0.00110, temp_bias=6.0, dod_bias=0.10, c_bias=0.3),
    "aggressive": dict(decay_k=0.00190, temp_bias=14.0, dod_bias=0.22, c_bias=0.8),
}

RATED_CAPACITY_AH = 2.0


def assign_profile(cell_idx: int) -> str:
    # roughly even spread across cells: cycles through mild, moderate, aggressive
    order = ["mild", "moderate", "aggressive"]
    return order[cell_idx % 3]


def cell_variation_seed(cell_idx: int) -> None:
    """Slightly different noise/decay jitter per cell, even within the
    same profile, so cells aren't identical twins."""
    pass


def soh_curve(cycle_ids: np.ndarray, decay_k: float, floor: float = 0.66) -> np.ndarray:
    """Exponential decay from 100% down toward ~floor, with slight
    per-cell randomness in the decay constant."""
    k = decay_k * rng.uniform(0.9, 1.1)
    raw = 100.0 * np.exp(-k * cycle_ids)
    soh = floor * 100 + (100 - floor * 100) * (raw / 100.0)
    return soh


def generate_cell(cell_idx: int) -> pd.DataFrame:
    cell_id = f"cell_{cell_idx:02d}"
    profile_name = assign_profile(cell_idx)
    p = PROFILES[profile_name]

    n_cycles = rng.integers(MIN_CYCLES, MAX_CYCLES + 1)
    cycle_ids = np.arange(1, n_cycles + 1)

    soh = soh_curve(cycle_ids, p["decay_k"])
    soh += rng.normal(0, 0.4, size=n_cycles)  # noise
    soh = np.clip(soh, 60, 100)
    soh = np.maximum.accumulate(soh[::-1])[::-1]  # keep roughly monotonic-ish decline
    soh = np.clip(soh + rng.normal(0, 0.3, size=n_cycles), 60, 100.5)

    # degradation-aware signal drift: as SoH drops, internal resistance
    # rises (voltage sag grows), capacity/discharge time shrinks
    health_frac = soh / 100.0

    voltage_mean = 3.7 - (1 - health_frac) * 0.15 + rng.normal(0, 0.01, n_cycles)
    voltage_min = voltage_mean - (0.15 + (1 - health_frac) * 0.25) - rng.normal(0, 0.01, n_cycles).clip(min=0)
    voltage_max = voltage_mean + (0.15 + (1 - health_frac) * 0.05) + rng.normal(0, 0.01, n_cycles).clip(min=0)

    base_current = 1.0 + p["c_bias"]
    current_mean = base_current + rng.normal(0, 0.05, n_cycles)
    current_max = current_mean + 0.5 + p["c_bias"] * 0.5 + rng.normal(0, 0.05, n_cycles).clip(min=0)

    temperature_mean = 25 + p["temp_bias"] + rng.normal(0, 1.5, n_cycles)
    temperature_max = temperature_mean + 8 + rng.normal(0, 2.0, n_cycles).clip(min=0)

    soc_min = np.clip(0.15 - p["dod_bias"] * 0.3 + rng.normal(0, 0.02, n_cycles), 0.02, 0.4)
    soc_max = np.clip(0.95 + rng.normal(0, 0.01, n_cycles), 0.85, 1.0)

    charge_time_min = (RATED_CAPACITY_AH * health_frac / current_mean) * 60 * rng.uniform(0.95, 1.05, n_cycles)
    discharge_time_min = (RATED_CAPACITY_AH * health_frac / current_mean) * 60 * rng.uniform(0.9, 1.0, n_cycles)

    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=6 * i) for i in range(n_cycles)]

    df = pd.DataFrame({
        "cycle_id": cycle_ids,
        "cell_id": cell_id,
        "timestamp": timestamps,
        "voltage_mean": voltage_mean.round(4),
        "voltage_min": voltage_min.round(4),
        "voltage_max": voltage_max.round(4),
        "current_mean": current_mean.round(4),
        "current_max": current_max.round(4),
        "temperature_mean": temperature_mean.round(2),
        "temperature_max": temperature_max.round(2),
        "soc_min": soc_min.round(4),
        "soc_max": soc_max.round(4),
        "charge_time_min": charge_time_min.round(2),
        "discharge_time_min": discharge_time_min.round(2),
        "soh_ground_truth": soh.round(3),
    })
    df.attrs["profile"] = profile_name
    return df


def generate_all(n_cells: int = N_CELLS) -> pd.DataFrame:
    frames = [generate_cell(i) for i in range(n_cells)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    full_df = generate_all()

    full_path = "data/synthetic_battery_data.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Full dataset: {full_df.shape[0]} rows -> {full_path}")

    # small sample for teammates to start against immediately
    sample = full_df.groupby("cell_id").head(3).head(27)
    sample_path = "data/sample_battery_data.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Sample: {sample.shape[0]} rows -> {sample_path}")

    print("\nCells and profiles:")
    for i in range(N_CELLS):
        print(f"  cell_{i:02d}: {assign_profile(i)}")

    print("\nHead of full dataset:")
    print(full_df.head())
