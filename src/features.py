import os
import numpy as np
import pandas as pd

RATED_CAPACITY_AH = 5.0

def engineer_battery_features(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """
    Computes per-cycle engineered features from raw telemetry data.
    Ensures safe math without NaNs or Infs leaking into the feature set.
    """
    if input_path is None:
        input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_battery_data.csv')
        
    df = pd.read_csv(input_path)
    
    # Sort to ensure running cumulative counts are computed strictly per cell
    df = df.sort_values(by=['cell_id', 'cycle_id']).reset_index(drop=True)
    
    # 1. Charge duration (minutes)
    df['charge_time'] = df['charge_time_min']
    
    # 2. Depth of Discharge (DoD %)
    df['discharge_depth'] = (df['soc_max'] - df['soc_min']).clip(lower=0.0, upper=100.0)
    
    # 3. C-Rate proxy
    df['c_rate'] = df['current_mean'] / RATED_CAPACITY_AH
    
    # 4. Internal Resistance (IR) proxy = Delta V / Delta I (Safe Division)
    delta_v = df['voltage_max'] - df['voltage_min']
    delta_i = df['current_max'] - df['current_mean']
    safe_delta_i = np.where(delta_i <= 1e-4, 1e-4, delta_i)
    df['internal_resistance_proxy'] = delta_v / safe_delta_i
    
    # 5. Cumulative thermal stress (cycles with peak temp > 40°C)
    df['is_high_temp'] = (df['temperature_max'] > 40.0).astype(int)
    df['cumulative_time_above_40C'] = df.groupby('cell_id')['is_high_temp'].cumsum()
    df.drop(columns=['is_high_temp'], inplace=True)
    
    # 6. Cumulative DoD stress (cycles with DoD > 80%)
    df['is_deep_dod'] = (df['discharge_depth'] > 80.0).astype(int)
    df['cumulative_deep_dod_cycles'] = df.groupby('cell_id')['is_deep_dod'].cumsum()
    df.drop(columns=['is_deep_dod'], inplace=True)
    
    # 7. Cumulative cycle count index per cell
    df['cumulative_cycle_count'] = df.groupby('cell_id').cumcount() + 1
    
    feature_cols = [
        'charge_time', 'discharge_depth', 'c_rate',
        'internal_resistance_proxy', 'cumulative_time_above_40C',
        'cumulative_deep_dod_cycles', 'cumulative_cycle_count',
        'temperature_mean', 'temperature_max'
    ]
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        output_path = os.path.join(output_dir, 'battery_features.csv')
        
    df.to_csv(output_path, index=False)
    print(f"Feature engineering completed. Processed {len(df)} cycles. Saved to {output_path}")
    return df

if __name__ == '__main__':
    engineer_battery_features()
