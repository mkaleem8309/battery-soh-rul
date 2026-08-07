import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_battery_data(
    n_cells=9,
    cycles_per_cell=400,
    rated_capacity_ah=5.0,
    seed=42
):
    """
    Generates synthetic per-cycle battery telemetry data for multiple cells with varying usage profiles.
    Decay rate depends on temperature, Depth of Discharge (DoD), and C-rate.
    """
    np.random.seed(seed)
    
    profiles = {
        'mild': {'temp_mean': 25.0, 'dod_target': 0.60, 'crate_target': 0.6, 'decay_mult': 0.0004},
        'moderate': {'temp_mean': 35.0, 'dod_target': 0.80, 'crate_target': 1.0, 'decay_mult': 0.0007},
        'aggressive': {'temp_mean': 45.0, 'dod_target': 0.95, 'crate_target': 1.8, 'decay_mult': 0.0011}
    }
    
    profile_names = list(profiles.keys())
    records = []
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    
    for i in range(1, n_cells + 1):
        cell_id = f"CELL_{i:02d}"
        profile_type = profile_names[(i - 1) % len(profile_names)]
        prof = profiles[profile_type]
        
        current_soh = 100.0  # start at 100%
        cell_time = start_time + timedelta(hours=i * 2)
        
        for cycle in range(1, cycles_per_cell + 1):
            # Stressors with per-cycle noise
            temp_mean = np.random.normal(prof['temp_mean'], 2.0)
            temp_max = temp_mean + np.random.uniform(3.0, 7.0)
            
            dod = np.clip(np.random.normal(prof['dod_target'], 0.05), 0.3, 0.99)
            soc_max = np.random.uniform(96.0, 100.0)
            soc_min = np.clip(soc_max - (dod * 100.0), 0.0, 95.0)
            
            crate = np.clip(np.random.normal(prof['crate_target'], 0.1), 0.2, 2.5)
            current_mean = crate * rated_capacity_ah
            current_max = current_mean * np.random.uniform(1.2, 1.5)
            
            # Voltages (IR increases as SoH drops)
            ir_factor = 1.0 + (100.0 - current_soh) * 0.015
            voltage_min = np.random.uniform(2.9, 3.1)
            voltage_max = np.random.uniform(4.15, 4.25) + (ir_factor * 0.02)
            voltage_mean = (voltage_min + voltage_max) / 2.0 + np.random.normal(0, 0.02)
            
            # Durations (minutes)
            discharge_time_min = (dod * rated_capacity_ah / current_mean) * 60.0 + np.random.normal(0, 2.0)
            charge_time_min = 60.0 / crate + np.random.normal(0, 3.0)
            
            # Ground truth SoH calculation
            # Decay rate increases non-linearly with Temp > 25°C, DoD > 80%, C-rate > 1.0C
            temp_stress = max(1.0, (temp_mean / 25.0) ** 1.5)
            dod_stress = max(1.0, (dod / 0.8) ** 1.3)
            crate_stress = max(1.0, crate ** 1.2)
            
            cycle_decay = prof['decay_mult'] * temp_stress * dod_stress * crate_stress * 10.0
            cycle_decay += np.random.normal(0, 0.01)  # realistic noise
            cycle_decay = max(0.001, cycle_decay)
            
            current_soh -= cycle_decay
            current_soh = max(60.0, current_soh)  # bound lower limit
            
            # Advance timestamp by cycle duration + idle time
            cycle_duration_hours = (charge_time_min + discharge_time_min) / 60.0 + np.random.uniform(0.5, 2.0)
            cell_time += timedelta(hours=cycle_duration_hours)
            
            records.append({
                'cell_id': cell_id,
                'usage_profile': profile_type,
                'cycle_id': cycle,
                'timestamp': cell_time.strftime('%Y-%m-%d %H:%M:%S'),
                'voltage_mean': round(float(voltage_mean), 3),
                'voltage_min': round(float(voltage_min), 3),
                'voltage_max': round(float(voltage_max), 3),
                'current_mean': round(float(current_mean), 3),
                'current_max': round(float(current_max), 3),
                'temperature_mean': round(float(temp_mean), 2),
                'temperature_max': round(float(temp_max), 2),
                'soc_min': round(float(soc_min), 2),
                'soc_max': round(float(soc_max), 2),
                'charge_time_min': round(float(charge_time_min), 2),
                'discharge_time_min': round(float(discharge_time_min), 2),
                'soh_ground_truth': round(float(current_soh), 3)
            })
            
    df = pd.DataFrame(records)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'synthetic_battery_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Dataset generated with {len(df)} rows across {n_cells} cells. Saved to {file_path}")
    return df

if __name__ == '__main__':
    generate_synthetic_battery_data()
