import os
import numpy as np
import pandas as pd

EOL_SOH_THRESHOLD = 80.0  # 80% State-of-Health is standard End-of-Life (EOL)
MAX_RUL_SANITY_CAP = 3000  # Cap RUL extrapolation to prevent unrealistic explosion (e.g. >3000 cycles)

def predict_cell_rul(df_cell: pd.DataFrame, window_size: int = 50) -> dict:
    """
    Predicts Remaining Useful Life (RUL) in cycles with uncertainty bands (Worst, Likely, Best)
    based on the trailing SoH degradation trend of a single battery cell.
    
    Assumptions:
    - Assumes future cell operating conditions (temp, DoD, C-rate) remain consistent 
      with trailing `window_size` cycles.
    - EOL threshold is defined at 80.0% SoH.
    - RUL upper bound is capped at MAX_RUL_SANITY_CAP (3000 cycles) for physical realism.
    """
    df_cell = df_cell.sort_values(by='cycle_id').reset_index(drop=True)
    
    current_cycle = int(df_cell['cycle_id'].iloc[-1])
    current_soh = float(df_cell['soh_predicted'].iloc[-1] if 'soh_predicted' in df_cell.columns else df_cell['soh_ground_truth'].iloc[-1])
    
    if current_soh <= EOL_SOH_THRESHOLD:
        return {
            'cell_id': str(df_cell['cell_id'].iloc[0]),
            'current_cycle': current_cycle,
            'current_soh': round(current_soh, 2),
            'slope_per_cycle': 0.0,
            'rul_worst': 0,
            'rul_likely': 0,
            'rul_best': 0,
            'status': 'Reached EOL'
        }
        
    recent_df = df_cell.tail(min(len(df_cell), window_size)).copy()
    
    cycles = recent_df['cycle_id'].values
    soh_vals = recent_df['soh_predicted'].values if 'soh_predicted' in recent_df.columns else recent_df['soh_ground_truth'].values
    
    poly = np.polyfit(cycles, soh_vals, 1)
    slope = poly[0]  # negative slope (% SoH per cycle)
    
    deltas = np.diff(soh_vals)
    loss_rates = -deltas
    
    likely_loss_rate = max(0.001, -slope)
    worst_loss_rate = max(likely_loss_rate, np.percentile(loss_rates, 90) if len(loss_rates) > 0 else likely_loss_rate * 1.3)
    best_loss_rate = min(likely_loss_rate, max(0.0005, np.percentile(loss_rates, 10)) if len(loss_rates) > 0 else likely_loss_rate * 0.7)
    
    remaining_soh = current_soh - EOL_SOH_THRESHOLD
    
    rul_likely = min(MAX_RUL_SANITY_CAP, max(0, int(round(remaining_soh / likely_loss_rate))))
    rul_worst = min(rul_likely, max(0, int(round(remaining_soh / worst_loss_rate))))
    rul_best = min(MAX_RUL_SANITY_CAP, max(rul_likely, int(round(remaining_soh / best_loss_rate))))
    
    return {
        'cell_id': str(df_cell['cell_id'].iloc[0]),
        'current_cycle': current_cycle,
        'current_soh': round(current_soh, 2),
        'slope_per_cycle': round(float(slope), 4),
        'rul_worst': int(rul_worst),
        'rul_likely': int(rul_likely),
        'rul_best': int(rul_best)
    }

def run_rul_predictions(features_path: str = None) -> pd.DataFrame:
    if features_path is None:
        features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'battery_features.csv')
        
    df = pd.read_csv(features_path)
    
    pred_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'soh_predictions_test_cell.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        df = df.merge(pred_df[['cell_id', 'cycle_id', 'soh_predicted']], on=['cell_id', 'cycle_id'], how='left')
        df['soh_predicted'] = df['soh_predicted'].fillna(df['soh_ground_truth'])
    else:
        df['soh_predicted'] = df['soh_ground_truth']
        
    results = []
    for cell_id, group in df.groupby('cell_id'):
        res = predict_cell_rul(group, window_size=50)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'rul_predictions.csv')
    res_df.to_csv(output_path, index=False)
    print(f"RUL prediction completed for {len(res_df)} cells. Saved to {output_path}")
    return res_df

if __name__ == '__main__':
    run_rul_predictions()
