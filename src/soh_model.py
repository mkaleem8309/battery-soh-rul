import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLUMNS = [
    'cumulative_cycle_count',
    'charge_time',
    'discharge_depth',
    'c_rate',
    'internal_resistance_proxy',
    'cumulative_time_above_40C',
    'temperature_mean',
    'temperature_max'
]

def train_soh_labeled_model(df: pd.DataFrame, test_cell_id: str = 'CELL_09'):
    """
    Train a RandomForestRegressor mapping engineered features to ground-truth SoH.
    Holds out one specific cell (test_cell_id) for validation.
    """
    train_df = df[df['cell_id'] != test_cell_id].copy()
    test_df = df[df['cell_id'] == test_cell_id].copy()
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['soh_ground_truth']
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['soh_ground_truth']
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    test_df['soh_predicted'] = y_pred
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rf, test_df, mae, r2

def estimate_soh_unlabeled_fallback(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Unlabeled fallback approach:
    Estimates SoH by integrating discharge current over time to get cycle capacity,
    normalizes against initial (first-cycle) capacity per cell, and applies a rolling mean.
    """
    df = df.copy()
    # Capacity proxy (Ah) = current_mean (A) * (discharge_time_min / 60.0)
    df['discharge_capacity_ah'] = df['current_mean'] * (df['discharge_time_min'] / 60.0)
    
    # First cycle baseline capacity per cell
    initial_cap = df.groupby('cell_id')['discharge_capacity_ah'].transform('first')
    
    # Normalized capacity percentage
    df['raw_capacity_soh'] = (df['discharge_capacity_ah'] / initial_cap) * 100.0
    
    # Smooth with rolling average per cell
    df['soh_unlabeled_estimated'] = df.groupby('cell_id')['raw_capacity_soh'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    return df

def run_soh_pipeline(features_path: str = None):
    if features_path is None:
        features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'battery_features.csv')
        
    df = pd.read_csv(features_path)
    
    # Execute Labeled Model (Default)
    rf_model, test_results, mae, r2 = train_soh_labeled_model(df, test_cell_id='CELL_09')
    
    print(f"--- SOH LABELED MODEL EVALUATION (Test Cell: CELL_09) ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}% SoH")
    print(f"R^2 Score: {r2:.4f}")
    
    # Save predictions
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    test_results.to_csv(os.path.join(output_dir, 'soh_predictions_test_cell.csv'), index=False)
    
    # Phase 4: Driver Identification
    drivers_df = extract_driver_importances(rf_model, df)
    drivers_df.to_csv(os.path.join(output_dir, 'driver_importances.csv'), index=False)
    
    return rf_model, test_results, mae, r2, drivers_df

def extract_driver_importances(rf_model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts top degradation drivers from feature_importances_
    and cross-checks with Pearson correlation against total SoH loss.
    """
    importances = rf_model.feature_importances_
    
    # Compute total SoH loss per cycle for correlation sanity-check
    df = df.copy()
    df['soh_loss'] = 100.0 - df['soh_ground_truth']
    
    driver_records = []
    for feat, imp in zip(FEATURE_COLUMNS, importances):
        corr = df[feat].corr(df['soh_loss'])
        driver_records.append({
            'feature': feat,
            'importance_score': float(imp),
            'correlation_with_soh_loss': float(corr)
        })
        
    drivers_df = pd.DataFrame(driver_records)
    drivers_df = drivers_df.sort_values(by='importance_score', ascending=False).reset_index(drop=True)
    return drivers_df


if __name__ == '__main__':
    run_soh_pipeline()
