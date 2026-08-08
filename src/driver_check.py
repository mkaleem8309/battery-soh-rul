import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in ("src", "outputs"):
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

IMPORTANCES_PATH = os.path.join(PROJECT_ROOT, "outputs", "feature_importances.csv")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features.csv")

STRESSORS = {
    "temperature exposure": "cumulative_time_above_40C",
    "discharge depth": "discharge_depth",
    "c_rate": "c_rate",
}


def load_importances():
    df = pd.read_csv(IMPORTANCES_PATH)
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def load_features():
    return pd.read_csv(FEATURES_PATH)


def top_5_drivers(importances_df):
    top5 = importances_df.head(5)
    print("=== Top 5 Drivers (from model feature importances) ===")
    for i, row in top5.iterrows():
        print(f"{i+1}. {row['Feature']}  (importance: {row['Importance']:.4f})")
    return top5


def correlation_sanity_check(features_df):
    print("\n=== Sanity Check: correlation with SoH ===")
    print("(negative correlation = stressor goes up, SoH goes down -- expected)\n")

    results = []
    for label, col in STRESSORS.items():
        if col not in features_df.columns:
            print(f"Skipping '{label}' — column '{col}' not found in features.csv")
            continue
        corr = features_df[col].corr(features_df["soh_ground_truth"])
        results.append((label, col, corr))
        direction = "as expected (higher stress -> lower SoH)" if corr < 0 else "UNEXPECTED (check this)"
        print(f"{label:25s} vs soh_ground_truth: r = {corr:+.3f}  -> {direction}")

    return pd.DataFrame(results, columns=["stressor", "column", "correlation"])


def main():
    importances_df = load_importances()
    features_df = load_features()

    top5 = top_5_drivers(importances_df)
    corr_df = correlation_sanity_check(features_df)

    print("\n=== Summary ===")
    top_feature = importances_df.iloc[0]["Feature"]
    print(f"Model's #1 driver: {top_feature}")
    if top_feature in STRESSORS.values():
        matching = corr_df[corr_df["column"] == top_feature]
        if not matching.empty:
            r = matching.iloc[0]["correlation"]
            verdict = "supported by correlation data" if r < -0.1 else "weak/no correlation support -- worth flagging to the team"
            print(f"Correlation check: r = {r:+.3f} -> {verdict}")


if __name__ == "__main__":
    main()