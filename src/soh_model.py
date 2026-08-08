import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/features.csv")

X = df[
    [
        "charge_time",
        "discharge_depth",
        "c_rate",
        "internal_resistance_proxy",
        "cumulative_time_above_40C",
        "cumulative_cycle_count",
    ]
]

y = df["soh_ground_truth"]

groups = df["cell_id"]

n_splits = groups.nunique()

splitter = GroupKFold(n_splits=n_splits)

oof_predictions = pd.Series(index=df.index, dtype=float)

feature_importances = []

for train_idx, test_idx in splitter.split(X, y, groups):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]

    model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    oof_predictions.iloc[test_idx] = predictions

    feature_importances.append(model.feature_importances_)

mae = mean_absolute_error(y, oof_predictions)
r2 = r2_score(y, oof_predictions)

print("OOF MAE:", mae)
print("OOF R²:", r2)


prediction_df = pd.DataFrame({
    "cycle_id": df["cycle_id"],
    "cell_id": df["cell_id"],
    "soh_predicted": oof_predictions,
    "soh_ground_truth": df["soh_ground_truth"]
})

prediction_df.to_csv(
    "outputs/soh_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("Rows:", len(prediction_df))
print("Cells:", prediction_df["cell_id"].nunique())
print("MAE:", mae)
print("R²:", r2)

plt.figure(figsize=(10, 5))

plot_cell = prediction_df["cell_id"].iloc[0]

plot_data = prediction_df[prediction_df["cell_id"] == plot_cell]

plt.plot(
    plot_data["cycle_id"],
    plot_data["soh_ground_truth"],
    label="Ground Truth",
    linewidth=2
)

plt.plot(
    plot_data["cycle_id"],
    plot_data["soh_predicted"],
    label="Predicted",
    linewidth=2
)

plt.xlabel("Cycle ID")
plt.ylabel("State of Health (%)")
plt.title(f"Predicted SoH vs Ground Truth - {plot_cell}")

plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/soh_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Prediction plot saved successfully!")

average_importances = sum(feature_importances) / len(feature_importances)

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": average_importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_df.to_csv(
    "outputs/feature_importances.csv",
    index=False
)

print("\nFeature importances saved successfully!")
print(feature_importance_df)