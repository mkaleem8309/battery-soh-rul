import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
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

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

prediction_df = pd.DataFrame({
    "cycle_id": df.iloc[test_idx]["cycle_id"].values,
    "cell_id": df.iloc[test_idx]["cell_id"].values,
    "soh_predicted": predictions,
    "soh_ground_truth": y_test.values
})

prediction_df.to_csv("outputs/soh_predictions.csv", index=False)

print("\nPredictions saved successfully!")

print("MAE:", mae)
print("R²:", r2)

plt.figure(figsize=(10,5))

plt.plot(
    prediction_df["cycle_id"],
    prediction_df["soh_ground_truth"],
    label="Ground Truth",
    linewidth=2
)

plt.plot(
    prediction_df["cycle_id"],
    prediction_df["soh_predicted"],
    label="Predicted",
    linewidth=2
)

plt.xlabel("Cycle ID")
plt.ylabel("State of Health (%)")
plt.title("Predicted SoH vs Ground Truth")

plt.legend()

plt.grid(True)

plt.show()


feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
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