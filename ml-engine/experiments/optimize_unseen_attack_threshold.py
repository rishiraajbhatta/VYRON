import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)

DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")

MODEL_PATH = Path(
    "ml-engine/models/xgboost_unseen_attack.joblib"
)

print("=" * 70)
print("UNSEEN ATTACK EXPERIMENT")
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

# Load model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["feature_columns"]

# Load validation data
validation_df = pd.read_csv(
    DATA_DIR / "validation.csv"
)

X = validation_df[features]
y = validation_df["Target"]

# Get probabilities
probabilities = model.predict_proba(X)[:, 1]

results = []

print("\nTesting classification thresholds...\n")

for threshold in np.arange(0.01, 0.51, 0.01):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions
    ).ravel()

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y, predictions),
        "Macro_F1": macro_f1,
        "Benign_Recall": tn / (tn + fp),
        "Malicious_Recall": tp / (tp + fn),
        "False_Positives": fp,
        "False_Negatives": fn
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "ml-engine/experiments/unseen_attack_threshold_results.csv",
    index=False
)

best = results_df.loc[
    results_df["Macro_F1"].idxmax()
]

print("BEST THRESHOLD")
print("-" * 70)

print(best.to_string())

print("\nTOP 10 THRESHOLDS")

print(
    results_df.sort_values(
        "Macro_F1",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nResults saved successfully.")

print("\nTHRESHOLD OPTIMIZATION COMPLETE!")