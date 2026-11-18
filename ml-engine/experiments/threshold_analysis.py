import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

DATA_DIR = Path("ml-engine/datasets/processed")
MODEL_DIR = Path("ml-engine/models")
OUTPUT_DIR = Path("ml-engine/experiments")

print("=" * 65)
print("IOT INTRUSION DETECTION")
print("CLASSIFICATION THRESHOLD ANALYSIS")
print("=" * 65)

# Load validation dataset only
validation_df = pd.read_csv(DATA_DIR / "validation.csv")

X_val = validation_df.drop(columns=["Target"])
y_val = validation_df["Target"]

print(f"\nValidation records: {len(X_val):,}")
print("Positive class: MALICIOUS")
print("Negative class: BENIGN")

models = {
    "Logistic Regression": "logistic_regression_baseline.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

thresholds = np.arange(0.05, 1.00, 0.05)

all_results = []

for model_name, filename in models.items():

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    model = joblib.load(MODEL_DIR / filename)

    probabilities = model.predict_proba(X_val)[:, 1]

    model_results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        result = {
            "Model": model_name,
            "Threshold": round(threshold, 2),
            "Accuracy": accuracy_score(y_val, predictions),
            "Balanced Accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "Macro F1": f1_score(
                y_val, predictions, average="macro"
            ),
            "Malicious Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Malicious Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "Benign Recall": tn / (tn + fp) if (tn + fp) else 0,
            "False Positives": fp,
            "False Negatives": fn
        }

        model_results.append(result)
        all_results.append(result)

    results_df = pd.DataFrame(model_results)

    best_result = results_df.loc[
        results_df["Macro F1"].idxmax()
    ]

    print("\nBest threshold based on validation Macro F1:")
    print(f"Threshold: {best_result['Threshold']:.2f}")
    print(f"Macro F1: {best_result['Macro F1']:.4f}")
    print(f"Accuracy: {best_result['Accuracy']:.4f}")
    print(
        f"Malicious Recall: "
        f"{best_result['Malicious Recall']:.4f}"
    )
    print(
        f"Benign Recall: "
        f"{best_result['Benign Recall']:.4f}"
    )
    print(f"False Positives: {best_result['False Positives']}")
    print(f"False Negatives: {best_result['False Negatives']}")

    print("\nThreshold comparison:")
    print(
        results_df[
            [
                "Threshold",
                "Macro F1",
                "Malicious Recall",
                "Benign Recall",
                "False Positives",
                "False Negatives"
            ]
        ].to_string(index=False)
    )

# Save all threshold results
final_df = pd.DataFrame(all_results)

output_file = OUTPUT_DIR / "threshold_results.csv"
final_df.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 65)
print(f"\nResults saved to: {output_file}")
