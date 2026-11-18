import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

DATA_DIR = Path("ml-engine/datasets/processed")
MODEL_DIR = Path("ml-engine/models")
OUTPUT_DIR = Path("ml-engine/experiments")

print("=" * 65)
print("IOT INTRUSION DETECTION")
print("TEST EVALUATION WITHOUT NUMBER")
print("=" * 65)

# Load test data
test_df = pd.read_csv(DATA_DIR / "test.csv")

y_test = test_df["Target"]

# Original models use all 39 features
X_test_original = test_df.drop(columns=["Target"])

# Ablation models use 38 features
X_test_reduced = test_df.drop(
    columns=["Target", "Number"]
)

models = {
    "Random Forest (Original)": (
        "random_forest_model.joblib",
        X_test_original
    ),
    "Random Forest (Without Number)": (
        "random_forest_without_number.joblib",
        X_test_reduced
    ),
    "XGBoost (Original)": (
        "xgboost_model.joblib",
        X_test_original
    ),
    "XGBoost (Without Number)": (
        "xgboost_without_number.joblib",
        X_test_reduced
    )
}

results = []

for model_name, (filename, X_test) in models.items():

    print("\n" + "=" * 65)
    print(f"EVALUATING: {model_name}")
    print("=" * 65)

    model = joblib.load(MODEL_DIR / filename)

    probabilities = model.predict_proba(X_test)[:, 1]

    for threshold in [0.50, 0.05]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        result = {
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "Macro F1": f1_score(
                y_test, predictions, average="macro"
            ),
            "Benign Recall": recall_score(
                y_test, predictions, pos_label=0
            ),
            "Malicious Recall": recall_score(
                y_test, predictions, pos_label=1
            ),
            "False Positives": fp,
            "False Negatives": fn
        }

        results.append(result)

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy: {result['Accuracy']:.4f}")
        print(f"Balanced Accuracy: {result['Balanced Accuracy']:.4f}")
        print(f"Macro F1: {result['Macro F1']:.4f}")
        print(f"Benign Recall: {result['Benign Recall']:.4f}")
        print(f"Malicious Recall: {result['Malicious Recall']:.4f}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

# Save results
results_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "without_number_test_results.csv"

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("FINAL FEATURE ABLATION COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")
print("\nEVALUATION COMPLETE!")