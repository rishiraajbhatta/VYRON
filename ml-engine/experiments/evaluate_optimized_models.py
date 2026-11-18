import pandas as pd
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
print("OPTIMIZED THRESHOLD TEST EVALUATION")
print("=" * 65)

# Load untouched test dataset
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]

print(f"\nTest records: {len(X_test):,}")

models = {
    "Logistic Regression": "logistic_regression_baseline.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

threshold = 0.05
all_results = []

for model_name, filename in models.items():

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    model = joblib.load(MODEL_DIR / filename)

    probabilities = model.predict_proba(X_test)[:, 1]

    for current_threshold in [0.50, threshold]:

        predictions = (
            probabilities >= current_threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro"
        )

        benign_recall = tn / (tn + fp)
        malicious_recall = tp / (tp + fn)

        result = {
            "Model": model_name,
            "Threshold": current_threshold,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "Macro F1": macro_f1,
            "Benign Recall": benign_recall,
            "Malicious Recall": malicious_recall,
            "False Positives": fp,
            "False Negatives": fn
        }

        all_results.append(result)

        print(f"\nThreshold: {current_threshold:.2f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {result['Balanced Accuracy']:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Benign Recall: {benign_recall:.4f}")
        print(f"Malicious Recall: {malicious_recall:.4f}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

# Save results
results_df = pd.DataFrame(all_results)

output_file = OUTPUT_DIR / "optimized_test_results.csv"
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("FINAL OPTIMIZED MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")
print("\nOPTIMIZED THRESHOLD EVALUATION COMPLETE!")