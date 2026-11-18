import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("ml-engine/datasets/processed")
MODEL_DIR = Path("ml-engine/models")

print("=" * 65)
print("IOT INTRUSION DETECTION")
print("INDEPENDENT TEST DATASET EVALUATION")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD TEST DATASET
# --------------------------------------------------

print("\nLoading untouched test dataset...")

test_df = pd.read_csv(DATA_DIR / "test.csv")

X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]

print(f"Test records: {len(X_test):,}")
print(f"Features: {X_test.shape[1]}")

# --------------------------------------------------
# 2. LOAD TRAINED MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": "logistic_regression_baseline.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

results = []

# --------------------------------------------------
# 3. EVALUATE EACH MODEL
# --------------------------------------------------

for model_name, filename in models.items():

    print("\n" + "=" * 65)
    print(f"EVALUATING: {model_name}")
    print("=" * 65)

    model_path = MODEL_DIR / filename

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, probabilities)

    pr_auc = average_precision_score(y_test, probabilities)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "MALICIOUS"],
        digits=4
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Macro F1": macro_f1
    })

# --------------------------------------------------
# 4. FINAL MODEL COMPARISON
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST PERFORMANCE COMPARISON")
print("=" * 65)

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    "ml-engine/experiments/test_results.csv",
    index=False
)

print("\nResults saved to:")
print("ml-engine/experiments/test_results.csv")

print("\nINDEPENDENT TESTING COMPLETE!")