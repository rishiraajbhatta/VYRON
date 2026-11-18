import time
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")
MODEL_DIR = Path("ml-engine/models")
OUTPUT_DIR = Path("ml-engine/experiments")

UNSEEN_ATTACKS = [
    "DDOS-ICMP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN"
]

RANDOM_STATE = 42

print("=" * 75)
print("MACHINE LEARNING MODEL COMPARISON")
print("UNSEEN ATTACK GENERALIZATION")
print("=" * 75)

# ==================================================
# 1. LOAD DATASETS
# ==================================================

print("\nLoading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

excluded_columns = [
    "Target",
    "Original_Label",
    "Number"
]

features = [
    column for column in train_df.columns
    if column not in excluded_columns
]

X_train = train_df[features]
y_train = train_df["Target"]

X_validation = validation_df[features]
y_validation = validation_df["Target"]

X_test = test_df[features]
y_test = test_df["Target"]

print(f"Training records: {len(X_train):,}")
print(f"Validation records: {len(X_validation):,}")
print(f"Testing records: {len(X_test):,}")
print(f"Features: {len(features)}")

# ==================================================
# 2. CLASS WEIGHT
# ==================================================

benign_count = int((y_train == 0).sum())
malicious_count = int((y_train == 1).sum())

scale_pos_weight = benign_count / malicious_count

print(f"\nBenign training records: {benign_count:,}")
print(f"Malicious training records: {malicious_count:,}")

# ==================================================
# 3. DEFINE MODELS
# ==================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4
    )
}

# ==================================================
# 4. TRAIN AND EVALUATE MODELS
# ==================================================

all_results = []

for model_name, model in models.items():

    print("\n" + "=" * 75)
    print(f"TRAINING: {model_name}")
    print("=" * 75)

    # ------------------------------
    # Training
    # ------------------------------

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.perf_counter() - start_time

    print(f"\nTraining time: {training_time:.2f} seconds")

    # ------------------------------
    # Validation predictions
    # ------------------------------

    print("\nSelecting threshold using validation data...")

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = 0.50
    best_macro_f1 = -1

    for threshold in np.arange(0.01, 1.00, 0.01):

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_validation,
            validation_predictions,
            average="macro"
        )

        if score > best_macro_f1:

            best_macro_f1 = score
            best_threshold = round(float(threshold), 2)

    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation Macro F1: {best_macro_f1:.4f}")

    # ------------------------------
    # Test prediction
    # ------------------------------

    print("\nEvaluating on untouched test data...")

    prediction_start = time.perf_counter()

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = time.perf_counter() - prediction_start

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    # ------------------------------
    # Overall metrics
    # ------------------------------

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        test_predictions
    )

    macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro"
    )

    print("\nOVERALL TEST RESULTS")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        test_predictions,
        target_names=["BENIGN", "MALICIOUS"],
        digits=4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # ------------------------------
    # Individual unseen attacks
    # ------------------------------

    print("\nUNSEEN ATTACK DETECTION")

    result = {
        "Model": model_name,
        "Threshold": best_threshold,
        "Training_Time_Seconds": round(training_time, 2),
        "Prediction_Time_Seconds": round(prediction_time, 2),
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Macro_F1": macro_f1
    }

    for attack in UNSEEN_ATTACKS:

        mask = (
            test_df["Original_Label"] == attack
        )

        attack_predictions = test_predictions[mask]

        detection_rate = float(
            np.mean(attack_predictions == 1)
        )

        result[attack] = detection_rate

        print(
            f"{attack}: {detection_rate:.4%}"
        )

    all_results.append(result)

    # ------------------------------
    # Save model
    # ------------------------------

    filename = (
        model_name.lower().replace(" ", "_")
        + "_unseen_comparison.joblib"
    )

    model_path = MODEL_DIR / filename

    joblib.dump({
        "model": model,
        "feature_columns": features,
        "threshold": best_threshold,
        "model_name": model_name
    }, model_path)

    print(f"\nModel saved: {model_path}")

# ==================================================
# 5. FINAL COMPARISON
# ==================================================

results_df = pd.DataFrame(all_results)

results_path = (
    OUTPUT_DIR / "unseen_model_comparison_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(index=False)
)

print(f"\nResults saved: {results_path}")

print("\n" + "=" * 75)
print("MODEL COMPARISON COMPLETE")
print("=" * 75)