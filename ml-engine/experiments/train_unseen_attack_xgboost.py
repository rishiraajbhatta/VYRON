import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score
)

# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")

MODEL_DIR = Path("ml-engine/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_unseen_attack.joblib"

RANDOM_STATE = 42

print("=" * 70)
print("IOT INTRUSION DETECTION")
print("XGBOOST - UNSEEN ATTACK EXPERIMENT")
print("=" * 70)

# ==================================================
# 1. LOAD DATASETS
# ==================================================

print("\nLoading processed datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")

print(f"Training records: {len(train_df):,}")
print(f"Validation records: {len(validation_df):,}")

# ==================================================
# 2. PREPARE FEATURES AND TARGETS
# ==================================================

print("\nPreparing features...")

excluded_columns = [
    "Target",
    "Original_Label",
    "Number"
]

feature_columns = [
    column for column in train_df.columns
    if column not in excluded_columns
]

X_train = train_df[feature_columns]
y_train = train_df["Target"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["Target"]

print(f"Features used: {len(feature_columns)}")

print("Number feature excluded: YES")

# ==================================================
# 3. CALCULATE CLASS WEIGHT
# ==================================================

benign_count = (y_train == 0).sum()
malicious_count = (y_train == 1).sum()

scale_pos_weight = (
    benign_count / malicious_count
)

print("\nClass distribution:")
print(f"Benign: {benign_count:,}")
print(f"Malicious: {malicious_count:,}")
print(f"Scale positive weight: {scale_pos_weight:.6f}")

# ==================================================
# 4. INITIALIZE XGBOOST
# ==================================================

print("\nInitializing XGBoost...")

model = XGBClassifier(
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

# ==================================================
# 5. TRAIN MODEL
# ==================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")

# ==================================================
# 6. VALIDATION EVALUATION
# ==================================================

print("\nEvaluating on validation dataset...")

probabilities = model.predict_proba(
    X_validation
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)

accuracy = accuracy_score(
    y_validation,
    predictions
)

macro_f1 = f1_score(
    y_validation,
    predictions,
    average="macro"
)

roc_auc = roc_auc_score(
    y_validation,
    probabilities
)

pr_auc = average_precision_score(
    y_validation,
    probabilities
)

print("\nVALIDATION RESULTS")
print("-" * 70)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        predictions,
        target_names=["BENIGN", "MALICIOUS"],
        digits=4
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        predictions
    )
)

# ==================================================
# 7. SAVE MODEL
# ==================================================

print("\nSaving trained model...")

model_artifact = {
    "model": model,
    "feature_columns": feature_columns,
    "excluded_features": ["Number"],
    "threshold": 0.50
}

joblib.dump(
    model_artifact,
    MODEL_PATH
)

print(f"Model saved: {MODEL_PATH}")

# ==================================================
# 8. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("XGBOOST TRAINING COMPLETE")
print("=" * 70)

print(f"Training records: {len(train_df):,}")
print(f"Validation records: {len(validation_df):,}")
print(f"Features: {len(feature_columns)}")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation macro F1: {macro_f1:.4f}")

print("\nModel ready for threshold optimization!")