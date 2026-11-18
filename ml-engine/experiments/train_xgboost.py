import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("ml-engine/datasets/processed")
MODEL_DIR = Path("ml-engine/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("IOT INTRUSION DETECTION")
print("MODEL 3: XGBOOST")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATASETS
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"]

X_validation = validation_df.drop(columns=["Target"])
y_validation = validation_df["Target"]

print(f"Training records: {len(X_train):,}")
print(f"Validation records: {len(X_validation):,}")
print(f"Features: {X_train.shape[1]}")

# --------------------------------------------------
# 2. CALCULATE CLASS WEIGHT
# --------------------------------------------------

benign_count = (y_train == 0).sum()
malicious_count = (y_train == 1).sum()

scale_weight = malicious_count / benign_count

print(f"\nBenign records: {benign_count:,}")
print(f"Malicious records: {malicious_count:,}")
print(f"Class weight: {scale_weight:.2f}")

# --------------------------------------------------
# 3. BUILD MODEL
# --------------------------------------------------

print("\nBuilding XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1 / scale_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

# --------------------------------------------------
# 4. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 5. VALIDATE MODEL
# --------------------------------------------------

print("\nEvaluating on validation dataset...")

predictions = model.predict(X_validation)

probabilities = model.predict_proba(X_validation)[:, 1]

# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy_score(y_validation, predictions):.4f}")

print(f"ROC-AUC: {roc_auc_score(y_validation, probabilities):.4f}")

print(
    f"PR-AUC (Average Precision): "
    f"{average_precision_score(y_validation, probabilities):.4f}"
)

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
print(confusion_matrix(y_validation, predictions))

# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

print("\nTOP 10 IMPORTANT FEATURES")

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10).to_string(index=False))

# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "xgboost_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

print("\nXGBOOST TRAINING COMPLETE!")