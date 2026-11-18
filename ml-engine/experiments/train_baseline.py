import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
print("BASELINE MODEL: LOGISTIC REGRESSION")
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
# 2. BUILD MODEL
# --------------------------------------------------

print("\nBuilding Logistic Regression model...")

model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# --------------------------------------------------
# 3. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 4. VALIDATE MODEL
# --------------------------------------------------

print("\nEvaluating on validation dataset...")

predictions = model.predict(X_validation)

probabilities = model.predict_proba(X_validation)[:, 1]

# --------------------------------------------------
# 5. EVALUATION METRICS
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
# 6. SAVE MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "logistic_regression_baseline.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

print("\nBASELINE MODEL TRAINING COMPLETE!")