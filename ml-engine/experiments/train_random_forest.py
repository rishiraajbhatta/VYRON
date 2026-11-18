import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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
print("MODEL 2: RANDOM FOREST")
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

print("\nBuilding Random Forest model...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# 3. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 4. VALIDATE MODEL
# --------------------------------------------------

print("\nEvaluating on validation dataset...")

predictions = model.predict(X_validation)

probabilities = model.predict_proba(X_validation)[:, 1]

# --------------------------------------------------
# 5. EVALUATION
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
# 6. FEATURE IMPORTANCE
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
# 7. SAVE MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "random_forest_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

print("\nRANDOM FOREST TRAINING COMPLETE!")