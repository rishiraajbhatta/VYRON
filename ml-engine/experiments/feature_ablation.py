import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("ml-engine/datasets/processed")

print("=" * 65)
print("FEATURE ABLATION EXPERIMENT")
print("INVESTIGATING THE NUMBER FEATURE")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df.drop(columns=["Target", "Number"])
y_train = train_df["Target"]

X_validation = validation_df.drop(columns=["Target", "Number"])
y_validation = validation_df["Target"]

print(f"Training records: {len(X_train):,}")
print(f"Validation records: {len(X_validation):,}")
print(f"Remaining features: {X_train.shape[1]}")

# --------------------------------------------------
# 2. RANDOM FOREST WITHOUT NUMBER
# --------------------------------------------------

print("\n" + "=" * 65)
print("RANDOM FOREST WITHOUT NUMBER")
print("=" * 65)

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_validation)
rf_probabilities = rf_model.predict_proba(X_validation)[:, 1]

print("\nRandom Forest Results:")

print(f"Accuracy: {accuracy_score(y_validation, rf_predictions):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_validation, rf_probabilities):.4f}")
print(f"PR-AUC: {average_precision_score(y_validation, rf_probabilities):.4f}")

print("\nClassification Report:")

print(classification_report(
    y_validation,
    rf_predictions,
    target_names=["BENIGN", "MALICIOUS"],
    digits=4
))

# --------------------------------------------------
# 3. XGBOOST WITHOUT NUMBER
# --------------------------------------------------

print("\n" + "=" * 65)
print("XGBOOST WITHOUT NUMBER")
print("=" * 65)

benign_count = (y_train == 0).sum()
malicious_count = (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=benign_count / malicious_count,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

print("\nTraining XGBoost...")

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_validation)
xgb_probabilities = xgb_model.predict_proba(X_validation)[:, 1]

print("\nXGBoost Results:")

print(f"Accuracy: {accuracy_score(y_validation, xgb_predictions):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_validation, xgb_probabilities):.4f}")
print(f"PR-AUC: {average_precision_score(y_validation, xgb_probabilities):.4f}")

print("\nClassification Report:")

print(classification_report(
    y_validation,
    xgb_predictions,
    target_names=["BENIGN", "MALICIOUS"],
    digits=4
))

# --------------------------------------------------
# 4. FEATURE IMPORTANCE
# --------------------------------------------------

print("\n" + "=" * 65)
print("TOP 10 XGBOOST FEATURES WITHOUT NUMBER")
print("=" * 65)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10).to_string(index=False))

print("\nFEATURE ABLATION EXPERIMENT COMPLETE!")