import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_DIR = Path("ml-engine/datasets/processed")
MODEL_DIR = Path("ml-engine/models")

print("=" * 65)
print("IOT INTRUSION DETECTION")
print("FEATURE ABLATION: REMOVING NUMBER")
print("=" * 65)

# Load training data
train_df = pd.read_csv(DATA_DIR / "train.csv")

X_train = train_df.drop(columns=["Target", "Number"])
y_train = train_df["Target"]

print(f"\nTraining records: {len(X_train):,}")
print(f"Features remaining: {X_train.shape[1]}")
print("Removed feature: Number")

benign_count = (y_train == 0).sum()
malicious_count = (y_train == 1).sum()

scale_pos_weight = benign_count / malicious_count

# --------------------------------------------------
# 1. RANDOM FOREST
# --------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST WITHOUT NUMBER")
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

rf_model.fit(X_train, y_train)

joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest_without_number.joblib"
)

print("Random Forest training complete.")
print("Model saved successfully.")

# --------------------------------------------------
# 2. XGBOOST
# --------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING XGBOOST WITHOUT NUMBER")
print("=" * 65)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

xgb_model.fit(X_train, y_train)

joblib.dump(
    xgb_model,
    MODEL_DIR / "xgboost_without_number.joblib"
)

print("XGBoost training complete.")
print("Model saved successfully.")

print("\n" + "=" * 65)
print("FEATURE ABLATION TRAINING COMPLETE")
print("=" * 65)

print("\nModels saved:")
print("1. random_forest_without_number.joblib")
print("2. xgboost_without_number.joblib")

print("\nFeatures used:", X_train.shape[1])