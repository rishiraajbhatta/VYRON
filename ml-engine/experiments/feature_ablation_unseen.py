import pandas as pd
import numpy as np
import xgboost as xgb

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")

MODEL_DIR = Path("ml-engine/models")

UNSEEN_ATTACKS = [
    "DDOS-ICMP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN"
]

ABLATIONS = {
    "Original": [],
    "Without_ACK_Flag": [
        "ack_flag_number"
    ],
    "Without_ACK_Features": [
        "ack_flag_number",
        "ack_count"
    ]
}

print("=" * 75)
print("UNSEEN ATTACK FEATURE ABLATION")
print("=" * 75)

# ==================================================
# 1. LOAD DATA
# ==================================================

print("\nLoading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

excluded = ["Target", "Original_Label", "Number"]

all_features = [
    col for col in train_df.columns
    if col not in excluded
]

y_train = train_df["Target"]
y_validation = validation_df["Target"]
y_test = test_df["Target"]

# ==================================================
# 2. CLASS WEIGHT
# ==================================================

benign_count = int((y_train == 0).sum())
malicious_count = int((y_train == 1).sum())

scale_pos_weight = benign_count / malicious_count

print(f"Training records: {len(train_df):,}")
print(f"Validation records: {len(validation_df):,}")
print(f"Test records: {len(test_df):,}")

# ==================================================
# 3. TRAIN AND EVALUATE
# ==================================================

all_results = []

for experiment_name, removed_features in ABLATIONS.items():

    print("\n" + "=" * 75)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 75)

    features = [
        col for col in all_features
        if col not in removed_features
    ]

    print(f"Features used: {len(features)}")
    print(f"Removed features: {removed_features}")

    X_train = train_df[features]
    X_validation = validation_df[features]
    X_test = test_df[features]

    model = xgb.XGBClassifier(
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

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False
    )

    # Use the same threshold for every model.
    # This is a controlled feature-ablation comparison.
    threshold = 0.07

    # Overall test predictions
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    print("\nOVERALL TEST RESULTS")

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(y_test, test_predictions):.4f}"
    )

    print(
        f"Macro F1: "
        f"{f1_score(y_test, test_predictions, average='macro'):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # Individual unseen attacks
    print("\nUNSEEN ATTACK DETECTION")

    result = {
        "Experiment": experiment_name,
        "Feature_Count": len(features),
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_test, test_predictions
        ),
        "Macro_F1": f1_score(
            y_test, test_predictions, average="macro"
        )
    }

    for attack in UNSEEN_ATTACKS:

        mask = (
            test_df["Original_Label"] == attack
        )

        attack_predictions = test_predictions[mask]

        detection_rate = np.mean(
            attack_predictions == 1
        )

        result[attack] = detection_rate

        print(
            f"{attack}: {detection_rate:.4%}"
        )

    all_results.append(result)

    # Save model
    model_path = MODEL_DIR / (
        f"unseen_ablation_{experiment_name.lower()}.joblib"
    )

    import joblib

    joblib.dump(
        {
            "model": model,
            "feature_columns": features,
            "threshold": threshold,
            "removed_features": removed_features
        },
        model_path
    )

    print(f"\nModel saved: {model_path}")

# ==================================================
# 4. COMPARISON
# ==================================================

results_df = pd.DataFrame(all_results)

output_path = Path(
    "ml-engine/experiments/unseen_feature_ablation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 75)
print("FEATURE ABLATION COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))

print("\nResults saved:", output_path)

print("\nFEATURE ABLATION COMPLETE!")