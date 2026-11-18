import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)

# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")

MODEL_PATH = Path(
    "ml-engine/models/xgboost_unseen_attack.joblib"
)

THRESHOLD = 0.07

UNSEEN_ATTACKS = [
    "DDOS-ICMP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN"
]

print("=" * 75)
print("IOT INTRUSION DETECTION")
print("UNSEEN ATTACK EVALUATION")
print("=" * 75)

# ==================================================
# 1. LOAD MODEL AND TEST DATA
# ==================================================

print("\nLoading trained model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["feature_columns"]

print("\nLoading untouched test dataset...")

test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Test records: {len(test_df):,}")

X_test = test_df[features]
y_test = test_df["Target"]

# ==================================================
# 2. GENERATE PREDICTIONS
# ==================================================

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

# ==================================================
# 3. OVERALL TEST EVALUATION
# ==================================================

print("\n" + "=" * 75)
print("OVERALL TEST RESULTS")
print("=" * 75)

print(f"Threshold: {THRESHOLD}")

print(
    f"Accuracy: {accuracy_score(y_test, predictions):.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, predictions):.4f}"
)

print(
    f"Macro F1: "
    f"{f1_score(y_test, predictions, average='macro'):.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, probabilities):.4f}"
)

print(
    f"PR-AUC: "
    f"{average_precision_score(y_test, probabilities):.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "MALICIOUS"],
        digits=4
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))

# ==================================================
# 4. EVALUATE SEEN ATTACKS
# ==================================================

print("\n" + "=" * 75)
print("SEEN ATTACK EVALUATION")
print("=" * 75)

seen_mask = ~test_df["Original_Label"].isin(
    UNSEEN_ATTACKS
)

X_seen = X_test.loc[seen_mask]
y_seen = y_test.loc[seen_mask]

seen_probabilities = model.predict_proba(X_seen)[:, 1]

seen_predictions = (
    seen_probabilities >= THRESHOLD
).astype(int)

print(f"Seen test records: {len(X_seen):,}")

print(
    f"Seen Macro F1: "
    f"{f1_score(y_seen, seen_predictions, average='macro'):.4f}"
)

print(
    f"Seen Accuracy: "
    f"{accuracy_score(y_seen, seen_predictions):.4f}"
)

print(
    "\nSeen classification report:\n"
)

print(
    classification_report(
        y_seen,
        seen_predictions,
        target_names=["BENIGN", "MALICIOUS"],
        digits=4
    )
)

# ==================================================
# 5. EVALUATE UNSEEN ATTACKS
# ==================================================

print("\n" + "=" * 75)
print("UNSEEN ATTACK EVALUATION")
print("=" * 75)

unseen_mask = test_df["Original_Label"].isin(
    UNSEEN_ATTACKS
)

X_unseen = X_test.loc[unseen_mask]
y_unseen = y_test.loc[unseen_mask]

unseen_probabilities = model.predict_proba(X_unseen)[:, 1]

unseen_predictions = (
    unseen_probabilities >= THRESHOLD
).astype(int)

print(f"Unseen attack records: {len(X_unseen):,}")

print(
    f"Unseen attack detection rate: "
    f"{np.mean(unseen_predictions == 1):.4f}"
)

print(
    f"Unseen attack recall: "
    f"{np.mean(unseen_predictions == 1):.4f}"
)

# ==================================================
# 6. INDIVIDUAL ATTACK CATEGORY EVALUATION
# ==================================================

print("\n" + "=" * 75)
print("INDIVIDUAL UNSEEN ATTACK RESULTS")
print("=" * 75)

category_results = []

for attack in UNSEEN_ATTACKS:

    mask = (
        test_df["Original_Label"] == attack
    )

    attack_probabilities = probabilities[mask]
    attack_predictions = predictions[mask]

    total = len(attack_predictions)

    detected = int(
        np.sum(attack_predictions == 1)
    )

    missed = total - detected

    detection_rate = (
        detected / total if total else 0
    )

    category_results.append({
        "Attack": attack,
        "Total": total,
        "Detected": detected,
        "Missed": missed,
        "Detection_Rate": detection_rate
    })

    print(f"\nAttack: {attack}")
    print(f"Total records: {total:,}")
    print(f"Detected: {detected:,}")
    print(f"Missed: {missed:,}")
    print(f"Detection rate: {detection_rate:.4%}")

# ==================================================
# 7. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(category_results)

output_path = Path(
    "ml-engine/experiments/unseen_attack_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nIndividual attack results saved:")
print(output_path)

# ==================================================
# 8. COMPLETE
# ==================================================

print("\n" + "=" * 75)
print("UNSEEN ATTACK EVALUATION COMPLETE")
print("=" * 75)