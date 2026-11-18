import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# Configuration
DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")

MODEL_PATH = Path(
    "ml-engine/models/xgboost_unseen_attack.joblib"
)

UNSEEN_ATTACKS = [
    "DDOS-ICMP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN"
]

THRESHOLDS = [
    0.07,
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
    0.95,
    0.99,
    0.999
]

print("=" * 75)
print("THRESHOLD ROBUSTNESS ANALYSIS")
print("=" * 75)

# Load model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["feature_columns"]

# Load test data
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_test = test_df[features]

# Generate probabilities
print("\nGenerating prediction probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

test_df["Probability"] = probabilities

results = []

# Evaluate every threshold
for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    benign_mask = test_df["Target"] == 0

    benign_total = int(benign_mask.sum())

    false_positives = int(
        np.sum(predictions[benign_mask] == 1)
    )

    benign_specificity = (
        1 - false_positives / benign_total
    )

    row = {
        "Threshold": threshold,
        "Benign_Specificity": benign_specificity,
        "False_Positives": false_positives
    }

    print("\n" + "-" * 75)
    print(f"THRESHOLD: {threshold}")

    print(
        f"Benign specificity: {benign_specificity:.4%}"
    )

    print(
        f"False positives: {false_positives:,}"
    )

    # Individual unseen attack detection
    for attack in UNSEEN_ATTACKS:

        attack_mask = (
            test_df["Original_Label"] == attack
        )

        total = int(attack_mask.sum())

        detected = int(
            np.sum(predictions[attack_mask] == 1)
        )

        detection_rate = (
            detected / total if total else 0
        )

        row[attack] = detection_rate

        print(
            f"{attack}: "
            f"{detection_rate:.4%} "
            f"({detected:,}/{total:,})"
        )

    results.append(row)

# Save results
results_df = pd.DataFrame(results)

output_path = Path(
    "ml-engine/experiments/threshold_robustness_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 75)
print("THRESHOLD ROBUSTNESS SUMMARY")
print("=" * 75)

print(
    results_df.to_string(index=False)
)

print("\nResults saved:", output_path)

print("\nTHRESHOLD ROBUSTNESS ANALYSIS COMPLETE!")