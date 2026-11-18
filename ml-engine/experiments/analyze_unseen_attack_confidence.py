import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# Configuration
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
print("UNSEEN ATTACK PREDICTION CONFIDENCE ANALYSIS")
print("=" * 75)

# Load model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["feature_columns"]

# Load test dataset
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_test = test_df[features]

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

test_df["Malicious_Probability"] = probabilities

# ==================================================
# 1. UNSEEN ATTACK CONFIDENCE
# ==================================================

print("\nINDIVIDUAL UNSEEN ATTACK CONFIDENCE")
print("-" * 75)

for attack in UNSEEN_ATTACKS:

    attack_df = test_df[
        test_df["Original_Label"] == attack
    ]

    scores = attack_df["Malicious_Probability"]

    print(f"\n{attack}")
    print(f"Records: {len(scores):,}")
    print(f"Minimum probability: {scores.min():.6f}")
    print(f"Mean probability: {scores.mean():.6f}")
    print(f"Median probability: {scores.median():.6f}")
    print(f"5th percentile: {scores.quantile(0.05):.6f}")
    print(f"25th percentile: {scores.quantile(0.25):.6f}")
    print(f"95th percentile: {scores.quantile(0.95):.6f}")
    print(f"Maximum probability: {scores.max():.6f}")

    print(
        "Predictions below threshold:",
        int((scores < THRESHOLD).sum())
    )

# ==================================================
# 2. BENIGN CONFIDENCE
# ==================================================

print("\n" + "=" * 75)
print("BENIGN TRAFFIC CONFIDENCE")
print("=" * 75)

benign_scores = test_df.loc[
    test_df["Target"] == 0,
    "Malicious_Probability"
]

print(f"Benign records: {len(benign_scores):,}")
print(f"Mean malicious probability: {benign_scores.mean():.6f}")
print(f"Median malicious probability: {benign_scores.median():.6f}")
print(f"95th percentile: {benign_scores.quantile(0.95):.6f}")
print(f"Maximum probability: {benign_scores.max():.6f}")

# ==================================================
# 3. SAVE CONFIDENCE RESULTS
# ==================================================

confidence_df = test_df[
    [
        "Original_Label",
        "Target",
        "Malicious_Probability"
    ]
]

output_path = Path(
    "ml-engine/experiments/unseen_attack_confidence.csv"
)

confidence_df.to_csv(
    output_path,
    index=False
)

print("\nConfidence results saved:", output_path)

print("\nCONFIDENCE ANALYSIS COMPLETE!")