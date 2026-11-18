import sys
import pandas as pd

from pathlib import Path

# Make the ml-engine directory importable
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.predictor import IoTIntrusionPredictor


# ==========================================
# LOAD PREDICTION ENGINE
# ==========================================

print("=" * 65)
print("PREDICTION ENGINE TEST")
print("=" * 65)

predictor = IoTIntrusionPredictor()


# ==========================================
# LOAD REAL TEST RECORDS
# ==========================================

test_path = (
    PROJECT_DIR
    / "datasets"
    / "unseen_attack_processed"
    / "test.csv"
)

print("\nLoading real test records...")

test_df = pd.read_csv(test_path)

# Select five benign records
benign_records = test_df[
    test_df["Original_Label"] == "BENIGN"
].head(5)

# Select five previously unseen attack records
malicious_records = test_df[
    test_df["Original_Label"] == "DDOS-ICMP_FLOOD"
].head(5)

# Combine records
sample = pd.concat(
    [benign_records, malicious_records],
    ignore_index=True
)

features = predictor.feature_columns

X_sample = sample[features]

print(f"Selected records: {len(sample)}")


# ==========================================
# INDIVIDUAL PREDICTION TEST
# ==========================================

print("\n" + "=" * 65)
print("INDIVIDUAL PREDICTION TEST")
print("=" * 65)

for index in [0, 5]:

    record = X_sample.iloc[index]

    actual_label = sample.iloc[index]["Original_Label"]

    result = predictor.predict_one(record)

    print(f"\nRecord: {index + 1}")
    print(f"Actual category: {actual_label}")
    print(f"Predicted: {result['prediction']}")
    print(
        f"Malicious probability: "
        f"{result['malicious_probability']:.6f}"
    )
    print(f"Threshold: {result['threshold']}")


# ==========================================
# BATCH PREDICTION TEST
# ==========================================

print("\n" + "=" * 65)
print("BATCH PREDICTION TEST")
print("=" * 65)

batch_results = predictor.predict_batch(X_sample)

comparison = pd.DataFrame({
    "Actual_Category": sample["Original_Label"],
    "Actual_Target": sample["Target"],
    "Predicted": batch_results["prediction"],
    "Malicious_Probability": batch_results[
        "malicious_probability"
    ]
})

print("\nPrediction Results:\n")

print(comparison.to_string(index=False))


# ==========================================
# TEST SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("PREDICTION ENGINE TEST COMPLETE")
print("=" * 65)

correct = (
    (comparison["Actual_Target"] == 1)
    == (comparison["Predicted"] == "MALICIOUS")
)

print(f"Correct predictions: {correct.sum()}/{len(correct)}")
print(f"Incorrect predictions: {(~correct).sum()}/{len(correct)}")