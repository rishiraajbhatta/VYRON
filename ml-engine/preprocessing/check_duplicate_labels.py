import pandas as pd
from pathlib import Path

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

pattern_labels = {}
total_rows = 0

print("CHECKING BINARY LABEL CONFLICTS")

for chunk in pd.read_csv(DATASET_PATH, chunksize=100000):

    total_rows += len(chunk)

    features = chunk.drop(columns=["Label"])

    hashes = pd.util.hash_pandas_object(
        features, index=False
    )

    binary_labels = chunk["Label"].apply(
        lambda x: "BENIGN" if x == "BENIGN" else "MALICIOUS"
    )

    for record_hash, label in zip(hashes, binary_labels):

        if record_hash not in pattern_labels:
            pattern_labels[record_hash] = set()

        pattern_labels[record_hash].add(label)

    print(f"Processed {total_rows:,} rows...")

conflicts = sum(
    1 for labels in pattern_labels.values()
    if len(labels) > 1
)

print("\nRESULTS")
print(f"Total records: {total_rows:,}")
print(f"Unique feature patterns: {len(pattern_labels):,}")
print(f"Binary conflicting patterns: {conflicts:,}")

print("\nBINARY INVESTIGATION COMPLETE")