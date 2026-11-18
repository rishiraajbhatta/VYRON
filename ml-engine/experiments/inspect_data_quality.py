import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

chunk_size = 100000

missing_counts = Counter()
infinite_counts = Counter()
duplicate_labels = Counter()

seen_hashes = set()
total_rows = 0
duplicate_rows = 0

print("=" * 60)
print("DETAILED DATA QUALITY ANALYSIS")
print("=" * 60)

for chunk in pd.read_csv(DATASET_PATH, chunksize=chunk_size):

    total_rows += len(chunk)

    # Missing values by column
    missing = chunk.isnull().sum()
    for column, count in missing.items():
        missing_counts[column] += count

    # Infinite values by column
    numeric = chunk.select_dtypes(include=np.number)
    infinite = np.isinf(numeric).sum()

    for column, count in infinite.items():
        infinite_counts[column] += count

    # Duplicate records by attack label
    hashes = pd.util.hash_pandas_object(chunk, index=False)

    for row_hash, label in zip(hashes, chunk["Label"]):
        if row_hash in seen_hashes:
            duplicate_rows += 1
            duplicate_labels[label] += 1
        else:
            seen_hashes.add(row_hash)

    print(f"Processed {total_rows:,} rows...")

print("\nMISSING VALUES BY COLUMN")
for column, count in missing_counts.items():
    if count > 0:
        print(f"{column}: {count}")

print("\nINFINITE VALUES BY COLUMN")
for column, count in infinite_counts.items():
    if count > 0:
        print(f"{column}: {count}")

print("\nDUPLICATES BY ATTACK CATEGORY")
for label, count in duplicate_labels.most_common():
    print(f"{label}: {count}")

print("\nTOTAL DUPLICATES:", duplicate_rows)

print("\nDETAILED ANALYSIS COMPLETE")