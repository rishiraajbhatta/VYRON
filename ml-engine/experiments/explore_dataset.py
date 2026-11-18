import pandas as pd
from pathlib import Path

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

print("=" * 60)
print("FULL DATASET ANALYSIS")
print("=" * 60)

# Process the dataset in chunks to save RAM
chunk_size = 100000

total_rows = 0
attack_counts = {}

print("\nReading dataset in chunks...")

for chunk in pd.read_csv(DATASET_PATH, chunksize=chunk_size):

    total_rows += len(chunk)

    counts = chunk["Label"].value_counts()

    for label, count in counts.items():
        attack_counts[label] = attack_counts.get(label, 0) + count

    print(f"Processed {total_rows:,} rows...")

print("\n" + "=" * 60)
print("FULL DATASET RESULTS")
print("=" * 60)

print(f"\nTotal records: {total_rows:,}")
print(f"Total attack categories: {len(attack_counts)}")

print("\nATTACK DISTRIBUTION:")

results = pd.Series(attack_counts).sort_values(ascending=False)

print(results.to_string())

print("\nBENIGN RECORDS:")
print(attack_counts.get("BENIGN", 0))

print("\nMALICIOUS RECORDS:")
print(total_rows - attack_counts.get("BENIGN", 0))

print("\n" + "=" * 60)
print("FULL DATASET ANALYSIS COMPLETE")
print("=" * 60)