import pandas as pd
from pathlib import Path

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

print("=" * 70)
print("IOT INTRUSION DETECTION")
print("CROSS-ATTACK FEATURE OVERLAP ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

feature_columns = [
    column for column in df.columns
    if column != "Label"
]

print(f"Records: {len(df):,}")
print(f"Features: {len(feature_columns)}")

print("\nGenerating feature-pattern hashes...")

df["_group_id"] = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False
)

print("\nIdentifying patterns shared across attack categories...")

label_counts = (
    df.groupby("_group_id")["Label"]
    .nunique()
)

overlapping_groups = label_counts[label_counts > 1]

print("\nRESULTS")
print("-" * 70)

print(
    "Unique feature patterns associated with multiple labels:",
    f"{len(overlapping_groups):,}"
)

overlap_ids = overlapping_groups.index

overlap_df = df[
    df["_group_id"].isin(overlap_ids)
]

print(
    "Total records belonging to overlapping patterns:",
    f"{len(overlap_df):,}"
)

print("\nMost common cross-label feature patterns:")

overlap_summary = (
    overlap_df.groupby("_group_id")
    .agg(
        record_count=("Label", "size"),
        category_count=("Label", "nunique"),
        categories=("Label", lambda x: ", ".join(sorted(x.unique())))
    )
    .sort_values("record_count", ascending=False)
)

print(overlap_summary.head(20).to_string())

print("\n" + "=" * 70)
print("OVERLAP ANALYSIS COMPLETE")
print("=" * 70)