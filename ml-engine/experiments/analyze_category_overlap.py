import pandas as pd
from pathlib import Path

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

print("=" * 75)
print("IOT INTRUSION DETECTION")
print("ATTACK CATEGORY OVERLAP STATISTICS")
print("=" * 75)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

feature_columns = [
    column for column in df.columns
    if column != "Label"
]

print("\nGenerating feature hashes...")

df["_group_id"] = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False
)

print("\nIdentifying overlapping patterns...")

label_counts = df.groupby("_group_id")["Label"].nunique()

overlapping_ids = label_counts[
    label_counts > 1
].index

df["_overlapping"] = df["_group_id"].isin(overlapping_ids)

print("\nCalculating category statistics...")

summary = (
    df.groupby("Label")
    .agg(
        total_records=("Label", "size"),
        overlapping_records=("_overlapping", "sum")
    )
)

summary["clean_records"] = (
    summary["total_records"] -
    summary["overlapping_records"]
)

summary["overlap_percentage"] = (
    summary["overlapping_records"] /
    summary["total_records"] * 100
)

summary = summary.sort_values(
    "clean_records",
    ascending=False
)

print("\nCATEGORY OVERLAP SUMMARY")
print("-" * 75)

print(
    summary.to_string(
        formatters={
            "total_records": "{:,.0f}".format,
            "overlapping_records": "{:,.0f}".format,
            "clean_records": "{:,.0f}".format,
            "overlap_percentage": "{:.2f}%".format
        }
    )
)

print("\n" + "=" * 75)
print("CATEGORY OVERLAP ANALYSIS COMPLETE")
print("=" * 75)