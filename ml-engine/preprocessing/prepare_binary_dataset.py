import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

OUTPUT_DIR = Path("ml-engine/datasets/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 65)
print("CURASYNC? NO — IoT INTRUSION DETECTION")
print("BINARY DATASET PREPROCESSING")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Original records: {len(df):,}")

# --------------------------------------------------
# 2. CREATE BINARY LABELS
# --------------------------------------------------

print("\nConverting attack labels...")

df["Target"] = np.where(
    df["Label"] == "BENIGN", 0, 1
)

df = df.drop(columns=["Label"])

# --------------------------------------------------
# 3. IDENTIFY IDENTICAL FEATURE PATTERNS
# --------------------------------------------------

print("\nIdentifying duplicate feature patterns...")

feature_columns = [
    column for column in df.columns
    if column != "Target"
]

df["_group_id"] = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False
)

# Find feature patterns associated with both labels
label_counts = df.groupby("_group_id")["Target"].nunique()

conflicting_groups = label_counts[
    label_counts > 1
].index

print(f"Conflicting patterns: {len(conflicting_groups):,}")

# Remove all records belonging to conflicting patterns
df = df[
    ~df["_group_id"].isin(conflicting_groups)
].copy()

print(f"Records after conflict removal: {len(df):,}")

# --------------------------------------------------
# 4. HANDLE INFINITE VALUES
# --------------------------------------------------

print("\nHandling infinite values...")

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# --------------------------------------------------
# 5. SPLIT UNIQUE FEATURE GROUPS
# --------------------------------------------------

print("\nCreating leakage-resistant data splits...")

groups = (
    df.groupby("_group_id")["Target"]
    .first()
    .reset_index()
)

train_groups, temporary_groups = train_test_split(
    groups,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=groups["Target"]
)

validation_groups, test_groups = train_test_split(
    temporary_groups,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temporary_groups["Target"]
)

train_ids = set(train_groups["_group_id"])
validation_ids = set(validation_groups["_group_id"])
test_ids = set(test_groups["_group_id"])

train_df = df[df["_group_id"].isin(train_ids)].copy()
validation_df = df[df["_group_id"].isin(validation_ids)].copy()
test_df = df[df["_group_id"].isin(test_ids)].copy()

# --------------------------------------------------
# 6. HANDLE MISSING VALUES
# --------------------------------------------------

print("\nHandling missing values...")

# Calculate medians using training data only
medians = train_df[feature_columns].median()

train_df[feature_columns] = train_df[feature_columns].fillna(medians)

validation_df[feature_columns] = validation_df[feature_columns].fillna(medians)

test_df[feature_columns] = test_df[feature_columns].fillna(medians)

# --------------------------------------------------
# 7. SAVE PROCESSED DATASETS
# --------------------------------------------------

print("\nSaving datasets...")

for name, dataset in [
    ("train.csv", train_df),
    ("validation.csv", validation_df),
    ("test.csv", test_df)
]:
    dataset = dataset.drop(columns=["_group_id"])

    dataset.to_csv(
        OUTPUT_DIR / name,
        index=False
    )

    print(f"Saved {name}: {len(dataset):,} records")

# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("PREPROCESSING SUMMARY")
print("=" * 65)

print(f"Training records:   {len(train_df):,}")
print(f"Validation records: {len(validation_df):,}")
print(f"Testing records:    {len(test_df):,}")

print("\nTraining label distribution:")
print(train_df["Target"].value_counts())

print("\nValidation label distribution:")
print(validation_df["Target"].value_counts())

print("\nTesting label distribution:")
print(test_df["Target"].value_counts())

print("\nPREPROCESSING COMPLETE!")