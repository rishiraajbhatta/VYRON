import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

# ==================================================
# CONFIGURATION
# ==================================================

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

OUTPUT_DIR = Path(
    "ml-engine/datasets/unseen_attack_processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

UNSEEN_ATTACKS = [
    "DDOS-ICMP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN"
]

print("=" * 75)
print("IOT INTRUSION DETECTION")
print("UNSEEN ATTACK CATEGORY PREPROCESSING")
print("=" * 75)

# ==================================================
# 1. LOAD DATASET
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Original records: {len(df):,}")

# Preserve original attack labels
df["Original_Label"] = df["Label"]

# Binary classification target
df["Target"] = np.where(
    df["Original_Label"] == "BENIGN",
    0,
    1
)

# ==================================================
# 2. IDENTIFY FEATURE COLUMNS
# ==================================================

feature_columns = [
    column for column in df.columns
    if column not in [
        "Label",
        "Original_Label",
        "Target"
    ]
]

print(f"Feature count: {len(feature_columns)}")

# ==================================================
# 3. GENERATE FEATURE GROUP IDS
# ==================================================

print("\nGenerating feature-pattern hashes...")

df["_group_id"] = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False
)

# ==================================================
# 4. REMOVE CROSS-CATEGORY OVERLAPPING PATTERNS
# ==================================================

print("\nIdentifying ambiguous feature patterns...")

label_counts = (
    df.groupby("_group_id")["Original_Label"]
    .nunique()
)

conflicting_ids = set(
    label_counts[label_counts > 1].index
)

print(
    "Cross-category overlapping patterns:",
    f"{len(conflicting_ids):,}"
)

df = df[
    ~df["_group_id"].isin(conflicting_ids)
].copy()

print(
    "Records after overlap removal:",
    f"{len(df):,}"
)

# ==================================================
# 5. HANDLE INFINITE VALUES
# ==================================================

print("\nHandling infinite values...")

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# ==================================================
# 6. SEPARATE SEEN AND UNSEEN ATTACKS
# ==================================================

print("\nSeparating seen and unseen attack categories...")

available_labels = set(df["Original_Label"].unique())

missing_attacks = (
    set(UNSEEN_ATTACKS) - available_labels
)

if missing_attacks:
    raise ValueError(
        f"Selected unseen attacks missing from dataset: {missing_attacks}"
    )

unseen_df = df[
    df["Original_Label"].isin(UNSEEN_ATTACKS)
].copy()

seen_df = df[
    ~df["Original_Label"].isin(UNSEEN_ATTACKS)
].copy()

print(f"Seen-category records: {len(seen_df):,}")
print(f"Unseen-category records: {len(unseen_df):,}")

# ==================================================
# 7. SPLIT SEEN DATA INTO TRAIN / VALIDATION / TEST
# ==================================================

print("\nSplitting seen categories into three datasets...")

seen_groups = (
    seen_df.groupby("_group_id")["Target"]
    .first()
    .reset_index()
)

train_groups, temporary_groups = train_test_split(
    seen_groups,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=seen_groups["Target"]
)

validation_groups, seen_test_groups = train_test_split(
    temporary_groups,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temporary_groups["Target"]
)

train_ids = set(train_groups["_group_id"])
validation_ids = set(validation_groups["_group_id"])
seen_test_ids = set(seen_test_groups["_group_id"])

train_df = seen_df[
    seen_df["_group_id"].isin(train_ids)
].copy()

validation_df = seen_df[
    seen_df["_group_id"].isin(validation_ids)
].copy()

seen_test_df = seen_df[
    seen_df["_group_id"].isin(seen_test_ids)
].copy()

# ==================================================
# 8. CREATE FINAL TEST DATASET
# ==================================================

print("\nAdding unseen attacks to the test dataset...")

test_df = pd.concat(
    [seen_test_df, unseen_df],
    ignore_index=True
)

print(f"Training records: {len(train_df):,}")
print(f"Validation records: {len(validation_df):,}")
print(f"Testing records: {len(test_df):,}")

# ==================================================
# 9. VERIFY FEATURE GROUP SEPARATION
# ==================================================

print("\nVerifying feature-group separation...")

train_ids = set(train_df["_group_id"])
validation_ids = set(validation_df["_group_id"])
test_ids = set(test_df["_group_id"])

assert train_ids.isdisjoint(validation_ids)
assert train_ids.isdisjoint(test_ids)
assert validation_ids.isdisjoint(test_ids)

print("Feature-group leakage check: PASSED")

# ==================================================
# 10. HANDLE MISSING VALUES
# ==================================================

print("\nHandling missing values...")

# Fit imputation values on training data only
medians = train_df[feature_columns].median()

train_df[feature_columns] = (
    train_df[feature_columns].fillna(medians)
)

validation_df[feature_columns] = (
    validation_df[feature_columns].fillna(medians)
)

test_df[feature_columns] = (
    test_df[feature_columns].fillna(medians)
)

# ==================================================
# 11. REMOVE INTERNAL COLUMNS
# ==================================================

for dataset in [train_df, validation_df, test_df]:

    dataset.drop(
        columns=["_group_id", "Label"],
        inplace=True,
        errors="ignore"
    )

# ==================================================
# 12. SAVE DATASETS
# ==================================================

print("\nSaving processed datasets...")

datasets = {
    "train.csv": train_df,
    "validation.csv": validation_df,
    "test.csv": test_df
}

for filename, dataset in datasets.items():

    dataset.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )

    print(
        f"Saved {filename}: {len(dataset):,} records"
    )

# ==================================================
# 13. FINAL DATASET SUMMARY
# ==================================================

print("\n" + "=" * 75)
print("PREPROCESSING SUMMARY")
print("=" * 75)

for name, dataset in [
    ("TRAINING", train_df),
    ("VALIDATION", validation_df),
    ("TESTING", test_df)
]:

    print(f"\n{name}")

    print(f"Records: {len(dataset):,}")

    print("\nBinary distribution:")

    print(
        dataset["Target"]
        .value_counts()
        .sort_index()
    )

    print("\nOriginal attack categories:")

    print(
        dataset["Original_Label"]
        .value_counts()
        .to_string()
    )

# ==================================================
# 14. VERIFY UNSEEN ATTACK SEPARATION
# ==================================================

print("\n" + "=" * 75)
print("UNSEEN ATTACK SEPARATION VERIFICATION")
print("=" * 75)

train_attacks = set(
    train_df["Original_Label"].unique()
)

validation_attacks = set(
    validation_df["Original_Label"].unique()
)

test_attacks = set(
    test_df["Original_Label"].unique()
)

train_attacks.discard("BENIGN")
validation_attacks.discard("BENIGN")
test_attacks.discard("BENIGN")

print("\nSelected unseen attacks:")

for attack in UNSEEN_ATTACKS:
    print(f"  {attack}")

print("\nUnseen attacks found in training:")

print(
    train_attacks.intersection(UNSEEN_ATTACKS)
    or "NONE"
)

print("\nUnseen attacks found in validation:")

print(
    validation_attacks.intersection(UNSEEN_ATTACKS)
    or "NONE"
)

print("\nUnseen attacks found in testing:")

print(
    test_attacks.intersection(UNSEEN_ATTACKS)
    or "NONE"
)

assert not train_attacks.intersection(UNSEEN_ATTACKS)
assert not validation_attacks.intersection(UNSEEN_ATTACKS)
assert set(UNSEEN_ATTACKS).issubset(test_attacks)

print("\nAll unseen attack separation checks: PASSED")

print("\nPREPROCESSING COMPLETE!")
