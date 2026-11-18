import pandas as pd
from pathlib import Path

DATASET_PATH = Path("ml-engine/datasets/Merged01.csv")

print("=" * 65)
print("IOT INTRUSION DETECTION")
print("ATTACK CATEGORY INSPECTION")
print("=" * 65)

df = pd.read_csv(
    DATASET_PATH,
    usecols=["Label"]
)

distribution = df["Label"].value_counts()

print("\nTOTAL ATTACK CATEGORIES:", len(distribution))
print("\nCATEGORY DISTRIBUTION:\n")

for category, count in distribution.items():
    percentage = count / len(df) * 100

    print(
        f"{category:<30} "
        f"{count:>10,} "
        f"({percentage:.3f}%)"
    )

print("\n" + "=" * 65)
print("ATTACK CATEGORY INSPECTION COMPLETE")
print("=" * 65)