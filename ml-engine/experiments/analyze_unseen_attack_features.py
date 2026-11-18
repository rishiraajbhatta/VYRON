import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import roc_auc_score

# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("ml-engine/datasets/unseen_attack_processed")

MODEL_PATH = Path(
    "ml-engine/models/xgboost_unseen_attack.joblib"
)

UNSEEN_ATTACKS = [
    "DDOS-ICMP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN"
]

OUTPUT_DIR = Path("ml-engine/experiments")

print("=" * 75)
print("FEATURE-LEVEL INVESTIGATION")
print("UNSEEN ATTACK GENERALIZATION")
print("=" * 75)

# ==================================================
# 1. LOAD MODEL AND TEST DATA
# ==================================================

print("\nLoading trained model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["feature_columns"]

print("\nLoading test dataset...")

df = pd.read_csv(DATA_DIR / "test.csv")

benign_df = df[df["Target"] == 0]

print(f"Total test records: {len(df):,}")
print(f"Benign records: {len(benign_df):,}")

# ==================================================
# 2. XGBOOST FEATURE IMPORTANCE
# ==================================================

print("\n" + "=" * 75)
print("TRAINED MODEL FEATURE IMPORTANCE")
print("=" * 75)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 20 FEATURES:")

print(
    importance_df.head(20).to_string(index=False)
)

importance_df.to_csv(
    OUTPUT_DIR / "unseen_model_feature_importance.csv",
    index=False
)

# ==================================================
# 3. FEATURE SEPARATION ANALYSIS
# ==================================================

print("\n" + "=" * 75)
print("BENIGN VS UNSEEN ATTACK FEATURE ANALYSIS")
print("=" * 75)

all_results = []

for attack in UNSEEN_ATTACKS:

    print("\n" + "-" * 75)
    print(f"ANALYZING: {attack}")
    print("-" * 75)

    attack_df = df[
        df["Original_Label"] == attack
    ]

    results = []

    for feature in features:

        benign_values = benign_df[feature].to_numpy()
        attack_values = attack_df[feature].to_numpy()

        combined_values = np.concatenate([
            benign_values,
            attack_values
        ])

        combined_labels = np.concatenate([
            np.zeros(len(benign_values)),
            np.ones(len(attack_values))
        ])

        # Skip features with no variation
        if np.unique(combined_values).size < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(
                combined_labels,
                combined_values
            )

        results.append({
            "Feature": feature,
            "Benign_Median": np.median(benign_values),
            "Attack_Median": np.median(attack_values),
            "Univariate_ROC_AUC": auc,
            "Separation": abs(auc - 0.5) * 2
        })

    attack_results = pd.DataFrame(results)

    attack_results = attack_results.sort_values(
        "Separation",
        ascending=False
    )

    print("\nTOP 10 DISTINGUISHING FEATURES:")

    print(
        attack_results.head(10).to_string(index=False)
    )

    attack_results["Attack"] = attack

    all_results.append(attack_results)

# ==================================================
# 4. COMBINE CATEGORY RESULTS
# ==================================================

combined_df = pd.concat(
    all_results,
    ignore_index=True
)

combined_df.to_csv(
    OUTPUT_DIR / "unseen_attack_feature_separation.csv",
    index=False
)

# ==================================================
# 5. FEATURES COMMON TO ALL FOUR ATTACKS
# ==================================================

print("\n" + "=" * 75)
print("COMMON IMPORTANT FEATURES")
print("=" * 75)

top_features = {}

for attack in UNSEEN_ATTACKS:

    attack_data = combined_df[
        combined_df["Attack"] == attack
    ]

    top_features[attack] = set(
        attack_data.head(10)["Feature"]
    )

common_features = set.intersection(
    *top_features.values()
)

print("\nFeatures appearing in the top 10 for all attacks:")

if common_features:

    for feature in sorted(common_features):
        print("-", feature)

else:
    print("No common features found in all four top-10 lists.")

# ==================================================
# 6. COMPLETE
# ==================================================

print("\n" + "=" * 75)
print("FEATURE INVESTIGATION COMPLETE")
print("=" * 75)

print("\nSaved files:")
print("- unseen_model_feature_importance.csv")
print("- unseen_attack_feature_separation.csv")