import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.inspection import permutation_importance

DATA_DIR = Path("ml-engine/datasets/processed")
MODEL_DIR = Path("ml-engine/models")
OUTPUT_DIR = Path("ml-engine/experiments")

print("=" * 65)
print("IOT INTRUSION DETECTION")
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

# Load validation data
validation_df = pd.read_csv(DATA_DIR / "validation.csv")

X_val = validation_df.drop(columns=["Target"])
y_val = validation_df["Target"]

# Use a reproducible sample to reduce computation time
sample = validation_df.sample(
    n=min(10000, len(validation_df)),
    random_state=42
)

X_sample = sample.drop(columns=["Target"])
y_sample = sample["Target"]

models = {
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

for model_name, filename in models.items():

    print("\n" + "=" * 65)
    print(f"ANALYZING: {model_name}")
    print("=" * 65)

    model = joblib.load(MODEL_DIR / filename)

    # Built-in feature importance
    importance_df = pd.DataFrame({
        "Feature": X_val.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP 15 BUILT-IN FEATURE IMPORTANCE:")
    print(importance_df.head(15).to_string(index=False))

    # Permutation importance
    print("\nCalculating permutation importance...")

    permutation = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=3,
        random_state=42,
        scoring="f1_macro",
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Importance": permutation.importances_mean,
        "Std": permutation.importances_std
    })

    permutation_df = permutation_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP 15 PERMUTATION IMPORTANCE:")
    print(permutation_df.head(15).to_string(index=False))

    # Save results
    importance_df.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_importance.csv",
        index=False
    )

    permutation_df.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_permutation.csv",
        index=False
    )

    # Plot built-in importance
    plt.figure(figsize=(11, 7))

    top_features = importance_df.head(15).sort_values(
        by="Importance"
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title(f"{model_name} - Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_importance.png",
        dpi=300
    )

    # Plot permutation importance
    plt.figure(figsize=(11, 7))

    top_permutation = permutation_df.head(15).sort_values(
        by="Importance"
    )

    plt.barh(
        top_permutation["Feature"],
        top_permutation["Importance"]
    )

    plt.title(f"{model_name} - Top 15 Permutation Importance")
    plt.xlabel("Decrease in Macro F1 after permutation")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_permutation.png",
        dpi=300
    )

# Display all four charts together
plt.show()

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 65)

print("\nResults saved inside:")
print("ml-engine/experiments/")