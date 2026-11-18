import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("ml-engine/experiments")

# Load results
test_results = pd.read_csv(RESULTS_DIR / "test_results.csv")
threshold_results = pd.read_csv(RESULTS_DIR / "threshold_results.csv")

print("Generating research visualizations...")

# 1. Model performance comparison
metrics = ["Accuracy", "ROC-AUC", "PR-AUC", "Macro F1"]

test_results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Model Performance Comparison on Held-Out Test Data")
plt.ylabel("Score")
plt.ylim(0.75, 1.02)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300)



# 2. Threshold vs Macro F1
plt.figure(figsize=(10, 6))

for model_name, group in threshold_results.groupby("Model"):
    plt.plot(
        group["Threshold"],
        group["Macro F1"],
        marker="o",
        label=model_name
    )

plt.title("Classification Threshold vs Macro F1")
plt.xlabel("Classification Threshold")
plt.ylabel("Macro F1")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "threshold_vs_macro_f1.png", dpi=300)



# 3. Malicious Recall vs Benign Recall
plt.figure(figsize=(10, 6))

for model_name, group in threshold_results.groupby("Model"):
    plt.plot(
        group["Benign Recall"],
        group["Malicious Recall"],
        marker="o",
        label=model_name
    )

plt.title("Malicious Recall vs Benign Recall")
plt.xlabel("Benign Recall")
plt.ylabel("Malicious Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "recall_tradeoff.png", dpi=300)



# 4. False Positives vs False Negatives
plt.figure(figsize=(10, 6))

for model_name, group in threshold_results.groupby("Model"):
    plt.plot(
        group["False Positives"],
        group["False Negatives"],
        marker="o",
        label=model_name
    )

plt.title("False Positive vs False Negative Trade-off")
plt.xlabel("False Positives")
plt.ylabel("False Negatives")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "false_positive_vs_negative.png", dpi=300)



print("\nAll visualizations generated successfully!")

print("\nSaved files:")
print("1. model_comparison.png")
print("2. threshold_vs_macro_f1.png")
print("3. recall_tradeoff.png")
print("4. false_positive_vs_negative.png")
plt.show()