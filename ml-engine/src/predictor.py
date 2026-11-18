import joblib
import numpy as np
import pandas as pd

from pathlib import Path


class IoTIntrusionPredictor:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            self.project_root
            / "ml-engine"
            / "models"
            / "xgboost_unseen_comparison.joblib"
        )

        self.training_data_path = (
            self.project_root
            / "ml-engine"
            / "datasets"
            / "unseen_attack_processed"
            / "train.csv"
        )

        print("Loading XGBoost intrusion detection model...")

        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.threshold = artifact["threshold"]

        print("Loading training feature statistics...")

        training_data = pd.read_csv(
            self.training_data_path,
            usecols=self.feature_columns
        )

        self.feature_medians = training_data.median()

        print("Model loaded successfully.")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Detection threshold: {self.threshold}")

    def _prepare_data(self, data):

        if isinstance(data, dict):

            data = pd.DataFrame([data])

        elif isinstance(data, list):

            data = pd.DataFrame(data)

        elif isinstance(data, pd.Series):

            data = data.to_frame().T

        elif not isinstance(data, pd.DataFrame):

            raise TypeError(
                "Input must be a dictionary, list of dictionaries, "
                "pandas Series, or pandas DataFrame."
            )

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in data.columns
        ]

        if missing_features:

            print(
                f"Warning: {len(missing_features)} features missing. "
                "Training medians will be used."
            )

        prepared = data.reindex(
            columns=self.feature_columns
        ).copy()

        for column in self.feature_columns:

            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce"
            )

        prepared = prepared.replace(
            [np.inf, -np.inf],
            np.nan
        )

        prepared = prepared.fillna(
            self.feature_medians
        )

        return prepared

    def predict_one(self, data):

        prepared = self._prepare_data(data)

        probability = float(
            self.model.predict_proba(prepared)[0][1]
        )

        prediction = (
            "MALICIOUS"
            if probability >= self.threshold
            else "BENIGN"
        )

        return {
            "prediction": prediction,
            "malicious_probability": round(probability, 6),
            "threshold": self.threshold
        }

    def predict_batch(self, data):

        prepared = self._prepare_data(data)

        probabilities = self.model.predict_proba(
            prepared
        )[:, 1]

        predictions = np.where(
            probabilities >= self.threshold,
            "MALICIOUS",
            "BENIGN"
        )

        results = pd.DataFrame({
            "prediction": predictions,
            "malicious_probability": probabilities.round(6)
        })

        return results


if __name__ == "__main__":

    print("=" * 60)
    print("IOT INTRUSION DETECTION PREDICTION ENGINE")
    print("=" * 60)

    predictor = IoTIntrusionPredictor()

    print("\nPrediction engine initialized successfully.")

    print("\nAvailable methods:")
    print("1. predict_one(data)")
    print("2. predict_batch(data)")

    print("\nPrediction engine is ready.")