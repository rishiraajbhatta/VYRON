import json
import time
import random
import urllib.request
import urllib.error

import pandas as pd

from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml-engine"
    / "datasets"
    / "unseen_attack_processed"
    / "test.csv"
)

API_URL = "http://127.0.0.1:8000/predict/batch"

NUMBER_OF_BENIGN_RECORDS = 5
NUMBER_OF_MALICIOUS_RECORDS = 5

RANDOM_STATE = 42


# ==================================================
# SEND TRAFFIC TO BACKEND
# ==================================================

def send_to_backend(records):

    payload = json.dumps({
        "records": records
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=60
        ) as response:

            return json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.URLError as error:

        print("\nCould not connect to the backend.")
        print(f"Error: {error}")

        return None


# ==================================================
# MAIN SIMULATOR
# ==================================================

def main():

    print("=" * 65)
    print("IOT NETWORK TRAFFIC SIMULATOR")
    print("=" * 65)

    print("\nLoading CICIoT2023 test records...")

    if not DATASET_PATH.exists():

        print("Dataset file not found:")
        print(DATASET_PATH)
        return

    dataset = pd.read_csv(DATASET_PATH)

    benign = dataset[
        dataset["Target"] == 0
    ]

    malicious = dataset[
        dataset["Target"] == 1
    ]

    if len(benign) < NUMBER_OF_BENIGN_RECORDS:
        print("Not enough benign records.")
        return

    if len(malicious) < NUMBER_OF_MALICIOUS_RECORDS:
        print("Not enough malicious records.")
        return

    # Select actual records from the dataset
    benign_sample = benign.sample(
        n=NUMBER_OF_BENIGN_RECORDS,
        random_state=RANDOM_STATE
    )

    malicious_sample = malicious.sample(
        n=NUMBER_OF_MALICIOUS_RECORDS,
        random_state=RANDOM_STATE
    )

    # Combine and shuffle
    selected_records = pd.concat(
        [benign_sample, malicious_sample]
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "Target",
            "Original_Label"
        ]
    ]

    records = selected_records[
        feature_columns
    ].to_dict(orient="records")

    print(f"\nSelected records: {len(records)}")
    print(f"Benign records: {NUMBER_OF_BENIGN_RECORDS}")
    print(f"Malicious records: {NUMBER_OF_MALICIOUS_RECORDS}")

    print("\nSending network traffic to backend...")

    time.sleep(1)

    response = send_to_backend(records)

    if response is None:
        return

    # ==================================================
    # DISPLAY PREDICTIONS
    # ==================================================

    print("\n" + "=" * 65)
    print("INTRUSION DETECTION RESULTS")
    print("=" * 65)

    results = response["results"]

    correct_predictions = 0

    for index, prediction in enumerate(results):

        actual_category = selected_records.iloc[index][
            "Original_Label"
        ]

        actual_target = int(
            selected_records.iloc[index]["Target"]
        )

        actual_prediction = (
            "MALICIOUS"
            if actual_target == 1
            else "BENIGN"
        )

        predicted = prediction["prediction"]

        probability = prediction[
            "malicious_probability"
        ]

        correct = actual_prediction == predicted

        if correct:
            correct_predictions += 1

        print(f"\nRecord {index + 1}")
        print(f"Actual category: {actual_category}")
        print(f"Actual classification: {actual_prediction}")
        print(f"Model prediction: {predicted}")
        print(f"Malicious probability: {probability:.6f}")
        print(f"Correct prediction: {'YES' if correct else 'NO'}")

    # ==================================================
    # SUMMARY
    # ==================================================

    total = len(results)

    print("\n" + "=" * 65)
    print("SIMULATION SUMMARY")
    print("=" * 65)

    print(f"Total records processed: {total}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total - correct_predictions}")

    print(
        f"Prediction agreement: "
        f"{(correct_predictions / total) * 100:.2f}%"
    )

    print("\nTraffic simulation completed successfully.")


if __name__ == "__main__":
    main()