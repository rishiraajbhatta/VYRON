import sys
import asyncio
import random

from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==================================================
# PROJECT PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_ENGINE_DIR = PROJECT_ROOT / "ml-engine"

DATASET_PATH = (
    ML_ENGINE_DIR
    / "datasets"
    / "unseen_attack_processed"
    / "test.csv"
)

sys.path.insert(0, str(ML_ENGINE_DIR))

from src.predictor import IoTIntrusionPredictor


# ==================================================
# INITIALIZE APPLICATION
# ==================================================

app = FastAPI(
    title="VYRON - IoT Intrusion Detection API",
    description="Machine Learning Based IoT Network Intrusion Detection System",
    version="1.2.0"
)


# ==================================================
# CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# LOAD MACHINE LEARNING MODEL
# ==================================================

predictor = IoTIntrusionPredictor()


# ==================================================
# LOAD TEST DATASET
# ==================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {DATASET_PATH}"
    )

print("Loading VYRON test dataset...")

test_dataset = pd.read_csv(DATASET_PATH)

print(f"Test dataset loaded: {len(test_dataset)} records")


# ==================================================
# SIMULATION STATE
# ==================================================

simulation_running = False
simulation_task = None

simulation_interval = 1.0

event_counter = 0

simulation_events = deque(maxlen=500)


# ==================================================
# REQUEST MODELS
# ==================================================

class PredictionRequest(BaseModel):
    data: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]]


class SimulationStartRequest(BaseModel):
    interval_seconds: float = Field(
        default=1.0,
        ge=0.25,
        le=30.0
    )


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "message": "VYRON IoT Intrusion Detection API",
        "status": "running",
        "version": "1.2.0"
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost",
        "model_loaded": predictor.model is not None,
        "features": len(predictor.feature_columns),
        "threshold": predictor.threshold
    }


# ==================================================
# MODEL INFORMATION
# ==================================================

@app.get("/model/info")
def model_info():

    return {
        "model_name": "XGBoost",
        "features": len(predictor.feature_columns),
        "feature_columns": predictor.feature_columns,
        "threshold": predictor.threshold,
        "supported_predictions": [
            "BENIGN",
            "MALICIOUS"
        ]
    }


# ==================================================
# REAL DATASET SAMPLE LOADER
# ==================================================

@app.get("/samples/{sample_type}")
def load_sample(sample_type: str):

    if sample_type.lower() not in ["benign", "malicious"]:

        raise HTTPException(
            status_code=400,
            detail="Sample type must be benign or malicious."
        )

    try:

        target_value = (
            0 if sample_type.lower() == "benign" else 1
        )

        matching_records = test_dataset[
            test_dataset["Target"] == target_value
        ]

        if matching_records.empty:

            raise HTTPException(
                status_code=404,
                detail=f"No {sample_type} records found."
            )

        row = matching_records.iloc[0]

        feature_data = row[
            predictor.feature_columns
        ].to_dict()

        feature_data = {
            key: float(value)
            for key, value in feature_data.items()
        }

        return {
            "success": True,
            "expected_class": sample_type.upper(),
            "original_label": str(row["Original_Label"]),
            "sample_source": "CICIoT2023 held-out test dataset",
            "total_features": len(feature_data),
            "data": feature_data
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==================================================
# SINGLE PREDICTION
# ==================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        result = predictor.predict_one(request.data)

        return {
            "success": True,
            "result": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ==================================================
# BATCH PREDICTION
# ==================================================

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):

    if not request.records:

        raise HTTPException(
            status_code=400,
            detail="At least one record is required."
        )

    try:

        results = predictor.predict_batch(
            request.records
        )

        return {
            "success": True,
            "total_records": len(results),
            "results": results.to_dict(
                orient="records"
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ==================================================
# CONTINUOUS TRAFFIC SIMULATION ENGINE
# ==================================================

async def run_traffic_simulation():

    global simulation_running
    global event_counter

    print("\n[VYRON] Traffic simulation started.")

    try:

        while simulation_running:

            # Select a random real dataset record.
            row = test_dataset.sample(
                n=1
            ).iloc[0]

            # Extract the 38 model features.
            feature_data = {
                column: float(row[column])
                for column in predictor.feature_columns
            }

            # Get the original dataset information.
            actual_target = int(row["Target"])

            actual_class = (
                "MALICIOUS"
                if actual_target == 1
                else "BENIGN"
            )

            original_label = str(
                row["Original_Label"]
            )

            # Run XGBoost prediction.
            prediction = predictor.predict_one(
                feature_data
            )

            predicted_class = prediction["prediction"]

            malicious_probability = float(
                prediction["malicious_probability"]
            )

            # Compare model prediction with dataset label.
            matches_dataset = (
                actual_class == predicted_class
            )

            event_counter += 1

            event = {
                "id": event_counter,

                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),

                "actual_class": actual_class,

                "predicted_class": predicted_class,

                "original_label": original_label,

                "malicious_probability": malicious_probability,

                "threshold": float(
                    predictor.threshold
                ),

                "matches_dataset": matches_dataset,

                "detection_model": "XGBoost",

                "sample_source": "CICIoT2023 held-out test dataset"
            }

            simulation_events.append(event)

            print(
                f"[VYRON] Event #{event_counter} | "
                f"Actual: {actual_class} | "
                f"Predicted: {predicted_class} | "
                f"Probability: {malicious_probability:.6f}"
            )

            await asyncio.sleep(
                simulation_interval
            )

    except asyncio.CancelledError:

        print("[VYRON] Simulation task cancelled.")

        raise

    except Exception as error:

        print(
            f"[VYRON] Simulation error: {error}"
        )

    finally:

        simulation_running = False

        print("[VYRON] Traffic simulation stopped.")


# ==================================================
# START SIMULATION
# ==================================================

@app.post("/simulation/start")
async def start_simulation(
    request: SimulationStartRequest
):

    global simulation_running
    global simulation_task
    global simulation_interval

    if simulation_running:

        return {
            "success": True,
            "message": "Simulation is already running.",
            "running": True,
            "interval_seconds": simulation_interval
        }

    simulation_interval = request.interval_seconds

    simulation_running = True

    simulation_task = asyncio.create_task(
        run_traffic_simulation()
    )

    return {
        "success": True,
        "message": "Traffic simulation started successfully.",
        "running": True,
        "interval_seconds": simulation_interval
    }


# ==================================================
# STOP SIMULATION
# ==================================================

@app.post("/simulation/stop")
async def stop_simulation():

    global simulation_running
    global simulation_task

    if not simulation_running:

        return {
            "success": True,
            "message": "Simulation is not running.",
            "running": False
        }

    simulation_running = False

    if simulation_task is not None:

        simulation_task.cancel()

        try:

            await simulation_task

        except asyncio.CancelledError:

            pass

        simulation_task = None

    return {
        "success": True,
        "message": "Traffic simulation stopped successfully.",
        "running": False
    }


# ==================================================
# SIMULATION STATUS
# ==================================================

@app.get("/simulation/status")
def simulation_status():

    return {
        "running": simulation_running,

        "total_events": event_counter,

        "interval_seconds": simulation_interval,

        "stored_events": len(simulation_events),

        "detection_model": "XGBoost"
    }


# ==================================================
# RETRIEVE SIMULATION EVENTS
# ==================================================

@app.get("/simulation/events")
def get_simulation_events(
    after_id: int = 0,
    limit: int = 100
):

    if limit < 1 or limit > 500:

        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 500."
        )

    events = [
        event
        for event in simulation_events
        if event["id"] > after_id
    ]

    events = events[:limit]

    return {
        "success": True,

        "total_events": len(events),

        "latest_event_id": event_counter,

        "events": list(events)
    }