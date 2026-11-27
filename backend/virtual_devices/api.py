import random

from fastapi import APIRouter, HTTPException, Query

from .device_engine import VirtualDeviceEngine
from .device_models import DeviceActivityRequest


router = APIRouter(
    prefix="/devices",
    tags=["Virtual IoT Devices"],
)

# Shared engine instance for all device API requests.
device_engine = VirtualDeviceEngine()


# ==================================================
# GET ALL VIRTUAL DEVICES
# ==================================================

@router.get("/")
def get_devices():

    devices = device_engine.get_all_devices()

    return {
        "success": True,
        "total_devices": len(devices),
        "devices": [
            device.model_dump(mode="json")
            for device in devices
        ],
    }


# ==================================================
# GET DEVICE STATISTICS
# ==================================================

@router.get("/status")
def get_device_status():

    return {
        "success": True,
        **device_engine.get_device_status(),
    }


# ==================================================
# START A VIRTUAL DEVICE
# ==================================================

@router.post("/{device_id}/start")
def start_device(device_id: str):

    try:

        device = device_engine.start_device(device_id)

        return {
            "success": True,
            "message": f"{device.name} started successfully.",
            "device": device.model_dump(mode="json"),
        }

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


# ==================================================
# STOP A VIRTUAL DEVICE
# ==================================================

@router.post("/{device_id}/stop")
def stop_device(device_id: str):

    try:

        device = device_engine.stop_device(device_id)

        return {
            "success": True,
            "message": f"{device.name} stopped successfully.",
            "device": device.model_dump(mode="json"),
        }

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


# ==================================================
# SIMULATE DEVICE ACTIVITY
# ==================================================

@router.post("/{device_id}/simulate")
def simulate_device_activity(
    device_id: str,
    request: DeviceActivityRequest,
):

    try:

        event = device_engine.simulate_activity(
            device_id=device_id,
            scenario=request.scenario,
        )

        return {
            "success": True,
            "message": "Virtual device activity generated.",
            "event": event.model_dump(mode="json"),
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# ==================================================
# ANALYZE VIRTUAL DEVICE ACTIVITY USING XGBOOST
# ==================================================

@router.post("/{device_id}/analyze")
def analyze_device_activity(device_id: str):

    try:

        # Import the existing predictor and dataset
        # from the main application.
        from backend.main import (
            predictor,
            test_dataset,
        )

        # Find the requested virtual device.
        device = device_engine.get_device(device_id)

        if device is None:

            raise HTTPException(
                status_code=404,
                detail="Virtual device not found.",
            )

        if device.status.value == "offline":

            raise HTTPException(
                status_code=400,
                detail="Start the virtual device before requesting ML analysis.",
            )

        # Retrieve the latest activity event.
        event = device_engine.get_latest_event_for_device(
            device_id
        )

        if event is None:

            raise HTTPException(
                status_code=400,
                detail="Generate device activity before requesting ML analysis.",
            )

        # Select a genuine record from the held-out test dataset.
        row = test_dataset.sample(
            n=1
        ).iloc[0]

        # Extract the genuine 38 model features.
        feature_data = {
            column: float(row[column])
            for column in predictor.feature_columns
        }

        # Obtain the original dataset classification.
        actual_target = int(row["Target"])

        actual_class = (
            "MALICIOUS"
            if actual_target == 1
            else "BENIGN"
        )

        original_label = str(
            row["Original_Label"]
        )

        # Run the existing XGBoost predictor.
        prediction = predictor.predict_one(
            feature_data
        )

        predicted_class = prediction["prediction"]

        malicious_probability = float(
            prediction["malicious_probability"]
        )

        # Compare model output with the original dataset label.
        matches_dataset = (
            actual_class == predicted_class
        )

        # Construct the analysis result.
        analysis = {

            "model": "XGBoost",

            "prediction": predicted_class,

            "malicious_probability": malicious_probability,

            "threshold": float(
                predictor.threshold
            ),

            "actual_class": actual_class,

            "original_label": original_label,

            "matches_dataset": matches_dataset,

            "sample_source": "CICIoT2023 held-out test dataset",

            "analysis_type": "dataset_record_inference",

            "note": (
                "The model analyzed a genuine CICIoT2023 "
                "dataset record. This prediction does not "
                "represent direct inference from the simulated "
                "device activity."
            ),
        }

        # Attach analysis to the original device event.
        updated_event = device_engine.attach_analysis(
            event.event_id,
            analysis,
        )

        if updated_event is None:

            raise HTTPException(
                status_code=404,
                detail="Device event could not be updated.",
            )

        return {

            "success": True,

            "message": "XGBoost analysis completed successfully.",

            "device": {
                "device_id": device.device_id,
                "device_name": device.name,
                "device_type": device.device_type.value,
                "ip_address": device.ip_address,
            },

            "device_activity": {
                "event_id": updated_event.event_id,
                "activity": updated_event.activity,
                "scenario": updated_event.scenario,
                "timestamp": updated_event.timestamp.isoformat(),
            },

            "analysis": analysis,

            "event": updated_event.model_dump(
                mode="json"
            ),
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==================================================
# GET RECENT DEVICE EVENTS
# ==================================================

@router.get("/events")
def get_device_events(
    limit: int = Query(default=50, ge=1, le=500),
):

    events = device_engine.get_recent_events(limit)

    return {
        "success": True,
        "total_events": len(events),
        "events": [
            event.model_dump(mode="json")
            for event in events
        ],
    }