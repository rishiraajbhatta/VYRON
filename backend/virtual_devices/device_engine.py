from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from .device_models import (
    DeviceEvent,
    DeviceStatus,
    DeviceType,
)
from .device_registry import DeviceRegistry


class VirtualDeviceEngine:
    """
    Controls virtual IoT device activity and records simulation events.

    Device activity is simulated.
    ML analysis is performed separately using genuine
    CICIoT2023 held-out test dataset records.
    """

    def __init__(self, registry: Optional[DeviceRegistry] = None):

        self.registry = registry or DeviceRegistry()

        self.events = deque(maxlen=500)
        self.event_counter = 0

        self.activity_patterns: Dict[DeviceType, Dict[str, List[str]]] = {

            DeviceType.CCTV_CAMERA: {
                "normal": [
                    "Streaming surveillance footage",
                    "Sending periodic camera status updates",
                    "Transmitting video metadata",
                ],
                "suspicious": [
                    "Unusual outbound communication detected in simulation",
                    "Abnormal communication frequency simulated",
                    "Unexpected connection attempt simulated",
                ],
            },

            DeviceType.DOOR_LOCK: {
                "normal": [
                    "Sending door lock status",
                    "Processing an authentication request",
                    "Synchronising access control information",
                ],
                "suspicious": [
                    "Repeated authentication requests simulated",
                    "Unusual access communication simulated",
                    "Unexpected remote access attempt simulated",
                ],
            },

            DeviceType.TEMPERATURE_SENSOR: {
                "normal": [
                    "Sending temperature telemetry",
                    "Reporting periodic sensor readings",
                    "Synchronising sensor status",
                ],
                "suspicious": [
                    "Abnormal telemetry frequency simulated",
                    "Unexpected communication burst simulated",
                    "Unusual outbound data transfer simulated",
                ],
            },

            DeviceType.SMART_LIGHT: {
                "normal": [
                    "Sending light status update",
                    "Receiving a lighting control command",
                    "Synchronising device configuration",
                ],
                "suspicious": [
                    "Repeated control requests simulated",
                    "Unexpected remote command simulated",
                    "Unusual device communication simulated",
                ],
            },

            DeviceType.IOT_GATEWAY: {
                "normal": [
                    "Forwarding IoT device telemetry",
                    "Synchronising connected device status",
                    "Processing routine network communication",
                ],
                "suspicious": [
                    "Unusual traffic forwarding simulated",
                    "Abnormal connection activity simulated",
                    "Unexpected communication volume simulated",
                ],
            },
        }

    # ==================================================
    # DEVICE RETRIEVAL
    # ==================================================

    def get_all_devices(self):

        return self.registry.get_all_devices()

    def get_device(self, device_id: str):

        return self.registry.get_device(device_id)

    # ==================================================
    # START DEVICE
    # ==================================================

    def start_device(self, device_id: str):

        device = self.registry.get_device(device_id)

        if device is None:
            raise ValueError(
                f"Device '{device_id}' was not found."
            )

        if device.status == DeviceStatus.SIMULATING:
            return device

        self.registry.update_device_status(
            device_id,
            DeviceStatus.SIMULATING,
        )

        return device

    # ==================================================
    # STOP DEVICE
    # ==================================================

    def stop_device(self, device_id: str):

        device = self.registry.get_device(device_id)

        if device is None:
            raise ValueError(
                f"Device '{device_id}' was not found."
            )

        self.registry.update_device_status(
            device_id,
            DeviceStatus.OFFLINE,
        )

        return device

    # ==================================================
    # SIMULATE DEVICE ACTIVITY
    # ==================================================

    def simulate_activity(
        self,
        device_id: str,
        scenario: str = "normal",
    ) -> DeviceEvent:

        device = self.registry.get_device(device_id)

        if device is None:
            raise ValueError(
                f"Device '{device_id}' was not found."
            )

        if device.status == DeviceStatus.OFFLINE:
            raise ValueError(
                f"Device '{device.name}' is offline. "
                "Start it before simulating activity."
            )

        if scenario not in ("normal", "suspicious"):
            raise ValueError(
                "Invalid scenario. Choose 'normal' or 'suspicious'."
            )

        import random

        available_activities = self.activity_patterns[
            device.device_type
        ][scenario]

        activity = random.choice(available_activities)

        self.event_counter += 1

        timestamp = datetime.now(timezone.utc)

        event = DeviceEvent(
            event_id=self.event_counter,
            device_id=device.device_id,
            device_name=device.name,
            device_type=device.device_type,
            activity=activity,
            scenario=scenario,
            timestamp=timestamp,
            status="generated",
            analysis=None,
        )

        device.last_activity = timestamp

        self.events.appendleft(event)

        return event

    # ==================================================
    # GET LATEST EVENT FOR A DEVICE
    # ==================================================

    def get_latest_event_for_device(
        self,
        device_id: str,
    ) -> Optional[DeviceEvent]:

        for event in self.events:

            if event.device_id == device_id:
                return event

        return None

    # ==================================================
    # ATTACH ML ANALYSIS TO AN EVENT
    # ==================================================

    def attach_analysis(
        self,
        event_id: int,
        analysis: Dict[str, Any],
    ) -> Optional[DeviceEvent]:

        for event in self.events:

            if event.event_id == event_id:

                event.analysis = analysis

                return event

        return None

    # ==================================================
    # GET RECENT EVENTS
    # ==================================================

    def get_recent_events(self, limit: int = 50):

        limit = max(1, min(limit, 500))

        return list(self.events)[:limit]

    # ==================================================
    # DEVICE STATISTICS
    # ==================================================

    def get_device_status(self):

        devices = self.registry.get_all_devices()

        return {
            "total_devices": len(devices),

            "online_devices": sum(
                device.status != DeviceStatus.OFFLINE
                for device in devices
            ),

            "simulating_devices": sum(
                device.status == DeviceStatus.SIMULATING
                for device in devices
            ),

            "offline_devices": sum(
                device.status == DeviceStatus.OFFLINE
                for device in devices
            ),
        }