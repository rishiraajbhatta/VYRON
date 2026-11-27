from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    CCTV_CAMERA = "cctv_camera"
    DOOR_LOCK = "door_lock"
    TEMPERATURE_SENSOR = "temperature_sensor"
    SMART_LIGHT = "smart_light"
    IOT_GATEWAY = "iot_gateway"


class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    SIMULATING = "simulating"


class VirtualDevice(BaseModel):
    device_id: str
    name: str
    device_type: DeviceType
    description: str
    ip_address: str
    status: DeviceStatus = DeviceStatus.OFFLINE
    location: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    last_activity: Optional[datetime] = None


class DeviceActivityRequest(BaseModel):
    scenario: str = Field(
        default="normal",
        description="Activity scenario: normal or suspicious"
    )


class DeviceEvent(BaseModel):
    event_id: int
    device_id: str
    device_name: str
    device_type: DeviceType

    activity: str
    scenario: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    status: str = "generated"

    # ML analysis will be attached when the event is analyzed.
    analysis: Optional[dict[str, Any]] = None