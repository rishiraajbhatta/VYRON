from typing import Dict, List, Optional

from .device_models import (
    DeviceStatus,
    DeviceType,
    VirtualDevice,
)


class DeviceRegistry:
    """
    Maintains the virtual IoT devices registered in VYRON.
    """

    def __init__(self):
        self.devices: Dict[str, VirtualDevice] = {}
        self._register_default_devices()

    def _register_default_devices(self):
        default_devices = [
            VirtualDevice(
                device_id="VYRON-CCTV-001",
                name="Smart CCTV Camera",
                device_type=DeviceType.CCTV_CAMERA,
                description="Simulates a smart surveillance camera.",
                ip_address="192.168.1.101",
                location="Main Entrance",
            ),
            VirtualDevice(
                device_id="VYRON-LOCK-001",
                name="Smart Door Lock",
                device_type=DeviceType.DOOR_LOCK,
                description="Simulates a network-connected smart door lock.",
                ip_address="192.168.1.102",
                location="Main Entrance",
            ),
            VirtualDevice(
                device_id="VYRON-TEMP-001",
                name="Temperature Sensor",
                device_type=DeviceType.TEMPERATURE_SENSOR,
                description="Simulates an IoT temperature monitoring sensor.",
                ip_address="192.168.1.103",
                location="Server Room",
            ),
            VirtualDevice(
                device_id="VYRON-LIGHT-001",
                name="Smart Light",
                device_type=DeviceType.SMART_LIGHT,
                description="Simulates a remotely controlled smart light.",
                ip_address="192.168.1.104",
                location="Living Room",
            ),
            VirtualDevice(
                device_id="VYRON-GATEWAY-001",
                name="IoT Gateway",
                device_type=DeviceType.IOT_GATEWAY,
                description="Simulates a gateway connecting IoT devices.",
                ip_address="192.168.1.105",
                location="Network Room",
            ),
        ]

        for device in default_devices:
            self.devices[device.device_id] = device

    def get_all_devices(self) -> List[VirtualDevice]:
        return list(self.devices.values())

    def get_device(self, device_id: str) -> Optional[VirtualDevice]:
        return self.devices.get(device_id)

    def update_device_status(
        self,
        device_id: str,
        status: DeviceStatus,
    ) -> Optional[VirtualDevice]:
        device = self.get_device(device_id)

        if device is None:
            return None

        device.status = status
        return device

    def get_online_devices(self) -> List[VirtualDevice]:
        return [
            device
            for device in self.devices.values()
            if device.status != DeviceStatus.OFFLINE
        ]

    def get_device_count(self) -> int:
        return len(self.devices)