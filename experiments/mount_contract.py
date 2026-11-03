"""Bind transport validation to the mount settings actually used by training."""
import hashlib

MOUNT_TIMEOUT_MS = 600000


def mount_contract(driver_source: str) -> dict:
    return {'schema': 1, 'timeout_ms': MOUNT_TIMEOUT_MS,
            'driver_sha256': hashlib.sha256(driver_source.encode()).hexdigest()}


def require_mount_contract(receipt: dict, expected: dict) -> None:
    if receipt.get('status') != 'completed' or receipt.get('mount_contract') != expected:
        raise ValueError('CPU receipt does not verify this Drive mount policy/driver')
