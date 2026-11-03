import pytest
from experiments.mount_contract import mount_contract, require_mount_contract


def test_legacy_short_timeout_or_other_driver_cannot_unlock_training():
    expected = mount_contract('driver source A')
    require_mount_contract({'status': 'completed', 'mount_contract': expected}, expected)
    for receipt in ({'status': 'completed'},
                    {'status': 'completed', 'mount_contract': {**expected, 'timeout_ms': 120000}},
                    {'status': 'completed', 'mount_contract': mount_contract('driver source B')},
                    {'status': 'failed', 'mount_contract': expected}):
        with pytest.raises(ValueError):
            require_mount_contract(receipt, expected)
