from types import SimpleNamespace
import pytest

from experiments.checkpoint_cadence import CheckpointCadence


class Writer:
    def __init__(self):
        self.local, self.remote = [], []
        self.store = SimpleNamespace(check_time=lambda: None)
        self.last_server_receipt = None

    def original_save(self, path, model, optimizer, step, *rest):
        self.local.append(step)

    def __call__(self, path, model, optimizer, step, *rest):
        self.original_save(path, model, optimizer, step, *rest)
        self.remote.append(step)
        self.last_server_receipt = {'step': step}


def save(cadence, step):
    cadence(None, None, None, step, None, None, {'method': 'ar'}, [])


def test_full_continuation_keeps_local_saves_and_final_server_save():
    writer = Writer()
    cadence = CheckpointCadence(writer, 40, 100)
    for step in range(60, 201, 20):
        save(cadence, step)
    save(cadence, 200)
    assert writer.local == list(range(60, 201, 20)) + [200]
    assert writer.remote == [100, 200, 200]
    assert cadence.last_server_receipt == {'step': 200}


@pytest.mark.parametrize('steps,expected', [([60, 80, 80], [80]), ([60, 73], [73]), ([40], [40])])
def test_partial_or_no_update_final_state_is_still_uploaded(steps, expected):
    writer = Writer()
    cadence = CheckpointCadence(writer, 40, 100)
    for step in steps:
        save(cadence, step)
    assert writer.remote == expected


def test_cadence_rejects_off_grid_and_reversed_steps():
    with pytest.raises(ValueError):
        CheckpointCadence(Writer(), 40, 50)
    cadence = CheckpointCadence(Writer(), 40, 100)
    with pytest.raises(ValueError):
        save(cadence, 20)


def test_failed_cloud_publication_keeps_last_acknowledged_receipt_and_step():
    class FailingWriter(Writer):
        fail = False

        def __call__(self, path, model, optimizer, step, *rest):
            if self.fail:
                self.original_save(path, model, optimizer, step, *rest)
                raise RuntimeError('Injected cloud failure')
            super().__call__(path, model, optimizer, step, *rest)

    writer = FailingWriter()
    cadence = CheckpointCadence(writer, 40, 100)
    for step in (60, 80, 100, 120, 140, 160, 180):
        save(cadence, step)
    writer.fail = True
    with pytest.raises(RuntimeError, match='cloud failure'):
        save(cadence, 200)
    assert cadence.last_step == 180
    assert cadence.last_server_receipt == {'step': 100}
    writer.fail = False
    save(cadence, 200)
    assert cadence.last_step == 200
    assert cadence.last_server_receipt == {'step': 200}


def test_expired_storage_deadline_stops_local_save_before_mutation():
    writer = Writer()

    def expired():
        raise TimeoutError('No storage budget')

    writer.store.check_time = expired
    cadence = CheckpointCadence(writer, 40, 100)
    with pytest.raises(TimeoutError):
        save(cadence, 60)
    assert writer.local == writer.remote == []
    assert cadence.last_step == 40


def test_every20_configuration_retains_all_cloud_saves():
    writer = Writer()
    cadence = CheckpointCadence(writer, 40, 20)
    for step in (60, 80, 100, 100):
        save(cadence, step)
    assert writer.local == writer.remote == [60, 80, 100, 100]


def test_off_grid_resume_still_detects_off_grid_final_save():
    writer = Writer()
    cadence = CheckpointCadence(writer, 41, 100)
    for step in (60, 61):
        save(cadence, step)
    assert writer.local == [60, 61]
    assert writer.remote == [61]
