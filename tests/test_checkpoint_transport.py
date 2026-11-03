import pytest
import torch
from experiments.checkpoint_transport import run


def test_checkpoint_restores_model_adam_and_all_random_states(tmp_path):
    run(tmp_path, 'save')
    result = run(tmp_path, 'restore')
    assert result['next_update_equal'] is True
    with pytest.raises(FileExistsError):
        run(tmp_path, 'save')


def test_corrupted_optimizer_is_detected(tmp_path):
    run(tmp_path, 'save')
    path = tmp_path / 'checkpoint.pt'
    state = torch.load(path, weights_only=True)
    state['optimizer']['state'][0]['exp_avg'].add_(1)
    torch.save(state, path)
    with pytest.raises(RuntimeError, match='differs'):
        run(tmp_path, 'restore')
