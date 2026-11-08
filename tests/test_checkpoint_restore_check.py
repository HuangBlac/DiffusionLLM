"""Restore actual tiny Qwen/Adam/RNG checkpoints without performing audit updates."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.checkpoint_restore_check import verify
from experiments.full_parameter_pilot import ConditionalModel, save_checkpoint, update
import experiments.full_parameter_pilot as training


@pytest.fixture
def checkpoint(tmp_path, monkeypatch, request):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(11)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, tie_word_embeddings=True,
                        eos_token_id=0, bos_token_id=0, pad_token_id=0,
                        attention_dropout=0., attn_implementation='sdpa')
    config_dir = tmp_path / 'config'
    config.save_pretrained(config_dir)
    method = getattr(request, 'param', 'ar')
    model = ConditionalModel(Qwen2ForCausalLM(config), method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .95),
                                  weight_decay=.01, foreach=False)
    order, masks = random.Random(11), torch.Generator().manual_seed(1011)
    rows = [{'ids': [2, 3, 4, 5, 6, 7, 0], 'prefix_length': 2}]
    history = [update(model, optimizer, rows, 0, order, masks, 1, 1, .001)]
    history[0]['step'] = 1
    identity = {'method': method, 'lr': .001, 'torch': str(torch.__version__),
                'transformers': transformers.__version__,
                'script_sha256': hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest()}
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, 1, order, masks, identity, history)
    return path, config_dir, identity


@pytest.mark.parametrize('checkpoint', ['ar', 'dlm'], indirect=True)
def test_actual_restore_inference_and_rng_do_not_modify_checkpoint(checkpoint):
    path, config_dir, identity = checkpoint
    before = path.read_bytes()
    result = verify(path, config_dir, identity)
    assert result['restored'] is True
    assert result['step'] == 1
    assert result['rng_state_restored'] is True
    assert result['inference_finite'] is True
    assert result['optimizer_state_count'] > 0
    assert result['training_performed'] is False
    assert result['gpu_step_replay_tested'] is False
    assert path.read_bytes() == before


def test_actual_restore_from_independent_cpu_process(checkpoint, tmp_path):
    path, config_dir, identity = checkpoint
    identity_path, output = tmp_path / 'identity.json', tmp_path / 'audit.json'
    identity_path.write_text(json.dumps(identity), encoding='utf-8')
    # Simulate the cloud CPU-only environment on developer machines with a GPU.
    bootstrap = ("import torch,runpy; torch.cuda.is_available=lambda: False; "
                 "runpy.run_module('experiments.checkpoint_restore_check', run_name='__main__')")
    completed = subprocess.run(
        [sys.executable, '-s', '-c', bootstrap,
         '--checkpoint', str(path), '--config-dir', str(config_dir),
         '--identity', str(identity_path), '--output', str(output)],
        # Match the production audit bound; a full-suite cold import exceeded 120s.
        capture_output=True, text=True, timeout=600,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['restored'] and result['inference_finite']
    assert result['optimizer_state_count'] > 0


def test_wrong_checkpoint_identity_fails(checkpoint):
    path, config_dir, identity = checkpoint
    with pytest.raises(ValueError, match='checkpoint identity differs'):
        verify(path, config_dir, {**identity, 'lr': .5})


@pytest.mark.parametrize('fault, message', [
    ('shape', 'Adam state shape differs'),
    ('step', 'Adam update count differs'),
    ('fractional_step', 'Adam update count differs'),
    ('empty', 'Adam state coverage'),
    ('missing', 'Adam state coverage'),
])
def test_invalid_adam_state_fails(checkpoint, fault, message):
    path, config_dir, identity = checkpoint
    state = torch.load(path, weights_only=True)
    saved = state['optimizer']['state']
    first = next(iter(saved))
    if fault == 'shape':
        saved[first]['exp_avg'] = torch.zeros(1)
    elif fault == 'step':
        saved[first]['step'] = torch.tensor(0.)
    elif fault == 'fractional_step':
        saved[first]['step'] = torch.tensor(1.5)
    elif fault == 'empty':
        saved.clear()
    else:
        del saved[first]
    torch.save(state, path)
    with pytest.raises(ValueError, match=message):
        verify(path, config_dir, identity)


@pytest.mark.parametrize('key,value', [('exp_avg', float('nan')),
                                      ('exp_avg_sq', float('inf'))])
def test_nonfinite_adam_moments_fail(checkpoint, key, value):
    path, config_dir, identity = checkpoint
    state = torch.load(path, weights_only=True)
    moments = next(iter(state['optimizer']['state'].values()))
    moments[key].view(-1)[0] = value
    torch.save(state, path)
    with pytest.raises(ValueError, match='Adam state contains non-finite values'):
        verify(path, config_dir, identity)


def test_verified_history_prefix_is_checked_exactly(checkpoint):
    path, config_dir, identity = checkpoint
    state = torch.load(path, weights_only=True)
    history = state['history'] + [{'step': 2}]
    result = verify(path, config_dir, identity, history)
    assert result['history_reference_matches'] is True
    history[0] = {**history[0], 'objective': history[0]['objective'] + 1}
    with pytest.raises(ValueError, match='history differs from verified training report'):
        verify(path, config_dir, identity, history)
