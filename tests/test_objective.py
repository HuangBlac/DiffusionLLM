import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.small_qwen import Denoiser, LoRALinear, target_states


def test_alignment_uses_hidden_before_masked_target_without_label_leak():
    states = torch.arange(4.0).reshape(1, 4, 1)
    clean = torch.tensor([[0, 11, 22, 33]])
    mask = torch.tensor([[False, False, True, False]])
    shifted, labels = target_states(states, clean, mask, "shift")
    same, _ = target_states(states, clean, mask, "same")
    assert shifted.item() == 1 and same.item() == 2 and labels.item() == 22


def test_zero_initialized_lora_preserves_base_and_updates():
    layer = LoRALinear(nn.Linear(3, 4).requires_grad_(False))
    x = torch.randn(2, 3)
    assert torch.equal(layer(x), layer.base(x))
    layer(x).sum().backward()
    assert layer.b.grad.abs().sum() > 0
    assert layer.base.weight.grad is None


def test_masked_target_content_is_not_visible_and_attention_changes_behavior():
    torch.manual_seed(7)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attn_implementation="sdpa"))
    model = Denoiser(base).eval()
    clean = torch.tensor([[0, 1, 2, 3]])
    changed = torch.tensor([[0, 1, 2, 9]])
    mask = torch.tensor([[False, False, False, True]])
    with torch.no_grad():
        assert torch.equal(model.hidden(clean, mask), model.hidden(changed, mask))
        assert torch.allclose(model.hidden(clean, causal=True)[:, 0], model.hidden(changed, causal=True)[:, 0], atol=1e-6)
        assert not torch.allclose(model.hidden(clean)[:, 0], model.hidden(changed)[:, 0], atol=1e-6)


def test_only_adapters_and_virtual_mask_train():
    model = Denoiser(Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert "mask_embedding" in trainable
    assert all(name == "mask_embedding" or name.endswith((".a", ".b")) for name in trainable)
