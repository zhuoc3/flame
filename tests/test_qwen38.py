import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from flame.models.qwen38 import (
    _fused_causal_lm_forward,
    _install_safe_decay_initialization,
)


class RecordingCriterion:
    ignore_index = -100

    def __init__(self) -> None:
        self.labels = None

    def __call__(self, hidden, labels, _weight):
        self.labels = labels.detach().clone()
        return hidden.float().sum()


class DummyBase(torch.nn.Module):
    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
        hidden = inputs_embeds
        if hidden is None:
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class DummyCausalLM:
    def __init__(self) -> None:
        self.model = DummyBase()
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        self.criterion = RecordingCriterion()


class Qwen38ForwardTest(unittest.TestCase):
    def test_real_model_post_init_keeps_decay_parameters_finite(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        _install_safe_decay_initialization()
        config = AutoConfig.from_pretrained(
            Path(__file__).resolve().parents[2]
            / "configs/baseline_qwen38_proportional_762m.json"
        )
        config.num_hidden_layers = 1
        config.layer_types = ["linear_attention"]
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
            model.apply(lambda module: setattr(module, "_is_hf_initialized", False))
        model.to_empty(device="cpu")
        model.post_init()
        layer = model.model.layers[0].linear_attn
        self.assertTrue(torch.isfinite(layer.A_log).all())
        self.assertGreaterEqual(layer.A_log.exp().min().item(), 0.01)
        torch.testing.assert_close(layer.dt_bias, torch.ones_like(layer.dt_bias))

    def test_training_shifts_labels_and_does_not_materialize_logits(self) -> None:
        model = DummyCausalLM()
        labels = torch.tensor([[1, 2, 3, 4]])
        output = _fused_causal_lm_forward(
            model,
            input_ids=torch.tensor([[0, 1, 2, 3]]),
            labels=labels,
            cu_seqlens=torch.tensor([0, 4]),
        )
        self.assertIsNone(output.logits)
        self.assertIsNotNone(output.loss)
        torch.testing.assert_close(
            model.criterion.labels,
            torch.tensor([[2, 3, 4, model.criterion.ignore_index]]),
        )

    def test_inference_returns_requested_tail_logits(self) -> None:
        model = DummyCausalLM()
        output = _fused_causal_lm_forward(
            model,
            input_ids=torch.tensor([[0, 1, 2, 3]]),
            logits_to_keep=2,
        )
        self.assertEqual(output.logits.shape, (1, 2, 7))


if __name__ == "__main__":
    unittest.main()
