import unittest
from types import SimpleNamespace

import torch

from flame.models.qwen38 import _fused_causal_lm_forward


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
