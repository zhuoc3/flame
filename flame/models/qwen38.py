"""Production hooks for the proportional Qwen3.8 text baseline.

The official Transformers implementation materializes full vocabulary logits
before computing causal-LM loss.  At 16K context that tensor is unnecessarily
large.  This module keeps the audited Qwen3.8 architecture intact while using
the same accelerated kernels and fused linear cross entropy as the standalone
GPU probe.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from types import MethodType
from typing import Any

import torch


MODEL_TYPE = "qwen3_5_text"
EXPECTED_PARAMETERS = 761_812_856
EXPECTED_FLA_VERSION = "0.5.2"
EXPECTED_TRANSFORMERS_VERSION = "5.15.1"
EXPECTED_RUNTIME_ROOT = Path(
    "/orcd/scratch/orcd/010/zianshi/venvs/qwen38-probe"
).resolve()


def is_qwen38_config(config: Any) -> bool:
    return getattr(config, "model_type", None) == MODEL_TYPE


def _install_safe_decay_initialization() -> None:
    """Use upstream's positive lower bound for DeltaNet decay parameters."""

    from torch.nn import init
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5PreTrainedModel,
    )

    current = Qwen3_5PreTrainedModel._init_weights
    if getattr(current, "_flame_qwen38_safe_decay", False):
        return

    def safe_init_weights(self, module):
        current(self, module)
        if isinstance(module, Qwen3_5GatedDeltaNet):
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log).uniform_(0.01, 16).log_(),
            )
            init.constant_(module.dt_bias, 1.0)

    safe_init_weights._flame_qwen38_safe_decay = True
    Qwen3_5PreTrainedModel._init_weights = safe_init_weights


def configure_qwen38_runtime() -> dict[str, Any]:
    """Bind Transformers' reference calls to the audited CUDA kernels."""

    import causal_conv1d
    import fla
    import transformers
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5

    actual_fla = version("flash-linear-attention")
    fla_path = Path(fla.__file__).resolve()
    if actual_fla != EXPECTED_FLA_VERSION:
        raise RuntimeError(
            f"Qwen3.8 requires flash-linear-attention {EXPECTED_FLA_VERSION}, "
            f"found {actual_fla} at {fla.__file__}"
        )
    if EXPECTED_RUNTIME_ROOT not in fla_path.parents:
        raise RuntimeError(
            "Qwen3.8 imported an unaudited flash-linear-attention source: "
            f"{fla_path}; expected it below {EXPECTED_RUNTIME_ROOT}"
        )
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Qwen3.8 requires Transformers {EXPECTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )

    def qwen_causal_conv1d_fn(
        hidden_states,
        weight,
        bias=None,
        activation=None,
        **_kwargs,
    ):
        return causal_conv1d.causal_conv1d_fn(
            hidden_states,
            weight,
            bias=bias,
            activation=activation,
        )

    modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling_qwen3_5.causal_conv1d_fn = qwen_causal_conv1d_fn
    modeling_qwen3_5.causal_conv1d_update = causal_conv1d.causal_conv1d_update
    _install_safe_decay_initialization()
    return {
        "architecture": "Qwen3.8 text / Qwen3_5ForCausalLM",
        "fla_version": actual_fla,
        "fla_path": str(fla_path),
        "transformers_version": transformers.__version__,
        "gated_delta_kernel": chunk_gated_delta_rule.__module__,
        "causal_conv_kernel": causal_conv1d.causal_conv1d_fn.__module__,
        "safe_decay_lower_bound": 0.01,
        "delta_dt_bias": 1.0,
    }


def _fused_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    **kwargs,
):
    """Qwen causal-LM forward without a full logits tensor during training."""

    from transformers.modeling_outputs import CausalLMOutputWithPast

    # FLAME passes this FLA-only argument for every architecture.
    kwargs.pop("cu_seqlens", None)
    kwargs.pop("return_dict", None)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False if use_cache is None else use_cache,
        return_dict=True,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None
    if labels is not None:
        labels = labels.to(hidden_states.device)
        shifted_labels = torch.cat(
            (
                labels[..., 1:],
                torch.full_like(labels[..., :1], self.criterion.ignore_index),
            ),
            dim=-1,
        )
        loss = self.criterion(hidden_states, shifted_labels, self.lm_head.weight)
    else:
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, indices, :])

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def prepare_qwen38_model(model: torch.nn.Module) -> dict[str, Any]:
    """Install equation-preserving training fusions and fused LM loss."""

    from fla.modules import FusedLinearCrossEntropyLoss
    from scripts.qwen38_training_fusions import apply_qwen38_training_fusions

    if not is_qwen38_config(model.config):
        raise ValueError(f"Expected {MODEL_TYPE}, found {model.config.model_type}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Qwen3.8 parameter count changed: expected {EXPECTED_PARAMETERS:,}, "
            f"found {parameter_count:,}"
        )
    if getattr(model.model, "gradient_checkpointing", False):
        raise RuntimeError("Qwen3.8 production run requires activation checkpointing off")

    fusion_metadata = apply_qwen38_training_fusions(model)
    model.criterion = FusedLinearCrossEntropyLoss(num_chunks=1)
    model.forward = MethodType(_fused_causal_lm_forward, model)
    return {
        "parameters": parameter_count,
        "loss": "FusedLinearCrossEntropyLoss(num_chunks=1)",
        "activation_checkpointing": False,
        "training_fusions": fusion_metadata,
    }


@torch.no_grad()
def assert_qwen38_model_finite(model: torch.nn.Module, phase: str) -> None:
    """Fail before costly training if initialization or restore is corrupt."""

    finite = torch.ones((), dtype=torch.bool, device=next(model.parameters()).device)
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_floating_point() and tensor.numel():
            finite.logical_and_(torch.isfinite(tensor).all())
    if not bool(finite.item()):
        raise RuntimeError(f"Qwen3.8 has non-finite model state after {phase}")
