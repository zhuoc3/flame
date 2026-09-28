"""Seed-time initialization of fla GatedDeltaNet's A_log and dt_bias.

fla 0.1.2's GatedDeltaNet sets A_log / dt_bias only in __init__, and its
_init_weights covers Linear, Conv1d and Embedding. flame builds the model on the
meta device and then calls to_empty + post_init, so those two parameters are
left as uninitialized memory: a 130M seed carried values up to 2e19, which gave
NaN from step 1 at one seed and a model with dead heads at another. This redoes
the layer's own __init__ formula on the materialized parameters (the same
init h_gated_deltanet carries in its _init_weights).
"""
import math

import torch
from torch.distributed.tensor import DTensor


@torch.no_grad()
def init_gated_deltanet_decay(model: torch.nn.Module) -> int:
    """Re-initialize A_log / dt_bias of every GatedDeltaNet layer; returns the count.

    DTensor parameters are skipped: they only occur after FSDP sharding, where the
    values come from the seed checkpoint that this function initialized.
    """
    count = 0
    for module in model.modules():
        if type(module).__name__ != "GatedDeltaNet":
            continue
        A_log, dt_bias = module.A_log, module.dt_bias
        if isinstance(A_log, DTensor) or isinstance(dt_bias, DTensor):
            continue
        A = torch.empty(A_log.shape, dtype=torch.float32, device=A_log.device).uniform_(0, 16)
        A_log.copy_(torch.log(A))
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(dt_bias.shape, device=dt_bias.device) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A_now = A_log.float().exp()
        dt_now = torch.nn.functional.softplus(dt_bias.float())
        if not (torch.isfinite(A_log).all() and torch.isfinite(dt_bias).all()
                and A_now.max() <= 16 and dt_now.min() >= dt_init_floor * 0.99
                and dt_now.max() <= dt_max * 1.01):
            raise RuntimeError(f"GatedDeltaNet decay init out of range: A {A_now}, dt {dt_now}")
        count += 1
    return count
