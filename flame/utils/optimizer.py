# -*- coding: utf-8 -*-
"""Optimizer builder that honours ``_no_weight_decay``.

torchtitan's ``build_optimizers`` hardcodes ``weight_decay=0.1`` and builds a
single param group per model part (``components/optimizer.py``). That is correct
for the Llama-style models torchtitan ships, which have no parameters that should
be exempt. ``fla``'s SSM layers do: ``gated_deltanet``, ``gated_deltaproduct``,
``mamba2`` and ``samba`` all mark ``A_log`` / ``dt_bias`` / ``D`` with
``_no_weight_decay = True``, and so do ``powerssm`` and ``powerdelta``. Nothing
in flame or torchtitan ever reads that flag, so those parameters are decayed.

Measured on PowerDelta with *exactly zero* gradient, 10 AdamW steps at lr 3e-4 /
wd 0.1: ``A_log`` moves by 3.000e-04, matching the pure-decay prediction of
3.000e-04 to four significant figures. Over a 29,200-step cosine schedule an
unopposed parameter retains 0.619 of its value, which compresses the per-head
decay-rate spread ``A = exp(A_log)`` from 1600x to 96x -- i.e. it pushes every
head toward one uniform time constant, which is precisely what the flag exists to
prevent.

This wrapper moves flagged parameters into a second group with
``weight_decay=0.0``. It deliberately does NOT exempt anything else: norm gains
carry no flag, and there is a reasonable stability argument for decaying them
(Qwen's zero-centered RMSNorm is immune anyway, since its neutral value is 0).

The regrouping happens inside the builder, before ``build_lr_schedulers`` runs, so
``LambdaLR`` replicates its schedule across both groups. Doing it afterwards would
leave the second group pinned at its initial LR with no warmup or decay.
"""

from __future__ import annotations

from torchtitan.components.optimizer import build_optimizers as _build_optimizers
from torchtitan.tools.logging import logger


def build_optimizers_honor_no_weight_decay(*args, **kwargs):
    """``torchtitan.build_optimizers``, plus a ``weight_decay=0`` group."""
    container = _build_optimizers(*args, **kwargs)
    total = 0
    for optimizer in container.optimizers:
        group = optimizer.param_groups[0]
        exempt = [p for p in group["params"] if getattr(p, "_no_weight_decay", False)]
        if not exempt:
            continue
        exempt_ids = {id(p) for p in exempt}
        group["params"] = [p for p in group["params"] if id(p) not in exempt_ids]
        new_group = {k: v for k, v in group.items() if k != "params"}
        new_group["params"] = exempt
        new_group["weight_decay"] = 0.0
        optimizer.add_param_group(new_group)
        total += len(exempt)
    logger.info(
        f"weight decay exempted for {total} parameters carrying _no_weight_decay "
        f"(A_log / dt_bias / D); all others keep weight_decay=0.1"
    )
    return container
