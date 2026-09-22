# -*- coding: utf-8 -*-
"""Let a resume tolerate optimizer state the checkpoint legitimately lacks.

THE ASYMMETRY. ``torch.distributed.checkpoint.state_dict._init_optim_state``
returns early when ``optim.state`` is already populated::

    def _init_optim_state(optim):
        if optim.state:
            return          # <-- the two sides of the round-trip diverge here
        ...                 # otherwise: zero grads + one step, materialising
                            # state for EVERY parameter

So the save and the load disagree whenever some parameter never receives a
gradient:

* **save**, after training: the parameters that *do* train have already put
  entries in ``optim.state``, so the early return fires and the gradient-less
  ones are written with no state at all;
* **load**, into a fresh optimizer: ``optim.state`` is empty, the full
  initialisation runs, state is materialised for *every* parameter, and the
  load planner then demands keys the checkpoint never held::

      RuntimeError: Missing key in checkpoint state_dict:
        optimizer.state.model.layers.3.downup_attn.attn.up_W_k.weight.step

WHERE IT BITES. PowerDelta with ``downup_untie_level0="ascent_kv"`` gives
``up_W_k``/``up_W_v`` a level-0 copy (``_l0_weight``) and keeps the original
``nn.Linear.weight`` for ascent levels >= 1. At hierarchy depth K=1 no such
level exists, so the originals are never read, never trained, and never get
optimizer state -- 2 tensors per DownUp layer. With N=4 and
``downup_min_bottom=128`` that is every run at seq_len 256, 512 and 1024,
which therefore could not resume at all.

WHY SKIPPING IS CORRECT, NOT A WORKAROUND. Zero moments and step 0 is exactly
the right state for a tensor that has never been updated. The live optimizer
has just initialised precisely that, so leaving it in place reproduces what a
never-interrupted run would have held at the same step.

WHAT IS STILL FATAL. Only ``optimizer.state.*`` keys may be absent. A missing
``model.*`` key means the checkpoint does not describe the model being resumed
-- a real architecture mismatch -- and is re-raised. Every skipped key is
logged, so a silent resume onto the wrong optimizer shape is not possible.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torchtitan.tools.logging import logger

_PREFIX = "optimizer.state."


class AllowMissingOptimizerState(DefaultLoadPlanner):
    """``DefaultLoadPlanner`` that skips absent optimizer state and only that."""

    def __init__(self) -> None:
        super().__init__(allow_partial_load=True)

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False) -> None:
        super().set_up_planner(state_dict, metadata, is_coordinator)
        if metadata is None:
            return

        missing = sorted(
            fqn for fqn in self.state_dict if fqn not in metadata.state_dict_metadata
        )
        if not missing:
            return

        unexpected = [fqn for fqn in missing if not fqn.startswith(_PREFIX)]
        if unexpected:
            raise RuntimeError(
                f"Checkpoint is missing {len(unexpected)} non-optimizer key(s), so it "
                f"does not match the model being resumed. First few: {unexpected[:5]}"
            )

        if is_coordinator:
            params = sorted({fqn[len(_PREFIX):].rsplit(".", 1)[0] for fqn in missing})
            logger.warning(
                "Checkpoint holds no optimizer state for %d parameter(s); they resume "
                "with zero moments, which is correct for parameters that never "
                "received a gradient. Affected: %s",
                len(params),
                ", ".join(params),
            )


def install() -> None:
    """Make ``dcp.load`` default to the tolerant planner. Idempotent.

    Patches the module-level function rather than torchtitan's call site, so it
    covers ``CheckpointManager.load`` without editing the vendored copy. A
    caller that passes its own ``planner`` is left untouched.
    """
    if getattr(dcp.load, "_flame_allows_missing_optimizer_state", False):
        return

    _original_load = dcp.load

    def load(*args, **kwargs):
        if kwargs.get("planner") is None:
            kwargs["planner"] = AllowMissingOptimizerState()
        return _original_load(*args, **kwargs)

    load._flame_allows_missing_optimizer_state = True
    load.__wrapped__ = _original_load
    dcp.load = load
