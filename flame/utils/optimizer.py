# -*- coding: utf-8 -*-
"""Optimizer builder that exempts scale parameters from weight decay.

torchtitan's ``build_optimizers`` hardcodes ``weight_decay=0.1`` and builds a
single param group per model part (``components/optimizer.py``). That is correct
for the Llama-style models torchtitan ships, which have no parameters that should
be exempt. The models trained here do.

Exemption rule: a parameter is exempt if it carries ``_no_weight_decay`` **or**
has ``ndim < 2``. The union matters, because each half catches something the
other misses.

``_no_weight_decay`` alone is not enough
---------------------------------------
``fla``'s SSM layers mark ``A_log`` / ``dt_bias`` / ``D`` with the flag, and so
do ``powerssm`` and ``powerdelta`` -- but the flag is carried by only 504 of
PowerDelta-762M's parameters, while 34,944 more are norm *gains* that carry no
flag and are decayed away from their neutral value. PowerDelta has three norm
classes and they do not agree on what neutral means:

===============================  =========  =======  =====================
class                            instances  neutral  decay pulls
===============================  =========  =======  =====================
``PowerDeltaRMSNorm``                   57        0  toward neutral (safe)
``PowerDeltaRMSNormGated``              21        1  away from neutral
``PowerFormerRMSNorm`` (DownUp)         77        1  away from neutral
===============================  =========  =======  =====================

Only the first is zero-centered (``weight`` inits to zeros, applied as
``1 + weight``); the other two init to ones and multiply directly. An earlier
version of this module decayed all norm gains on the grounds that "Qwen's
zero-centered RMSNorm is immune anyway" -- true of the first row, not of the
other 98 instances.

``ndim < 2`` alone would be enough today, but the union is kept so a future
module that flags a matrix-shaped parameter is still honoured.

Why it matters, measured rather than argued
-------------------------------------------
On the live PowerDelta m3n4 762M run at step ~5,000 of 41,775, every
multiplicative gain tracked the pure-decay curve to within a few percent -- the
DownUp path norms to within 1% -- i.e. weight decay was essentially unopposed.

For most of them that is harmless: a norm gain followed by a Linear is
redundant (``W diag(g) norm(x)``), and the following Linears had grown to
2.5-3.2x their init std, leaving the product 2.4-2.9x its initial scale. The
exception is the **QK norms**, which are applied immediately before the
attention kernel with no learnable scale after them, so ``g_q * g_k`` *is* the
attention logit scale and nothing can absorb it. The descent path had already
lost 25% of its logit scale at 12% of training. Left alone for the full
schedule, an unopposed gain retains 0.503, which would put the DownUp attention
logits at roughly a quarter of their initial scale -- a much warmer softmax
than the architecture was designed and initialised for.

Exempting costs 0.014% of the model its (weak) regulariser. All of it is scale
parameters, none of it capacity.

Implementation note
-------------------
The regrouping happens inside the builder, before ``build_lr_schedulers`` runs,
so ``LambdaLR`` replicates its schedule across both groups. Doing it afterwards
would leave the second group pinned at its initial LR with no warmup or decay.
"""

from __future__ import annotations

from torchtitan.components.optimizer import build_optimizers as _build_optimizers
from torchtitan.tools.logging import logger


def _is_exempt(parameter) -> bool:
    """Scale parameters: flagged SSM constants, and every 1-D tensor.

    1-D covers all norm gains and any bias, which is the standard rule (decay
    matrices, not vectors). It needs no per-parameter bookkeeping, so a new
    module cannot silently miss the exemption -- which is exactly how the
    ``_no_weight_decay`` flag came to be dead metadata.
    """
    return getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2


def build_optimizers_honor_no_weight_decay(*args, **kwargs):
    """``torchtitan.build_optimizers``, plus a ``weight_decay=0`` group."""
    container = _build_optimizers(*args, **kwargs)
    exempt_total = flagged_total = decayed_total = 0
    for optimizer in container.optimizers:
        group = optimizer.param_groups[0]
        exempt = [p for p in group["params"] if _is_exempt(p)]
        decayed_total += len(group["params"]) - len(exempt)
        if not exempt:
            continue
        flagged_total += sum(
            1 for p in exempt if getattr(p, "_no_weight_decay", False)
        )
        exempt_ids = {id(p) for p in exempt}
        group["params"] = [p for p in group["params"] if id(p) not in exempt_ids]
        new_group = {k: v for k, v in group.items() if k != "params"}
        new_group["params"] = exempt
        new_group["weight_decay"] = 0.0
        optimizer.add_param_group(new_group)
        exempt_total += len(exempt)
    logger.info(
        f"weight decay exempted for {exempt_total} parameter tensors "
        f"({flagged_total} carrying _no_weight_decay, the rest 1-D scale "
        f"parameters); {decayed_total} tensors keep weight_decay=0.1"
    )
    return container
