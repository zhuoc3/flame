"""Fail-hard validation for production Weights & Biases logging."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def require_online_wandb(metric_backend, *, save_for_all_ranks: bool) -> dict[str, object]:
    """Collectively require every intended W&B logger to be initialized online."""

    from torchtitan.components.metrics import WandBLogger

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expected_loggers = world_size if save_for_all_ranks else 1
    run = (
        getattr(metric_backend.wandb, "run", None)
        if isinstance(metric_backend, WandBLogger)
        else None
    )
    local_valid = bool(
        isinstance(metric_backend, WandBLogger)
        and run is not None
        and os.environ.get("WANDB_MODE") == "online"
    )
    count = torch.tensor(int(local_valid), dtype=torch.int32)
    if dist.is_initialized() and world_size > 1:
        backend = str(dist.get_backend()).lower()
        if backend == "nccl":
            count = count.to(torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    active_loggers = int(count.item())
    if active_loggers != expected_loggers:
        raise RuntimeError(
            "Required online W&B initialization failed collectively: "
            f"active={active_loggers}, expected={expected_loggers}, "
            f"WANDB_MODE={os.environ.get('WANDB_MODE')!r}"
        )
    return {
        "active_loggers": active_loggers,
        "expected_loggers": expected_loggers,
        "run_id": getattr(run, "id", None),
        "run_url": getattr(run, "url", None),
    }
