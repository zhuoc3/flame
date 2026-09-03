"""Small distributed control-flow collectives used by the trainer."""

from __future__ import annotations

import torch
import torch.distributed as dist


def synchronize_stop_request(local_request: bool) -> bool:
    """Return whether any training rank requested a coordinated stop.

    A stop decision controls later checkpoint collectives and loop exit, so it
    must be identical on every rank.  NCCL cannot reduce CPU tensors; use the
    current CUDA device for NCCL and CPU for host-capable backends such as
    Gloo.
    """

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return bool(local_request)

    backend = str(dist.get_backend()).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    request = torch.tensor(int(bool(local_request)), dtype=torch.int32, device=device)
    dist.all_reduce(request, op=dist.ReduceOp.MAX)
    return bool(request.item())
