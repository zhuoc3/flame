"""Fail-hard, model-only historical checkpoint archives."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from torchtitan.tools.logging import logger


def model_archive_is_due(step: int, every_steps: int) -> bool:
    """Return whether ``step`` is an exact positive archive cadence boundary."""

    return every_steps > 0 and step > 0 and step % every_steps == 0


def save_model_only_archive(
    checkpoint: Any,
    dump_folder: str,
    step: int,
) -> None:
    """Synchronously save one non-resumable bf16 model snapshot.

    A complete archive is immutable and can be reused after a retry. An
    incomplete directory is removed by rank zero before all ranks retry the
    collective save. Errors deliberately propagate: callers that requested an
    archive must not advance the authoritative resumable checkpoint past a
    missing historical snapshot.
    """

    import torch.distributed.checkpoint as dcp
    from torchtitan.components.checkpoint import MODEL

    archive_folder = Path(dump_folder) / "archive" / f"step-{step}"
    metadata_path = archive_folder / ".metadata"
    rank = torch.distributed.get_rank()

    # Publish rank-zero filesystem preparation to every rank. This avoids a
    # collective hang if cleanup itself fails on the shared filesystem.
    preparation = [False, None]
    if rank == 0:
        try:
            if metadata_path.is_file():
                preparation[0] = True
            elif archive_folder.exists():
                shutil.rmtree(archive_folder)
        except Exception as error:
            preparation[1] = repr(error)
    broadcast_device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.distributed.get_backend() == "nccl"
        else torch.device("cpu")
    )
    torch.distributed.broadcast_object_list(
        preparation,
        src=0,
        device=broadcast_device,
    )
    archive_complete, preparation_error = preparation
    if preparation_error is not None:
        raise RuntimeError(
            f"Could not prepare model-only archive {archive_folder}: "
            f"{preparation_error}"
        )
    if archive_complete:
        if rank == 0:
            logger.info(f"Model-only bf16 archive already complete at {archive_folder}")
        return

    model_state = checkpoint.states[MODEL].state_dict()
    model_state.pop("freqs_cis", None)
    model_state = {
        key: (value.to(torch.bfloat16) if value.is_floating_point() else value)
        for key, value in model_state.items()
    }
    dcp.save(model_state, checkpoint_id=str(archive_folder))
    if rank == 0:
        logger.info(f"Saved model-only bf16 archive at {archive_folder}")
