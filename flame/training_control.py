"""Durable stop and completion helpers for the Flame training entry point."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from pathlib import Path


TRAINING_DONE_FORMAT_VERSION = 1


def partial_stop_reason(
    *,
    stop_request_file: str | None,
    slurm_end_time: int,
    slurm_time_limit_buffer_s: int,
    now: float | None = None,
) -> str | None:
    """Return why training should checkpoint and stop at the current boundary."""

    if stop_request_file:
        try:
            request_stat = os.stat(stop_request_file, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect FLAME_STOP_REQUEST_FILE={stop_request_file!r}"
            ) from exc
        else:
            if not stat.S_ISREG(request_stat.st_mode):
                raise RuntimeError(
                    "FLAME_STOP_REQUEST_FILE must name a regular file: "
                    f"{stop_request_file}"
                )
            return f"stop request file present: {stop_request_file}"

    current_time = time.time() if now is None else now
    if (
        slurm_end_time > 0
        and current_time > slurm_end_time - slurm_time_limit_buffer_s
    ):
        return f"SLURM time limit within {slurm_time_limit_buffer_s}s"
    return None


def should_run_terminal_validation(
    *,
    training_completed: bool,
    validation_enabled: bool,
    last_validated_step: int | None,
    current_step: int,
) -> bool:
    """Whether the terminal step still needs its validation pass."""

    return (
        training_completed
        and validation_enabled
        and last_validated_step != current_step
    )


def publish_training_done(
    dump_folder: str | os.PathLike[str],
    *,
    step: int,
    effective_max_steps: int,
    final_validation_step: int | None,
    fixed_test_completed: bool,
    completed_at_unix: float | None = None,
) -> dict[str, object]:
    """Atomically publish a structured marker after durable terminal work."""

    if step < effective_max_steps:
        raise ValueError(
            f"cannot complete at step {step} before target {effective_max_steps}"
        )
    if final_validation_step is not None and final_validation_step != step:
        raise ValueError(
            "final_validation_step must equal the completed training step"
        )

    root = Path(dump_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"training dump folder does not exist: {root}")
    marker = root / "TRAINING_DONE"
    payload: dict[str, object] = {
        "format_version": TRAINING_DONE_FORMAT_VERSION,
        "status": "complete",
        "step": int(step),
        "effective_max_steps": int(effective_max_steps),
        "final_validation_step": final_validation_step,
        "fixed_test_completed": bool(fixed_test_completed),
        "completed_at_unix": (
            time.time() if completed_at_unix is None else float(completed_at_unix)
        ),
    }

    descriptor, temporary_name = tempfile.mkstemp(prefix=".TRAINING_DONE.", dir=root)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return payload
