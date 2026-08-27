"""Durable stop and completion helpers for the Flame training entry point."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path


TRAINING_DONE_FORMAT_VERSION = 1


def resolve_test_stop_after_step(
    raw_value: str | None,
    *,
    allow_test_max_steps: str | None,
    effective_max_steps: int,
) -> int | None:
    """Validate the deterministic partial-stop hook used by lifecycle tests."""

    if raw_value is None:
        return None
    if allow_test_max_steps != "1":
        raise RuntimeError(
            "FLAME_TEST_STOP_AFTER_STEP is only allowed when "
            "QWEN38_ALLOW_TEST_MAX_STEPS=1"
        )
    if re.fullmatch(r"[0-9]+", raw_value) is None:
        raise ValueError("FLAME_TEST_STOP_AFTER_STEP must be a positive integer")
    stop_after_step = int(raw_value)
    if stop_after_step <= 0:
        raise ValueError("FLAME_TEST_STOP_AFTER_STEP must be a positive integer")
    if stop_after_step > effective_max_steps:
        raise ValueError(
            "FLAME_TEST_STOP_AFTER_STEP cannot exceed the effective maximum step"
        )
    return stop_after_step


def partial_stop_reason(
    *,
    stop_request_file: str | None,
    slurm_end_time: int,
    slurm_time_limit_buffer_s: int,
    test_stop_after_step: int | None = None,
    current_step: int | None = None,
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

    if test_stop_after_step is not None:
        if current_step is None:
            raise ValueError("current_step is required for a deterministic test stop")
        if current_step >= test_stop_after_step:
            return f"deterministic test stop after completed step {test_stop_after_step}"

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
    marker_name: str = "TRAINING_DONE",
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
    if (
        not marker_name
        or marker_name in {".", ".."}
        or Path(marker_name).name != marker_name
        or re.fullmatch(r"[A-Za-z0-9_.-]+", marker_name) is None
    ):
        raise ValueError("completion marker name must be a safe basename")
    marker = root / marker_name
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

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker_name}.", dir=root
    )
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
