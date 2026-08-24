import os


def pending_job_cancellation_enabled() -> bool:
    """Keep legacy chain cleanup unless a launcher explicitly opts out."""

    value = os.environ.get("FLAME_CANCEL_PENDING_ON_COMPLETE", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}
