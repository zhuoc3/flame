"""Small helpers for controlling distributed gradient accumulation."""


def controls_gradient_sync(parallel_dims, ga_steps: int) -> bool:
    """Whether this topology needs explicit accumulation sync control."""
    return ga_steps > 1 and (
        parallel_dims.dp_replicate_enabled or parallel_dims.dp_shard_enabled
    )
