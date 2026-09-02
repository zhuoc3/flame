from collections.abc import Iterable, Sequence

import torch


def count_causal_lm_targets(labels: torch.Tensor, ignore_index: int = -100) -> int:
    """Count labels that remain after the model's internal causal shift."""
    if labels.ndim < 1:
        raise ValueError("causal language-model labels must have at least one dimension")
    return int(labels[..., 1:].ne(ignore_index).sum().item())


def count_causal_lm_targets_from_valid_lengths(
    valid_lengths: Iterable[int], loss_start_position: int
) -> int:
    """Count post-shift, post-prefix-mask targets from sequence lengths."""

    if loss_start_position < 0:
        raise ValueError("loss_start_position must be non-negative")
    first_target_position = max(1, loss_start_position)
    return sum(
        max(0, int(valid_length) - first_target_position)
        for valid_length in valid_lengths
    )


def causal_lm_loss_scales(
    local_target_counts: Sequence[int],
    *,
    global_target_count: int,
    gradient_average_group_size: int,
) -> tuple[float, ...]:
    """Scale local mean losses into one global target-token mean.

    DDP and FSDP average gradients across their data-parallel group. Multiplying
    each microbatch mean by ``group_size * local_count / global_count`` makes
    the subsequent gradient average equal the sum over all target-token losses
    divided by their global count. Summing the scaled losses locally and taking
    the same group mean also produces the exact loss value for logging.
    """
    if gradient_average_group_size <= 0:
        raise ValueError("gradient_average_group_size must be positive")
    if global_target_count <= 0:
        raise ValueError("global_target_count must be positive")
    if any(count < 0 for count in local_target_counts):
        raise ValueError("local target counts must be non-negative")
    return tuple(
        gradient_average_group_size * count / global_target_count
        for count in local_target_counts
    )
