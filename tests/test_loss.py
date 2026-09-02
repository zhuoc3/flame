import unittest

import torch

from flame.loss import (
    causal_lm_loss_scales,
    count_causal_lm_targets,
    count_causal_lm_targets_from_valid_lengths,
)


class TokenAveragedLossTest(unittest.TestCase):
    def test_literal_two_sequence_batch_reports_2_point_8(self) -> None:
        # The first row contributes 8,192 targets at loss 1; the padded row
        # contributes 2,048 targets at loss 10. This is one BS=2, GA=1 batch.
        target_mean = (8192 * 1.0 + 2048 * 10.0) / (8192 + 2048)
        scales = causal_lm_loss_scales(
            [8192 + 2048],
            global_target_count=8192 + 2048,
            gradient_average_group_size=1,
        )
        self.assertEqual(scales, (1.0,))
        self.assertAlmostEqual(target_mean * scales[0], 2.8)

    def test_counts_only_unmasked_post_shift_targets(self) -> None:
        labels = torch.tensor(
            [
                [-100, -100, 2, 3, -100],
                [5, 6, -100, 8, 9],
            ]
        )
        # Position zero is never a causal target, even when it is not masked.
        self.assertEqual(count_causal_lm_targets(labels), 5)

    def test_counts_valid_lengths_after_causal_shift_and_prefix_mask(self) -> None:
        valid_lengths = [16384, 10240, 8192, 17]
        self.assertEqual(
            count_causal_lm_targets_from_valid_lengths(valid_lengths, 8192),
            8192 + 2048,
        )
        self.assertEqual(
            count_causal_lm_targets_from_valid_lengths(valid_lengths, 0),
            sum(length - 1 for length in valid_lengths),
        )

    def test_dp_and_gradient_accumulation_produce_exact_token_mean(self) -> None:
        # Rank 0 sees two 4096-target microbatches at loss 1. Rank 1 sees two
        # 1024-target microbatches at loss 10. DDP/FSDP averages the two ranks.
        global_targets = 8192 + 2048
        rank0_scales = causal_lm_loss_scales(
            [4096, 4096],
            global_target_count=global_targets,
            gradient_average_group_size=2,
        )
        rank1_scales = causal_lm_loss_scales(
            [1024, 1024],
            global_target_count=global_targets,
            gradient_average_group_size=2,
        )
        rank0_scaled_loss = sum(1.0 * scale for scale in rank0_scales)
        rank1_scaled_loss = sum(10.0 * scale for scale in rank1_scales)
        ddp_averaged_loss = (rank0_scaled_loss + rank1_scaled_loss) / 2

        self.assertAlmostEqual(ddp_averaged_loss, 2.8)
        self.assertEqual(rank0_scales, (0.8, 0.8))
        self.assertEqual(rank1_scales, (0.2, 0.2))

    def test_uniform_counts_recover_inverse_accumulation_scaling(self) -> None:
        scales = causal_lm_loss_scales(
            [100, 100, 100, 100],
            global_target_count=2 * 4 * 100,
            gradient_average_group_size=2,
        )
        self.assertEqual(scales, (0.25, 0.25, 0.25, 0.25))


if __name__ == "__main__":
    unittest.main()
