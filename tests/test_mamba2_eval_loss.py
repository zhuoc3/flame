import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Mamba2EvalLossTest(unittest.TestCase):
    def test_eval_does_not_reuse_training_fused_linear_loss(self) -> None:
        source = (ROOT / "fla/models/mamba2/modeling_mamba2.py").read_text()
        start = source.index("        if labels is not None:")
        end = source.index("\n        if not return_dict:", start)
        loss_branch = source[start:end]

        self.assertIn("if fuse_linear_and_cross_entropy:", loss_branch)
        self.assertIn("elif self.config.fuse_cross_entropy:", loss_branch)
        self.assertIn("criterion = FusedCrossEntropyLoss", loss_branch)
        eval_branch = loss_branch.split(
            "elif self.config.fuse_cross_entropy:", 1
        )[1].split("else:", 1)[0]
        self.assertNotIn("self.criterion", eval_branch)
