import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torchtitan.components.checkpoint import MODEL

from flame.model_archive import model_archive_is_due, save_model_only_archive


class ModelArchiveTest(unittest.TestCase):
    def test_exact_cadence_is_independent_of_other_intervals(self) -> None:
        due = [step for step in range(15002) if model_archive_is_due(step, 5000)]
        self.assertEqual(due, [5000, 10000, 15000])
        self.assertFalse(model_archive_is_due(52307, 5000))
        self.assertFalse(model_archive_is_due(5000, 0))

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_backend", return_value="gloo")
    @patch("torch.distributed.get_rank", return_value=0)
    def test_archive_is_flat_model_only_bf16(
        self, _get_rank, _get_backend, _broadcast
    ) -> None:
        state = {
            "weight": torch.ones(2, dtype=torch.float32),
            "counter": torch.ones(1, dtype=torch.int64),
            "freqs_cis": torch.ones(1),
        }
        checkpoint = SimpleNamespace(
            states={MODEL: SimpleNamespace(state_dict=lambda: dict(state))}
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.distributed.checkpoint.save") as save:
                save_model_only_archive(checkpoint, directory, 5000)
        saved_state = save.call_args.args[0]
        self.assertEqual(set(saved_state), {"weight", "counter"})
        self.assertEqual(saved_state["weight"].dtype, torch.bfloat16)
        self.assertEqual(saved_state["counter"].dtype, torch.int64)

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_backend", return_value="gloo")
    @patch("torch.distributed.get_rank", return_value=0)
    def test_archive_save_failure_propagates(
        self, _get_rank, _get_backend, _broadcast
    ) -> None:
        checkpoint = SimpleNamespace(
            states={
                MODEL: SimpleNamespace(
                    state_dict=lambda: {"weight": torch.ones(1)}
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "torch.distributed.checkpoint.save",
                side_effect=RuntimeError("injected archive failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected archive failure"):
                    save_model_only_archive(checkpoint, directory, 5000)

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_backend", return_value="gloo")
    @patch("torch.distributed.get_rank", return_value=0)
    def test_complete_archive_is_immutable_and_reused(
        self, _get_rank, _get_backend, _broadcast
    ) -> None:
        checkpoint = SimpleNamespace(states={})
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive" / "step-5000"
            archive.mkdir(parents=True)
            (archive / ".metadata").touch()
            with patch("torch.distributed.checkpoint.save") as save:
                save_model_only_archive(checkpoint, directory, 5000)
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
