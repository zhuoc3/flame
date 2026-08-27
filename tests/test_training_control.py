import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flame.training_control import (
    partial_stop_reason,
    publish_training_done,
    resolve_test_stop_after_step,
    should_run_terminal_validation,
)


class TrainingControlTest(unittest.TestCase):
    def test_stop_request_precedes_walltime_and_requires_a_regular_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = Path(temp_dir) / "stop"
            request.write_text("handoff\n", encoding="utf-8")
            reason = partial_stop_reason(
                stop_request_file=str(request),
                slurm_end_time=10_000,
                slurm_time_limit_buffer_s=100,
                now=1,
            )
            self.assertEqual(reason, f"stop request file present: {request}")

            request.unlink()
            request.mkdir()
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                partial_stop_reason(
                    stop_request_file=str(request),
                    slurm_end_time=0,
                    slurm_time_limit_buffer_s=100,
                )

    def test_walltime_stop_boundary(self):
        self.assertIsNone(
            partial_stop_reason(
                stop_request_file=None,
                slurm_end_time=1_000,
                slurm_time_limit_buffer_s=100,
                now=900,
            )
        )
        self.assertEqual(
            partial_stop_reason(
                stop_request_file=None,
                slurm_end_time=1_000,
                slurm_time_limit_buffer_s=100,
                now=901,
            ),
            "SLURM time limit within 100s",
        )

    def test_test_stop_is_strictly_gated_and_validated(self):
        self.assertIsNone(
            resolve_test_stop_after_step(
                None, allow_test_max_steps=None, effective_max_steps=55
            )
        )
        with self.assertRaisesRegex(RuntimeError, "only allowed"):
            resolve_test_stop_after_step(
                "51", allow_test_max_steps=None, effective_max_steps=55
            )
        with self.assertRaisesRegex(RuntimeError, "only allowed"):
            resolve_test_stop_after_step(
                "51", allow_test_max_steps="0", effective_max_steps=55
            )
        for invalid in ("", "0", "-1", "+1", "1.0", " 1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resolve_test_stop_after_step(
                        invalid,
                        allow_test_max_steps="1",
                        effective_max_steps=55,
                    )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            resolve_test_stop_after_step(
                "56", allow_test_max_steps="1", effective_max_steps=55
            )
        self.assertEqual(
            resolve_test_stop_after_step(
                "51", allow_test_max_steps="1", effective_max_steps=55
            ),
            51,
        )

    def test_test_stop_fires_at_exact_completed_optimizer_step(self):
        arguments = {
            "stop_request_file": None,
            "slurm_end_time": 0,
            "slurm_time_limit_buffer_s": 100,
            "test_stop_after_step": 51,
        }
        self.assertIsNone(partial_stop_reason(current_step=50, **arguments))
        expected = "deterministic test stop after completed step 51"
        self.assertEqual(
            partial_stop_reason(current_step=51, **arguments), expected
        )
        self.assertEqual(
            partial_stop_reason(current_step=52, **arguments), expected
        )

    def test_terminal_validation_runs_after_resume_at_max_but_not_twice(self):
        self.assertTrue(
            should_run_terminal_validation(
                training_completed=True,
                validation_enabled=True,
                last_validated_step=None,
                current_step=55,
            )
        )
        self.assertFalse(
            should_run_terminal_validation(
                training_completed=True,
                validation_enabled=True,
                last_validated_step=55,
                current_step=55,
            )
        )
        self.assertFalse(
            should_run_terminal_validation(
                training_completed=False,
                validation_enabled=True,
                last_validated_step=None,
                current_step=55,
            )
        )

    def test_training_done_is_structured_and_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "TRAINING_DONE"
            marker.write_text("legacy partial", encoding="utf-8")
            payload = publish_training_done(
                temp_dir,
                step=55,
                effective_max_steps=55,
                final_validation_step=55,
                fixed_test_completed=True,
                completed_at_unix=1234.5,
            )

            self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), payload)
            self.assertEqual(payload["format_version"], 1)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["step"], 55)
            self.assertEqual(payload["final_validation_step"], 55)
            self.assertTrue(payload["fixed_test_completed"])
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(temp_dir).glob(".TRAINING_DONE.*")), [])

    def test_training_done_can_use_an_explicit_private_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            private = Path(temp_dir) / ".TRAINING_DONE.pending-audit"
            payload = publish_training_done(
                temp_dir,
                step=55,
                effective_max_steps=55,
                final_validation_step=55,
                fixed_test_completed=False,
                marker_name=private.name,
                completed_at_unix=1234.5,
            )

            self.assertEqual(json.loads(private.read_text(encoding="utf-8")), payload)
            self.assertFalse((Path(temp_dir) / "TRAINING_DONE").exists())
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                list(Path(temp_dir).glob("..TRAINING_DONE.pending-audit.*")), []
            )

    def test_training_done_rejects_unsafe_marker_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for marker_name in ("", ".", "..", "../TRAINING_DONE", "a/b", "bad name"):
                with self.subTest(marker_name=marker_name):
                    with self.assertRaisesRegex(ValueError, "safe basename"):
                        publish_training_done(
                            temp_dir,
                            step=55,
                            effective_max_steps=55,
                            final_validation_step=55,
                            fixed_test_completed=False,
                            marker_name=marker_name,
                        )

    def test_training_done_failure_never_replaces_existing_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "TRAINING_DONE"
            marker.write_text("old\n", encoding="utf-8")
            with mock.patch(
                "flame.training_control.os.replace", side_effect=OSError("boom")
            ):
                with self.assertRaisesRegex(OSError, "boom"):
                    publish_training_done(
                        temp_dir,
                        step=55,
                        effective_max_steps=55,
                        final_validation_step=None,
                        fixed_test_completed=False,
                    )

            self.assertEqual(marker.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(Path(temp_dir).glob(".TRAINING_DONE.*")), [])

    def test_training_done_rejects_incomplete_or_mismatched_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "before target"):
                publish_training_done(
                    temp_dir,
                    step=54,
                    effective_max_steps=55,
                    final_validation_step=None,
                    fixed_test_completed=False,
                )
            with self.assertRaisesRegex(ValueError, "must equal"):
                publish_training_done(
                    temp_dir,
                    step=55,
                    effective_max_steps=55,
                    final_validation_step=50,
                    fixed_test_completed=False,
                )

    def test_training_entrypoint_orders_terminal_work_before_completion(self):
        source = (Path(__file__).resolve().parents[1] / "flame/train.py").read_text()
        loop = source.index("while train_state.step < _effective_max_steps:")
        stop_boundary = source.index("stop_reason = partial_stop_reason(", loop)
        periodic_validation = source.index("# Periodic validation", stop_boundary)
        forced_save = source.index("checkpoint.save(", periodic_validation)
        post_save_check = source.index("post_save_stop_reason = partial_stop_reason(", forced_save)
        post_save_rollover = source.index("if time_limit_triggered:", post_save_check)
        archive = source.index("# Model-only archive snapshot", post_save_rollover)
        completion_state = source.index("training_completed = (", forced_save)
        terminal_validation = source.index("if should_run_terminal_validation(")
        fixed_test = source.index("if training_completed and test_dataloader is not None:")
        checkpoint_close = source.index("checkpoint.close()", fixed_test)
        completion = source.index("publish_training_done(", checkpoint_close)
        checkpoint_load = source.index("checkpoint_loaded = checkpoint.load(")
        restored_rng = source.index("restored_random_state = (", checkpoint_load)
        loaded_proof = source.index(
            "checkpoint.save_loaded_test_checkpoint_proof(", restored_rng
        )
        finite_check = source.index(
            "if qwen38_runtime_metadata is not None and checkpoint_loaded:",
            loaded_proof,
        )
        self.assertLess(stop_boundary, periodic_validation)
        self.assertIn(
            "test_stop_after_step=test_stop_after_step",
            source[stop_boundary:periodic_validation],
        )
        self.assertIn(
            "current_step=train_state.step",
            source[stop_boundary:periodic_validation],
        )
        self.assertIn(
            "if stop_reason is not None:\n                time_limit_triggered = True",
            source[stop_boundary:periodic_validation],
        )
        self.assertLess(periodic_validation, forced_save)
        self.assertIn(
            "or time_limit_triggered",
            source[forced_save : source.index("\n            )", forced_save)],
        )
        self.assertLess(forced_save, post_save_check)
        self.assertLess(post_save_check, post_save_rollover)
        self.assertLess(post_save_rollover, archive)
        self.assertIn(
            "time_limit_triggered = True",
            source[post_save_check:post_save_rollover],
        )
        self.assertIn(
            "not time_limit_triggered",
            source[completion_state:terminal_validation],
        )
        self.assertLess(terminal_validation, fixed_test)
        self.assertLess(fixed_test, checkpoint_close)
        self.assertLess(checkpoint_close, completion)
        self.assertLess(checkpoint_load, restored_rng)
        self.assertLess(restored_rng, loaded_proof)
        self.assertLess(loaded_proof, finite_check)


if __name__ == "__main__":
    unittest.main()
