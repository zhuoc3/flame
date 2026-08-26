import random
import tempfile
import unittest
import warnings
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

from flame.components.checkpoint import (
    FIXED_TEST_STATE_KEY,
    FIXED_VALIDATION_STATE_KEY,
    PARALLEL_TOPOLOGY_STATE_KEY,
    FixedTestPlanState,
    FixedValidationPlanState,
    ParallelTopology,
    ParallelTopologyState,
    RandomState,
    TrainState,
    inspect_checkpoint_topology,
)
from flame.config_manager import JobConfig
from flame.data import iter_preserving_torch_cpu_rng


def _hsdp_4x2() -> ParallelTopology:
    return ParallelTopology(world_size=8, dp_replicate=4, dp_shard=2)


def _hsdp_4x4() -> ParallelTopology:
    return ParallelTopology(world_size=16, dp_replicate=4, dp_shard=4)


class CheckpointTopologyTest(unittest.TestCase):
    def test_dataloader_iterator_does_not_advance_training_rng(self) -> None:
        loader = torch.utils.data.DataLoader(torch.arange(8), batch_size=2)
        torch.manual_seed(123)
        expected = torch.rand(4)
        torch.manual_seed(123)
        iterator = iter_preserving_torch_cpu_rng(loader)
        actual = torch.rand(4)
        self.assertEqual(next(iterator).tolist(), [0, 1])
        torch.testing.assert_close(actual, expected)

    def test_logger_initializes_before_rng_checkpoint_restore(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "flame/train.py").read_text()
        logger_init = source.index(
            "metric_logger = build_metrics_processor(job_config, parallel_dims)"
        )
        checkpoint_load = source.index(
            "checkpoint_loaded = checkpoint.load(step=requested_load_step)"
        )
        logger_config = source.index("metric_logger.log_config(")
        self.assertLess(logger_init, checkpoint_load)
        self.assertLess(checkpoint_load, logger_config)

    def test_resume_reapplies_rng_after_iterator_reconstruction(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "flame/train.py").read_text()
        checkpoint_load = source.index(
            "checkpoint_loaded = checkpoint.load(step=requested_load_step)"
        )
        rng_snapshot = source.index(
            "restored_random_state = (", checkpoint_load
        )
        train_iterator = source.index(
            "data_iterator = iter_preserving_torch_cpu_rng(dataloader)", rng_snapshot
        )
        val_iterator = source.index("val_iterator = (", train_iterator)
        rng_restore = source.index(
            "checkpoint.states[RANDOM_STATE_KEY].load_state_dict(", val_iterator
        )
        validation = source.index("def run_validation(step):", rng_restore)
        self.assertLess(checkpoint_load, rng_snapshot)
        self.assertLess(rng_snapshot, train_iterator)
        self.assertLess(train_iterator, val_iterator)
        self.assertLess(val_iterator, rng_restore)
        self.assertLess(rng_restore, validation)

    def test_rank_local_random_state_roundtrip_and_pinned_staging(self) -> None:
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)
        state = RandomState(rank=0)
        state_dict = {"random_state": state.state_dict()}
        cpu_state = _create_cpu_state_dict(
            state_dict,
            pin_memory=False,
            share_memory=False,
        )
        copied = _copy_state_dict(state_dict, cpu_state, non_blocking=False)

        expected = (random.random(), np.random.random(), torch.rand(1))
        random.seed(117)
        np.random.seed(118)
        torch.manual_seed(119)
        state.load_state_dict(copied["random_state"])
        actual = (random.random(), np.random.random(), torch.rand(1))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])
        with self.assertRaisesRegex(RuntimeError, "rank_1"):
            RandomState(rank=1).load_state_dict(copied["random_state"])

    def test_train_state_elapsed_supports_pinned_staging_and_legacy_load(self) -> None:
        expected = timedelta(days=2, seconds=3, microseconds=456789)
        state = TrainState(step=7, elapsed=expected)
        state_dict = {"train_state": state.state_dict()}

        self.assertIsInstance(state_dict["train_state"]["elapsed"], float)
        cpu_state = _create_cpu_state_dict(
            state_dict,
            pin_memory=False,
            share_memory=False,
        )
        copied = _copy_state_dict(state_dict, cpu_state, non_blocking=False)
        restored = TrainState()
        restored.load_state_dict(copied["train_state"])
        self.assertEqual(restored.elapsed, expected)

        legacy = state.state_dict()
        legacy["elapsed"] = expected
        restored.load_state_dict(legacy)
        self.assertEqual(restored.elapsed, expected)

    def test_fixed_test_is_opt_in_and_legacy_validation_defaults_are_unchanged(
        self,
    ) -> None:
        args = JobConfig().parser.parse_args([])
        self.assertIsNone(getattr(args, "training.fixed_test_parent_blocks_dir"))
        self.assertIsNone(getattr(args, "training.fixed_val_parent_blocks_dir"))
        self.assertIsNone(getattr(args, "training.val_data_dir"))

    def test_fixed_validation_plan_roundtrip_and_mismatch(self) -> None:
        plan = {
            "schema_version": 1,
            "manifest_sha256": "manifest-a",
            "tokens_payload_sha256": "payload-a",
            "num_sequences": 960,
            "seq_len": 16_384,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_id = Path(temp_dir) / "step-10000"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.save(
                    {
                        FIXED_VALIDATION_STATE_KEY: FixedValidationPlanState(plan)
                    },
                    checkpoint_id=checkpoint_id,
                )
                matching = FixedValidationPlanState(plan)
                dcp.load(
                    {FIXED_VALIDATION_STATE_KEY: matching},
                    checkpoint_id=checkpoint_id,
                )
                changed = dict(plan, manifest_sha256="manifest-b")
                with self.assertRaisesRegex(ValueError, "changed across resume"):
                    dcp.load(
                        {
                            FIXED_VALIDATION_STATE_KEY: FixedValidationPlanState(
                                changed
                            )
                        },
                        checkpoint_id=checkpoint_id,
                    )

        self.assertEqual(matching.loaded, plan)

    def test_fixed_test_plan_roundtrip_and_mismatch(self) -> None:
        plan = {
            "schema_version": 1,
            "manifest_sha256": "test-manifest-a",
            "tokens_payload_sha256": "test-payload-a",
            "num_sequences": 960,
            "seq_len": 16_384,
        }
        self.assertNotEqual(FIXED_TEST_STATE_KEY, FIXED_VALIDATION_STATE_KEY)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_id = Path(temp_dir) / "step-10000"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.save(
                    {FIXED_TEST_STATE_KEY: FixedTestPlanState(plan)},
                    checkpoint_id=checkpoint_id,
                )
                matching = FixedTestPlanState(plan)
                dcp.load(
                    {FIXED_TEST_STATE_KEY: matching},
                    checkpoint_id=checkpoint_id,
                )
                changed = dict(plan, tokens_payload_sha256="test-payload-b")
                with self.assertRaisesRegex(
                    ValueError, "Fixed test plan changed across resume"
                ):
                    dcp.load(
                        {FIXED_TEST_STATE_KEY: FixedTestPlanState(changed)},
                        checkpoint_id=checkpoint_id,
                    )

        self.assertEqual(matching.loaded, plan)

    def test_parallel_topology_validates_mesh_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match world size"):
            ParallelTopology(world_size=16, dp_replicate=4, dp_shard=2)

    def test_loaded_state_detects_change_but_saves_current_topology(self) -> None:
        current = _hsdp_4x4()
        state = ParallelTopologyState(current)

        state.load_state_dict(_hsdp_4x2().as_state_dict())

        self.assertTrue(state.topology_changed)
        self.assertEqual(state.changed_dimensions, ("world_size", "dp_shard"))
        self.assertEqual(state.loaded, _hsdp_4x2())
        self.assertEqual(ParallelTopology.from_state_dict(state.state_dict()), current)

    def test_inspect_checkpoint_topology_detects_reshape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_id = Path(temp_dir) / "step-10000"
            saved = ParallelTopologyState(_hsdp_4x2())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.save(
                    {PARALLEL_TOPOLOGY_STATE_KEY: saved},
                    checkpoint_id=checkpoint_id,
                )
                decision = inspect_checkpoint_topology(checkpoint_id, _hsdp_4x4())

        self.assertTrue(decision.metadata_found)
        self.assertEqual(decision.saved, _hsdp_4x2())
        self.assertTrue(decision.topology_changed)
        self.assertEqual(decision.changed_dimensions, ("world_size", "dp_shard"))
        self.assertTrue(decision.reconstruct_dataloader)

    def test_inspect_legacy_checkpoint_requires_loader_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_id = Path(temp_dir) / "step-10000"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.save(
                    {"train_state": {"step": torch.tensor(10000)}},
                    checkpoint_id=checkpoint_id,
                )
                decision = inspect_checkpoint_topology(checkpoint_id, _hsdp_4x2())

        self.assertFalse(decision.metadata_found)
        self.assertIsNone(decision.saved)
        self.assertFalse(decision.topology_changed)
        self.assertTrue(decision.reconstruct_dataloader)

    def test_inspect_legacy_checkpoint_recovers_rank_local_dp_degree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_id = Path(temp_dir) / "step-10000"
            legacy_loader = {
                f"rank_{rank}": BytesIO(f"rank {rank}".encode())
                for rank in range(8)
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.save(
                    {"dataloader": legacy_loader},
                    checkpoint_id=checkpoint_id,
                )
                decision = inspect_checkpoint_topology(checkpoint_id, _hsdp_4x2())

        self.assertFalse(decision.metadata_found)
        self.assertEqual(decision.legacy_data_parallel_degree, 8)

    def test_same_topology_does_not_require_loader_reconstruction(self) -> None:
        topology = _hsdp_4x2()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_id = Path(temp_dir) / "step-10000"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.save(
                    {PARALLEL_TOPOLOGY_STATE_KEY: ParallelTopologyState(topology)},
                    checkpoint_id=checkpoint_id,
                )
                decision = inspect_checkpoint_topology(checkpoint_id, topology)

        self.assertTrue(decision.metadata_found)
        self.assertFalse(decision.topology_changed)
        self.assertFalse(decision.reconstruct_dataloader)


if __name__ == "__main__":
    unittest.main()
