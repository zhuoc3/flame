import tempfile
import unittest
import warnings
from io import BytesIO
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from flame.components.checkpoint import (
    FIXED_VALIDATION_STATE_KEY,
    PARALLEL_TOPOLOGY_STATE_KEY,
    FixedValidationPlanState,
    ParallelTopology,
    ParallelTopologyState,
    inspect_checkpoint_topology,
)


def _hsdp_4x2() -> ParallelTopology:
    return ParallelTopology(world_size=8, dp_replicate=4, dp_shard=2)


def _hsdp_4x4() -> ParallelTopology:
    return ParallelTopology(world_size=16, dp_replicate=4, dp_shard=4)


class CheckpointTopologyTest(unittest.TestCase):
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
