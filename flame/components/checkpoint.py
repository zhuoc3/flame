# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful


PARALLEL_TOPOLOGY_STATE_KEY = "parallel_topology"
PARALLEL_TOPOLOGY_SCHEMA_VERSION = 1
FIXED_VALIDATION_STATE_KEY = "fixed_validation_plan"


def _scalar_int(value: Any) -> int:
    """Convert a scalar DCP value to an int without assuming its container."""

    return int(value.item()) if hasattr(value, "item") else int(value)


@dataclass(frozen=True)
class ParallelTopology:
    """Rank-independent description of the mesh that wrote a checkpoint."""

    world_size: int
    dp_replicate: int
    dp_shard: int
    cp: int = 1
    tp: int = 1
    pp: int = 1

    def __post_init__(self) -> None:
        dimensions = {
            "world_size": self.world_size,
            "dp_replicate": self.dp_replicate,
            "dp_shard": self.dp_shard,
            "cp": self.cp,
            "tp": self.tp,
            "pp": self.pp,
        }
        invalid = {name: value for name, value in dimensions.items() if value < 1}
        if invalid:
            raise ValueError(f"Parallel topology dimensions must be positive: {invalid}")

        mesh_size = self.dp_replicate * self.dp_shard * self.cp * self.tp * self.pp
        if mesh_size != self.world_size:
            raise ValueError(
                "Parallel topology dimensions do not match world size: "
                f"{self.dp_replicate} * {self.dp_shard} * {self.cp} * "
                f"{self.tp} * {self.pp} != {self.world_size}"
            )

    @property
    def dp_degree(self) -> int:
        return self.dp_replicate * self.dp_shard

    @classmethod
    def from_parallel_dims(cls, parallel_dims: Any) -> "ParallelTopology":
        """Build from TorchTitan ``ParallelDims`` without importing it here."""

        return cls(
            world_size=int(parallel_dims.world_size),
            dp_replicate=int(parallel_dims.dp_replicate),
            dp_shard=int(parallel_dims.dp_shard),
            cp=int(parallel_dims.cp),
            tp=int(parallel_dims.tp),
            pp=int(parallel_dims.pp),
        )

    def changed_dimensions(self, other: "ParallelTopology") -> tuple[str, ...]:
        names = ("world_size", "dp_replicate", "dp_shard", "cp", "tp", "pp")
        return tuple(name for name in names if getattr(self, name) != getattr(other, name))

    def as_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "schema_version": torch.tensor(
                PARALLEL_TOPOLOGY_SCHEMA_VERSION, dtype=torch.int32
            ),
            "world_size": torch.tensor(self.world_size, dtype=torch.int32),
            "dp_replicate": torch.tensor(self.dp_replicate, dtype=torch.int32),
            "dp_shard": torch.tensor(self.dp_shard, dtype=torch.int32),
            "cp": torch.tensor(self.cp, dtype=torch.int32),
            "tp": torch.tensor(self.tp, dtype=torch.int32),
            "pp": torch.tensor(self.pp, dtype=torch.int32),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> "ParallelTopology":
        schema_version = _scalar_int(state_dict["schema_version"])
        if schema_version != PARALLEL_TOPOLOGY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported parallel topology checkpoint schema: "
                f"{schema_version} (expected {PARALLEL_TOPOLOGY_SCHEMA_VERSION})"
            )
        return cls(
            world_size=_scalar_int(state_dict["world_size"]),
            dp_replicate=_scalar_int(state_dict["dp_replicate"]),
            dp_shard=_scalar_int(state_dict["dp_shard"]),
            cp=_scalar_int(state_dict["cp"]),
            tp=_scalar_int(state_dict["tp"]),
            pp=_scalar_int(state_dict["pp"]),
        )


class ParallelTopologyState(Stateful):
    """DCP state that detects, but does not undo, the current mesh topology.

    Loading records the topology that wrote the checkpoint while keeping
    ``current`` unchanged. Consequently, a checkpoint saved after a reshaped
    resume describes the new mesh rather than perpetuating stale metadata.
    """

    def __init__(self, current: ParallelTopology) -> None:
        self.current = current
        self.loaded: Optional[ParallelTopology] = None

    @property
    def topology_changed(self) -> bool:
        return self.loaded is not None and self.loaded != self.current

    @property
    def changed_dimensions(self) -> tuple[str, ...]:
        if self.loaded is None:
            return ()
        return self.loaded.changed_dimensions(self.current)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.current.as_state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.loaded = ParallelTopology.from_state_dict(state_dict)


class FixedValidationPlanState(Stateful):
    """Cursor-free checkpoint binding for an immutable validation token set."""

    def __init__(self, current: Mapping[str, Any]) -> None:
        self.current = dict(current)
        self.loaded: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, BytesIO]:
        payload = json.dumps(self.current, sort_keys=True).encode("utf-8")
        return {"plan": BytesIO(payload)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        payload = state_dict["plan"]
        payload.seek(0)
        loaded = json.loads(payload.read().decode("utf-8"))
        self.loaded = loaded
        if loaded != self.current:
            raise ValueError(
                "Fixed validation plan changed across resume: "
                f"saved={loaded}, current={self.current}"
            )


@dataclass(frozen=True)
class TopologyResumeDecision:
    """Result of inspecting topology metadata before the full DCP load."""

    current: ParallelTopology
    saved: Optional[ParallelTopology]
    legacy_data_parallel_degree: Optional[int] = None

    @property
    def metadata_found(self) -> bool:
        return self.saved is not None

    @property
    def topology_changed(self) -> bool:
        return self.saved is not None and self.saved != self.current

    @property
    def changed_dimensions(self) -> tuple[str, ...]:
        if self.saved is None:
            return ()
        return self.saved.changed_dimensions(self.current)

    @property
    def reconstruct_dataloader(self) -> bool:
        # A legacy checkpoint has no trustworthy rank-layout metadata. The
        # deterministic loader can recover exactly from train_state.step, so
        # rebuilding is safer than attempting to load rank-local state.
        return self.saved is None or self.topology_changed


def inspect_checkpoint_topology(
    checkpoint_id: str | Path,
    current: ParallelTopology,
    state_key: str = PARALLEL_TOPOLOGY_STATE_KEY,
) -> TopologyResumeDecision:
    """Read only topology state before loading model/optimizer/dataloader.

    DCP model and optimizer state reshard automatically, but rank-local loader
    state cannot be selected safely until the checkpoint's old topology is
    known. This small preliminary load supplies that decision. Checkpoints
    created before this metadata was introduced return ``saved=None``.
    """

    checkpoint_id = Path(checkpoint_id)
    if not checkpoint_id.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_id}")

    metadata = dcp.FileSystemReader(str(checkpoint_id)).read_metadata()
    prefix = f"{state_key}."
    if not any(key.startswith(prefix) for key in metadata.state_dict_metadata):
        # Old ParallelAwareDataLoader checkpoints contain one opaque rank_N
        # object per DP rank. This cannot recover HSDP factors, but it can
        # prove whether the DP degree itself changed and prevent a silent,
        # incorrect rank-local loader restore.
        legacy_prefix = "dataloader.rank_"
        legacy_ranks = set()
        for key in metadata.state_dict_metadata:
            if key.startswith(legacy_prefix):
                suffix = key[len(legacy_prefix) :]
                if suffix.isdigit():
                    legacy_ranks.add(int(suffix))
        legacy_dp_degree = None
        if legacy_ranks:
            candidate = max(legacy_ranks) + 1
            if legacy_ranks == set(range(candidate)):
                legacy_dp_degree = candidate
        return TopologyResumeDecision(
            current=current,
            saved=None,
            legacy_data_parallel_degree=legacy_dp_degree,
        )

    topology_state = ParallelTopologyState(current)
    dcp.load({state_key: topology_state}, checkpoint_id=str(checkpoint_id))
    if topology_state.loaded is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_id} contains {state_key!r} metadata but it was not loaded"
        )
    return TopologyResumeDecision(current=current, saved=topology_state.loaded)


@dataclass
class TrainState(Stateful):
    step: int = 0
    skipped_step: int = 0
    token: int = 0
    elapsed: timedelta = timedelta(0)
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "skipped_step": torch.tensor(self.skipped_step, dtype=torch.int32),
            "token": torch.tensor(self.token, dtype=torch.int64),
            "elapsed": self.elapsed,
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.skipped_step = state_dict.get("skipped_step", 0).item()
        self.token = state_dict["token"].item()
        self.elapsed = state_dict["elapsed"]
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)
