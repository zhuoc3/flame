#!/usr/bin/env python3
"""GPU integration test for FSDP-to-HSDP DCP resume.

Run the two phases from one four-GPU allocation::

    torchrun --standalone --nproc-per-node=2 \
      tests/test_distributed_elastic_checkpoint.py save CHECKPOINT_DIR
    torchrun --standalone --nproc-per-node=4 \
      tests/test_distributed_elastic_checkpoint.py load CHECKPOINT_DIR
    ELASTIC_TEST_REPLICATE=1 ELASTIC_TEST_SHARD=4 \
      torchrun --standalone --nproc-per-node=4 \
      tests/test_distributed_elastic_checkpoint.py load CHECKPOINT_DIR

The save phase writes a two-way FSDP model and Adam state after one update.
The load phases restore it into 2x2 HSDP and 1x4 FSDP meshes, compare with a
reference model trained directly on the identical logical batch, and verify
the next optimizer update and deterministic data cohort as well.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from flame.components.checkpoint import (
    FIXED_VALIDATION_STATE_KEY,
    PARALLEL_TOPOLOGY_STATE_KEY,
    FixedValidationPlanState,
    ParallelTopology,
    ParallelTopologyState,
    TrainState,
    inspect_checkpoint_topology,
)
from flame.data import MemmapTokenBlockDataset, build_dataloader
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.optimizer import OptimizersContainer


DEVICE_TYPE = os.environ.get("ELASTIC_TEST_DEVICE", "cuda")
FIXED_VALIDATION_PLAN = {
    "schema_version": 1,
    "manifest_sha256": "fixed-validation-manifest",
    "tokens_payload_sha256": "fixed-validation-payload",
    "num_sequences": 960,
    "seq_len": 16_384,
}


class _IdentityCollatorTokenizer:
    """Minimal tokenizer surface used by the language-model collator."""

    pad_token_id = 0


def _build_model(replicate: int, shard: int):
    torch.manual_seed(20260823)
    if DEVICE_TYPE == "cuda":
        torch.cuda.manual_seed_all(20260823)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.GELU(),
        # Keep every sharded dimension divisible by both test shard degrees.
        # This isolates DCP resharing from uneven/empty-shard behavior.
        nn.Linear(8, 4),
    ).to(DEVICE_TYPE)
    if replicate == 1:
        mesh = init_device_mesh(
            DEVICE_TYPE, (shard,), mesh_dim_names=("dp_shard",)
        )
    else:
        mesh = init_device_mesh(
            DEVICE_TYPE,
            (replicate, shard),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
    fully_shard(model, mesh=mesh)
    optimizers = OptimizersContainer(
        [model],
        torch.optim.AdamW,
        {"lr": 3e-3, "betas": (0.9, 0.95), "weight_decay": 0.01},
    )
    return model, optimizers


def _global_batch(update: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7000 + update)
    # Identical samples make the DP average independent of reduction-tree
    # roundoff, isolating checkpoint resharding from numerical noise.
    inputs = torch.randn(1, 4, generator=generator).repeat(4, 1)
    targets = torch.randn(1, 4, generator=generator).repeat(4, 1)
    return inputs, targets


def _update(model, optimizers, update: int) -> None:
    inputs, targets = _global_batch(update)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if inputs.shape[0] % world_size:
        raise AssertionError("Logical test batch must divide the test world size")
    local_inputs = inputs.chunk(world_size)[rank].to(DEVICE_TYPE)
    local_targets = targets.chunk(world_size)[rank].to(DEVICE_TYPE)
    optimizers.zero_grad()
    loss = torch.nn.functional.mse_loss(model(local_inputs), local_targets)
    loss.backward()
    optimizers.step()


def _local_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, DTensor):
        value = value.to_local()
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def _assert_models_close(actual: nn.Module, expected: nn.Module) -> None:
    actual_params = dict(actual.named_parameters())
    expected_params = dict(expected.named_parameters())
    assert actual_params.keys() == expected_params.keys()
    for name in actual_params:
        torch.testing.assert_close(
            _local_tensor(actual_params[name]),
            _local_tensor(expected_params[name]),
            rtol=2e-5,
            atol=2e-6,
            msg=f"model parameter mismatch: {name}",
        )


def _assert_optimizers_close(actual, expected, actual_model, expected_model) -> None:
    actual_named = dict(actual_model.named_parameters())
    expected_named = dict(expected_model.named_parameters())
    actual_optimizer = actual.optimizers[0]
    expected_optimizer = expected.optimizers[0]
    for name in actual_named:
        actual_state = actual_optimizer.state[actual_named[name]]
        expected_state = expected_optimizer.state[expected_named[name]]
        assert actual_state.keys() == expected_state.keys(), name
        for state_name in actual_state:
            torch.testing.assert_close(
                _local_tensor(actual_state[state_name]),
                _local_tensor(expected_state[state_name]),
                rtol=2e-5,
                atol=2e-6,
                msg=f"optimizer state mismatch: {name}.{state_name}",
            )


def _full_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def _logical_snapshot(model, optimizers) -> dict[str, torch.Tensor]:
    """Collect topology-independent full model and Adam tensors on every rank."""
    snapshot = {}
    named = dict(model.named_parameters())
    optimizer = optimizers.optimizers[0]
    for name in sorted(named):
        parameter = named[name]
        snapshot[f"model.{name}"] = _full_tensor(parameter)
        for state_name in sorted(optimizer.state[parameter]):
            snapshot[f"optimizer.{name}.{state_name}"] = _full_tensor(
                optimizer.state[parameter][state_name]
            )
    return snapshot


def _assert_snapshot_equal(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(
            actual[name], expected[name], rtol=0, atol=0, msg=name
        )


def _make_store(root: Path) -> MemmapTokenBlockDataset:
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        tokens = np.arange(8 * 8, dtype=np.uint16).reshape(8, 8)
        np.save(root / "tokens.npy", tokens)
        (root / "manifest.json").write_text(
            json.dumps({"seq_len": 8, "num_rows": 8, "dtype": "uint16"})
            + "\n"
        )
    dist.barrier()
    return MemmapTokenBlockDataset(root)


def _build_loader(store_root: Path, completed_steps: int):
    # parent_len=8, child_len=4, two parents/update => four samples/update.
    loader = build_dataloader(
        dataset=_make_store(store_root),
        tokenizer=_IdentityCollatorTokenizer(),
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=1,
        seq_len=4,
        gradient_accumulation_steps=4 // dist.get_world_size(),
        parent_blocks_per_step=2,
        seed=1234,
    )
    loader.set_optimizer_step(completed_steps)
    return loader


def _init_distributed() -> None:
    if DEVICE_TYPE == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend)


def _save(checkpoint_dir: Path) -> None:
    if dist.get_world_size() != 2:
        raise AssertionError("save phase requires two ranks")
    model, optimizers = _build_model(replicate=1, shard=2)
    _update(model, optimizers, update=0)
    train_state = TrainState(step=1)
    topology_state = ParallelTopologyState(
        ParallelTopology(world_size=2, dp_replicate=1, dp_shard=2)
    )
    loader = _build_loader(checkpoint_dir.parent / "store", completed_steps=1)
    states = {
        "model": ModelWrapper(model),
        "optimizer": optimizers,
        "train_state": train_state,
        PARALLEL_TOPOLOGY_STATE_KEY: topology_state,
        FIXED_VALIDATION_STATE_KEY: FixedValidationPlanState(
            FIXED_VALIDATION_PLAN
        ),
        "dataloader": loader,
    }
    snapshot = _logical_snapshot(model, optimizers)
    if dist.get_rank() == 0:
        torch.save(snapshot, checkpoint_dir.parent / "logical_snapshot.pt")
    dcp.save(states, checkpoint_id=str(checkpoint_dir))
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"saved FSDP checkpoint: {checkpoint_dir}", flush=True)


def _load(checkpoint_dir: Path) -> None:
    if dist.get_world_size() != 4:
        raise AssertionError("load phase requires four ranks")
    replicate = int(os.environ.get("ELASTIC_TEST_REPLICATE", "2"))
    shard = int(os.environ.get("ELASTIC_TEST_SHARD", "2"))
    current = ParallelTopology(
        world_size=4, dp_replicate=replicate, dp_shard=shard
    )
    decision = inspect_checkpoint_topology(checkpoint_dir, current)
    assert decision.saved == ParallelTopology(
        world_size=2, dp_replicate=1, dp_shard=2
    )
    assert decision.topology_changed

    model, optimizers = _build_model(replicate=replicate, shard=shard)
    train_state = TrainState()
    topology_state = ParallelTopologyState(current)
    fixed_validation_state = FixedValidationPlanState(FIXED_VALIDATION_PLAN)
    loader = _build_loader(checkpoint_dir.parent / "store", completed_steps=0)
    states = {
        "model": ModelWrapper(model),
        "optimizer": optimizers,
        "train_state": train_state,
        PARALLEL_TOPOLOGY_STATE_KEY: topology_state,
        FIXED_VALIDATION_STATE_KEY: fixed_validation_state,
        "dataloader": loader,
    }
    dcp.load(states, checkpoint_id=str(checkpoint_dir))
    loader.set_optimizer_step(train_state.step)
    assert train_state.step == 1
    assert topology_state.loaded == decision.saved
    assert fixed_validation_state.loaded == FIXED_VALIDATION_PLAN

    loaded_snapshot = _logical_snapshot(model, optimizers)
    expected_snapshot = torch.load(
        checkpoint_dir.parent / "logical_snapshot.pt",
        map_location="cpu",
        weights_only=True,
    )
    # Assert on every rank so a failure cannot strand peers in a later
    # collective and turn a useful mismatch into a timeout.
    _assert_snapshot_equal(loaded_snapshot, expected_snapshot)

    if DEVICE_TYPE == "cuda":
        # A fresh 2x2 HSDP reference sees the same four logical samples used to
        # create the two-rank checkpoint.
        reference_model, reference_optimizers = _build_model(
            replicate=replicate, shard=shard
        )
        _update(reference_model, reference_optimizers, update=0)
        _assert_models_close(model, reference_model)
        _assert_optimizers_close(
            optimizers, reference_optimizers, model, reference_model
        )

    # The reconstructed loader starts at optimizer step one on every rank.
    batch = next(iter(loader))["input_ids"]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, batch.tolist())
    flattened = [row for rank_batch in gathered for row in rank_batch]
    assert len(flattened) == 4
    assert len({tuple(row) for row in flattened}) == 4

    if DEVICE_TYPE == "cuda":
        _update(model, optimizers, update=1)
        _update(reference_model, reference_optimizers, update=1)
        _assert_models_close(model, reference_model)
        _assert_optimizers_close(
            optimizers, reference_optimizers, model, reference_model
        )

    # A checkpoint written after reshape advertises the new topology.
    reshaped_dir = checkpoint_dir.parent / f"reshaped_{replicate}x{shard}"
    dcp.save(states, checkpoint_id=str(reshaped_dir))
    dist.barrier()
    reshaped_decision = inspect_checkpoint_topology(reshaped_dir, current)
    assert reshaped_decision.saved == current
    assert not reshaped_decision.topology_changed
    if dist.get_rank() == 0:
        print(
            f"FSDP(1x2) -> DP({replicate}x{shard}) "
            "model/Adam/data resume passed",
            flush=True,
        )


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in {"save", "load"}:
        raise SystemExit(f"usage: {sys.argv[0]} save|load CHECKPOINT_DIR")
    phase = sys.argv[1]
    checkpoint_dir = Path(sys.argv[2]).resolve()
    _init_distributed()
    try:
        if phase == "save":
            _save(checkpoint_dir)
        else:
            _load(checkpoint_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
