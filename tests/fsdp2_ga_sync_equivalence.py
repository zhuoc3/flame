#!/usr/bin/env python3
"""Two-rank FSDP2 GA equivalence probe used by the production GPU gate."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.debug import CommDebugMode


class TinyModel(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(depth)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = torch.nn.functional.silu(layer(value))
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("every", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(20260825)
    model = TinyModel().to(device)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    snapshots = []
    accumulation = 4
    for update in range(2):
        optimizer.zero_grad()
        losses = []
        comm_mode = CommDebugMode()
        with comm_mode:
            for microstep in range(accumulation):
                model.set_requires_gradient_sync(
                    args.mode == "every" or microstep == accumulation - 1
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    9000 + update * 100 + rank * accumulation + microstep
                )
                # Match the BF16 activations used by production. Keeping an
                # FP32 MSE target promotes the fused loss to FP32 but leaves
                # MSE backward expecting a BF16 input gradient, which makes
                # the probe fail before it exercises FSDP2 communication.
                inputs = torch.randn(
                    16, 128, generator=generator, device=device, dtype=torch.bfloat16
                )
                targets = torch.randn(
                    16, 128, generator=generator, device=device, dtype=torch.bfloat16
                )
                prediction = model(inputs)
                if prediction.dtype != torch.bfloat16:
                    raise RuntimeError(f"expected BF16 prediction, got {prediction.dtype}")
                # Production cross-entropy accumulates its scalar loss in
                # FP32 while gradients entering BF16 activations are cast back
                # by autograd. Express that explicitly instead of relying on
                # mixed-dtype MSE behavior.
                loss = (
                    (prediction.float() - targets.float()).square().mean()
                    / accumulation
                )
                loss.backward()
                losses.append(float(loss.detach()))

        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient for {name}")
            gradients[name] = parameter.grad.full_tensor().cpu()
        grad_norm = float(
            torch.sqrt(sum(value.double().square().sum() for value in gradients.values()))
        )
        optimizer.step()
        parameters = {
            name: parameter.full_tensor().detach().cpu()
            for name, parameter in model.named_parameters()
        }
        optimizer_state = {}
        for name, parameter in model.named_parameters():
            state = optimizer.state[parameter]
            optimizer_state[name] = {
                "step": int(state["step"].item()),
                "exp_avg": state["exp_avg"].full_tensor().cpu(),
                "exp_avg_sq": state["exp_avg_sq"].full_tensor().cpu(),
            }
        snapshots.append(
            {
                "losses": losses,
                "gradients": gradients,
                "grad_norm": grad_norm,
                "parameters": parameters,
                "optimizer": optimizer_state,
                "comm_counts": {
                    str(operation): count
                    for operation, count in comm_mode.get_comm_counts().items()
                },
            }
        )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "snapshots": snapshots,
            },
            args.output,
        )
        print(json.dumps({"mode": args.mode, "updates": len(snapshots)}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
