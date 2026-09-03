import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from flame.distributed_control import synchronize_stop_request


def _gloo_stop_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        # Only rank one asks to stop; both ranks must take the stop branch.
        assert synchronize_stop_request(rank == 1)
        # No rank asks to stop; both ranks must continue.
        assert not synchronize_stop_request(False)
    finally:
        dist.destroy_process_group()


class DistributedControlTest(unittest.TestCase):
    def test_gloo_any_rank_stop_is_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "gloo-rendezvous")
            mp.spawn(
                _gloo_stop_worker,
                args=(2, rendezvous),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
