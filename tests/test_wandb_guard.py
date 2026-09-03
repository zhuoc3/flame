import os
import unittest
from unittest.mock import patch

from flame.wandb_guard import require_online_wandb
from torchtitan.components.metrics import BaseLogger, WandBLogger


class _Run:
    id = "audit-id"
    url = "https://wandb.invalid/audit-id"


class _Wandb:
    run = _Run()


class WandbGuardTest(unittest.TestCase):
    def test_rejects_silent_base_logger_fallback(self) -> None:
        with patch.dict(os.environ, {"WANDB_MODE": "online"}):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                require_online_wandb(BaseLogger(), save_for_all_ranks=False)

    def test_accepts_initialized_online_logger(self) -> None:
        backend = object.__new__(WandBLogger)
        backend.wandb = _Wandb()
        with patch.dict(os.environ, {"WANDB_MODE": "online"}):
            audit = require_online_wandb(backend, save_for_all_ranks=False)
        self.assertEqual(audit["active_loggers"], 1)
        self.assertEqual(audit["run_id"], "audit-id")

    def test_rejects_non_online_mode(self) -> None:
        backend = object.__new__(WandBLogger)
        backend.wandb = _Wandb()
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            with self.assertRaisesRegex(RuntimeError, "WANDB_MODE='offline'"):
                require_online_wandb(backend, save_for_all_ranks=False)


if __name__ == "__main__":
    unittest.main()
