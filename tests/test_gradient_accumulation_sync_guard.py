import unittest
from types import SimpleNamespace

from flame.gradient_accumulation import controls_gradient_sync


class GradientAccumulationSyncGuardTest(unittest.TestCase):
    def test_pure_fsdp_controls_gradient_sync(self):
        dims = SimpleNamespace(dp_replicate_enabled=False, dp_shard_enabled=True)
        self.assertTrue(controls_gradient_sync(dims, ga_steps=16))

    def test_ddp_and_hsdp_control_gradient_sync(self):
        ddp = SimpleNamespace(dp_replicate_enabled=True, dp_shard_enabled=False)
        hsdp = SimpleNamespace(dp_replicate_enabled=True, dp_shard_enabled=True)
        self.assertTrue(controls_gradient_sync(ddp, ga_steps=16))
        self.assertTrue(controls_gradient_sync(hsdp, ga_steps=16))

    def test_single_microstep_needs_no_manual_control(self):
        fsdp = SimpleNamespace(dp_replicate_enabled=False, dp_shard_enabled=True)
        self.assertFalse(controls_gradient_sync(fsdp, ga_steps=1))


if __name__ == "__main__":
    unittest.main()
