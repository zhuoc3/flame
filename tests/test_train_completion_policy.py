import os
import unittest
from unittest.mock import patch

from flame.completion import pending_job_cancellation_enabled


class TrainCompletionPolicyTest(unittest.TestCase):
    def test_legacy_default_remains_enabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(pending_job_cancellation_enabled())

    def test_explicit_false_values_disable_pending_job_cancellation(self) -> None:
        for value in ("0", "false", "False", "no", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"FLAME_CANCEL_PENDING_ON_COMPLETE": value},
                clear=True,
            ):
                self.assertFalse(pending_job_cancellation_enabled())


if __name__ == "__main__":
    unittest.main()
