import os
import sys
import unittest

from flame.data import OnlineTokenizedIterableDataset


class StatefulSource:
    def __init__(self):
        self.position = 0

    def shard(self, *_args):
        return self

    def __iter__(self):
        while self.position < 100:
            value = self.position
            self.position += 1
            yield {"text": str(value)}

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


class IntegerTokenizer:
    def __call__(self, values, return_attention_mask=False):
        return {"input_ids": [[int(value)] for value in values]}


def take(dataset, count):
    iterator = iter(dataset)
    return [int(next(iterator)["input_ids"].item()) for _ in range(count)]


class OnlineTokenizedValidationResetTest(unittest.TestCase):
    def test_reset_rewinds_wrapper_and_underlying_iterable_state(self):
        dataset = OnlineTokenizedIterableDataset(
            StatefulSource(), IntegerTokenizer(), seq_len=1
        )

        self.assertEqual(take(dataset, 2), [0, 1])
        self.assertEqual(take(dataset, 2), [2, 3])

        dataset.reset()
        self.assertIsNone(dataset.states)
        self.assertEqual(dataset.tokens, [])
        self.assertEqual(take(dataset, 2), [0, 1])

    def test_reset_is_repeatable_after_tokenizer_prefetch(self):
        dataset = OnlineTokenizedIterableDataset(
            StatefulSource(), IntegerTokenizer(), seq_len=1
        )

        self.assertEqual(take(dataset, 3), [0, 1, 2])
        dataset.reset()
        self.assertEqual(take(dataset, 3), [0, 1, 2])
        dataset.reset()
        self.assertEqual(take(dataset, 3), [0, 1, 2])


if __name__ == "__main__":
    # This pinned Python 3.12 environment's third-party ``multiprocess``
    # resource tracker can hang during interpreter teardown after importing
    # datasets. Flush the real unittest result, then bypass only that broken
    # teardown path. Normal unittest/pytest discovery does not enter here.
    program = unittest.main(exit=False)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
