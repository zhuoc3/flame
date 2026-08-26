import unittest

import torch
from datasets import Dataset, IterableDataset

from flame.data import OnlineTokenizedIterableDataset, ParallelAwareDataLoader


class _ToyTokenizer:
    def __call__(self, texts, return_attention_mask=False):
        del return_attention_mask
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


class ParallelAwareDataLoaderTest(unittest.TestCase):
    def test_online_validation_reset_replays_exact_initial_prefix(self):
        source = Dataset.from_dict({"text": ["ab", "cd", "ef"]}).to_iterable_dataset(
            num_shards=1
        )
        dataset = OnlineTokenizedIterableDataset(
            dataset=source,
            tokenizer=_ToyTokenizer(),
            seq_len=2,
            rank=0,
            world_size=1,
        )
        first = next(iter(dataset))["input_ids"]
        advanced = next(iter(dataset))["input_ids"]
        dataset.reset()
        replayed = next(iter(dataset))["input_ids"]

        self.assertFalse(torch.equal(first, advanced))
        torch.testing.assert_close(replayed, first)

    def test_zero_workers_disables_multiprocessing_only_options(self):
        dataset = IterableDataset.from_generator(
            lambda: ({"value": value} for value in range(4))
        )
        loader = ParallelAwareDataLoader(
            rank=0,
            dataset=dataset,
            batch_size=2,
            collate_fn=lambda rows: [row["value"] for row in rows],
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=True,
        )

        self.assertEqual(next(iter(loader)), [0, 1])


if __name__ == "__main__":
    unittest.main()
