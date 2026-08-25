import unittest

from datasets import IterableDataset

from flame.data import ParallelAwareDataLoader


class ParallelAwareDataLoaderTest(unittest.TestCase):
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
