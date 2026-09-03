import itertools
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from flame.data import (
    DataCollatorForLanguageModeling,
    DeterministicParentBlockDataset,
    DeterministicRepositorySliceDataset,
    FixedValidationSampler,
    Int64TokenBlockDatasetView,
    MemmapTokenBlockDataset,
    RepositorySliceViewDataset,
    TopologyIndependentDataLoader,
    TopologyIndependentSampler,
    build_dataloader,
    deterministic_permute,
)


PARENT_SEQ_LEN = 16_384
PARENTS_PER_STEP = 32


class _IndexDataset(Dataset):
    def __len__(self) -> int:
        return 1 << 30

    def __getitem__(self, index: int) -> int:
        return index


class _IdentityTokenizer:
    pad_token_id = 0


def _index_collate(indices):
    return torch.tensor(indices, dtype=torch.int64)


def _npy_payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            np.lib.format.read_array_header_2_0(handle)
        else:
            raise AssertionError(f"Unsupported test NPY version: {version}")
        digest.update(handle.read())
    return digest.hexdigest()


def _rank_step_indices(
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    samples_per_step: int,
    optimizer_step: int,
):
    sampler = TopologyIndependentSampler(
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        samples_per_step=samples_per_step,
    )
    sampler.set_optimizer_step(optimizer_step)
    return list(
        itertools.islice(
            iter(sampler), batch_size * gradient_accumulation_steps
        )
    )


class TopologyIndependentDataTest(unittest.TestCase):
    def _repository_view(self, root: Path) -> RepositorySliceViewDataset:
        source = root / "source"
        tokens_dir = source / "tokens"
        view = root / "view"
        tokens_dir.mkdir(parents=True)
        view.mkdir()
        np.asarray(
            [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 30, 31, 32],
            dtype="<u2",
        ).tofile(tokens_dir / "part-00000.bin")
        source_manifest = {
            "format": "stack-repository-token-store",
            "format_version": 1,
            "fingerprint": "source-fingerprint",
            "num_repositories": 3,
            "total_tokens": 16,
        }
        source_manifest_path = source / "manifest.json"
        source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
        repository_dtype = np.dtype(
            [
                ("token_shard", "<u4"),
                ("token_offset", "<u8"),
                ("num_tokens", "<u8"),
            ]
        )
        repositories = np.zeros(3, dtype=repository_dtype)
        repositories["token_offset"] = [0, 8, 13]
        repositories["num_tokens"] = [8, 5, 3]
        np.save(root / "repositories.npy", repositories, allow_pickle=False)
        np.save(
            view / "block_offsets.npy",
            np.asarray([0, 2, 3, 4], dtype="<u8"),
            allow_pickle=False,
        )
        manifest = {
            "format": "stack-repository-logical-block-view",
            "format_version": 1,
            "fingerprint": "view-fingerprint",
            "seq_len": 4,
            "num_rows": 4,
            "num_valid_tokens": 15,
            "min_tail_tokens": 3,
            "pad_token_id": 2,
            "source": {
                "path": os.path.relpath(source, view),
                "manifest_sha256": hashlib.sha256(
                    source_manifest_path.read_bytes()
                ).hexdigest(),
                "fingerprint": source_manifest["fingerprint"],
            },
            "repository_index_file": os.path.relpath(
                root / "repositories.npy", view
            ),
            "block_offsets_file": "block_offsets.npy",
            "shuffle": {
                "algorithm": "splitmix64-feistel6-cyclewalk-v1",
                "seed": 91,
                "epoch_keyed": True,
            },
        }
        (view / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return RepositorySliceViewDataset(view)

    def _virtual_dataset(self, seq_len: int, num_parents: int = 67):
        class _Parents(Dataset):
            def __len__(self):
                return num_parents

            def __getitem__(self, index):
                return {"input_ids": np.full(PARENT_SEQ_LEN, index, np.uint16)}

        return DeterministicParentBlockDataset(
            parent_dataset=_Parents(),
            seq_len=seq_len,
            parent_seq_len=PARENT_SEQ_LEN,
            parent_blocks_per_step=PARENTS_PER_STEP,
            seed=1234,
        )

    def test_prp_is_a_permutation_for_awkward_sizes(self):
        for size in (1, 2, 3, 31, 32, 33, 65, 257):
            for epoch in (0, 1, 19):
                values = [
                    deterministic_permute(i, size, seed=765, epoch=epoch)
                    for i in range(size)
                ]
                self.assertEqual(sorted(values), list(range(size)))

    def test_repository_view_reads_full_and_retained_tail_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._repository_view(Path(temporary))
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.block_location(3), (2, 0, 3))
            np.testing.assert_array_equal(dataset[0]["input_ids"], [10, 11, 12, 13])
            np.testing.assert_array_equal(dataset[1]["input_ids"], [14, 15, 16, 17])
            np.testing.assert_array_equal(dataset[2]["input_ids"], [20, 21, 22, 23])
            np.testing.assert_array_equal(dataset[3]["input_ids"], [30, 31, 32, 2])
            self.assertEqual(dataset[3]["valid_length"], 3)

    def test_repository_view_order_is_epoch_keyed_and_topology_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._repository_view(Path(temporary))
            virtual = DeterministicRepositorySliceDataset(dataset, blocks_per_step=2)
            epoch_zero = [virtual.view_index(index)[0] for index in range(4)]
            epoch_one = [virtual.view_index(index)[0] for index in range(4, 8)]
            self.assertEqual(sorted(epoch_zero), list(range(4)))
            self.assertEqual(sorted(epoch_one), list(range(4)))
            self.assertNotEqual(epoch_zero, epoch_one)

            expected = None
            for world_size, accumulation in ((1, 2), (2, 1)):
                indices = []
                for rank in range(world_size):
                    indices.extend(
                        _rank_step_indices(
                            rank=rank,
                            world_size=world_size,
                            batch_size=1,
                            gradient_accumulation_steps=accumulation,
                            samples_per_step=2,
                            optimizer_step=1,
                        )
                    )
                view_indices = sorted(virtual.view_index(index)[0] for index in indices)
                if expected is None:
                    expected = view_indices
                self.assertEqual(view_indices, expected)

    def test_repository_view_can_be_read_by_multiple_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._repository_view(Path(temporary))
            virtual = DeterministicRepositorySliceDataset(dataset, blocks_per_step=2)
            loader = torch.utils.data.DataLoader(virtual, batch_size=1, num_workers=2)
            rows = list(loader)
            self.assertEqual(len(rows), 4)
            valid_lengths = [int(row["attention_mask"].sum().item()) for row in rows]
            self.assertEqual(sorted(valid_lengths), [3, 4, 4, 4])

    def test_repository_view_data_plan_survives_dp_shape_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._repository_view(Path(temporary))
            common = {
                "dataset": dataset,
                "tokenizer": _IdentityTokenizer(),
                "rank": 0,
                "batch_size": 1,
                "seq_len": 4,
                "num_workers": 0,
                "parent_blocks_per_step": 2,
                "loss_start_position": 2,
            }
            dp1 = build_dataloader(
                **common, world_size=1, gradient_accumulation_steps=2
            )
            dp2 = build_dataloader(
                **common, world_size=2, gradient_accumulation_steps=1
            )
            self.assertEqual(dp1.data_plan, dp2.data_plan)
            dp2.load_state_dict(dp1.state_dict())

    def test_epoch_rollover_is_step_aligned_and_deterministic(self):
        dataset = self._virtual_dataset(seq_len=PARENT_SEQ_LEN, num_parents=67)
        self.assertEqual(dataset.steps_per_epoch, 2)

        cohorts = []
        for step in (0, 1, 2):
            triples = [
                dataset.parent_and_child(step * PARENTS_PER_STEP + lane)
                for lane in range(PARENTS_PER_STEP)
            ]
            parents = [parent for parent, child, epoch in triples]
            self.assertEqual(len(set(parents)), PARENTS_PER_STEP)
            self.assertEqual({child for parent, child, epoch in triples}, {0})
            self.assertEqual(
                {epoch for parent, child, epoch in triples}, {step // 2}
            )
            cohorts.append(parents)

        self.assertTrue(set(cohorts[0]).isdisjoint(cohorts[1]))
        self.assertEqual(
            cohorts[2],
            [
                dataset.parent_and_child(2 * PARENTS_PER_STEP + lane)[0]
                for lane in range(PARENTS_PER_STEP)
            ],
        )

    def test_architecture_matrix_uses_same_32_parents_each_step(self):
        # (sequence length, DP world size, local batch, accumulation)
        matrix = (
            (256, 2, 32, 32),
            (512, 4, 16, 16),
            (1_024, 4, 16, 8),
            (2_048, 4, 8, 8),
            (4_096, 4, 2, 16),
            (8_192, 4, 2, 8),
            (16_384, 4, 1, 8),
        )
        optimizer_step = 3
        expected_parents = None

        for seq_len, world_size, batch_size, accumulation in matrix:
            dataset = self._virtual_dataset(seq_len)
            all_indices = []
            for rank in range(world_size):
                all_indices.extend(
                    _rank_step_indices(
                        rank=rank,
                        world_size=world_size,
                        batch_size=batch_size,
                        gradient_accumulation_steps=accumulation,
                        samples_per_step=dataset.samples_per_step,
                        optimizer_step=optimizer_step,
                    )
                )

            step_begin = optimizer_step * dataset.samples_per_step
            self.assertEqual(
                sorted(all_indices),
                list(range(step_begin, step_begin + dataset.samples_per_step)),
            )
            parent_children = [dataset.parent_and_child(i) for i in all_indices]
            parents = {parent for parent, child, epoch in parent_children}
            if expected_parents is None:
                expected_parents = parents
            self.assertEqual(parents, expected_parents)
            for parent in parents:
                self.assertEqual(
                    sorted(
                        child
                        for seen_parent, child, epoch in parent_children
                        if seen_parent == parent
                    ),
                    list(range(PARENT_SEQ_LEN // seq_len)),
                )

    def test_256_microbatches_interleave_all_32_parents(self):
        dataset = self._virtual_dataset(256)
        world_size, batch_size, accumulation = 2, 32, 32
        rank_indices = [
            _rank_step_indices(
                rank=rank,
                world_size=world_size,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                samples_per_step=dataset.samples_per_step,
                optimizer_step=0,
            )
            for rank in range(world_size)
        ]

        for microstep in range(accumulation):
            global_microbatch = []
            for rank in range(world_size):
                begin = microstep * batch_size
                global_microbatch.extend(
                    rank_indices[rank][begin : begin + batch_size]
                )
            parent_ids = [
                dataset.parent_and_child(index)[0]
                for index in global_microbatch
            ]
            self.assertEqual(len(set(parent_ids)), PARENTS_PER_STEP)
            self.assertEqual(
                sorted(parent_ids.count(parent) for parent in set(parent_ids)),
                [2] * PARENTS_PER_STEP,
            )

    def test_16k_reshaping_dp8_dp16_dp32_has_identical_step(self):
        dataset = self._virtual_dataset(PARENT_SEQ_LEN)
        optimizer_step = 5
        topologies = ((8, 1, 4), (16, 1, 2), (32, 1, 1))
        expected = None

        for world_size, batch_size, accumulation in topologies:
            rank_indices = [
                _rank_step_indices(
                    rank=rank,
                    world_size=world_size,
                    batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    samples_per_step=dataset.samples_per_step,
                    optimizer_step=optimizer_step,
                )
                for rank in range(world_size)
            ]
            flattened = sorted(itertools.chain.from_iterable(rank_indices))
            if expected is None:
                expected = flattened
            self.assertEqual(flattened, expected)
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_resume_cursor_starts_at_next_optimizer_cohort(self):
        samples_per_step = PARENTS_PER_STEP
        sampler = TopologyIndependentSampler(
            rank=3,
            world_size=8,
            batch_size=1,
            gradient_accumulation_steps=4,
            samples_per_step=samples_per_step,
        )
        sampler.set_optimizer_step(10_000)
        actual = list(itertools.islice(iter(sampler), 4))
        self.assertEqual(
            actual,
            [10_000 * samples_per_step + 3 + 8 * micro for micro in range(4)],
        )

    def test_fixed_validation_is_identical_across_dp8_dp16_dp32(self):
        expected = list(range(960))
        expected_local_lengths = {8: 120, 16: 60, 32: 30}
        for world_size in (8, 16, 32):
            rank_indices = []
            for rank in range(world_size):
                sampler = FixedValidationSampler(
                    rank=rank,
                    world_size=world_size,
                    batch_size=1,
                    num_samples=960,
                )
                first = list(sampler)
                self.assertEqual(first, list(sampler))
                self.assertEqual(len(first), expected_local_lengths[world_size])
                rank_indices.extend(first)
            self.assertEqual(sorted(rank_indices), expected)
            self.assertEqual(len(rank_indices), len(set(rank_indices)))

    def test_fixed_validation_refuses_padding_or_duplicate_rank(self):
        with self.assertRaisesRegex(ValueError, "padding would duplicate"):
            FixedValidationSampler(
                rank=0, world_size=7, batch_size=1, num_samples=960
            )
        with self.assertRaisesRegex(ValueError, "rank must be"):
            FixedValidationSampler(
                rank=8, world_size=8, batch_size=1, num_samples=960
            )

    def test_multiworker_loader_preserves_sampler_order(self):
        expected_sampler = TopologyIndependentSampler(
            rank=2,
            world_size=4,
            batch_size=2,
            gradient_accumulation_steps=16,
            samples_per_step=128,
        )
        expected_sampler.set_optimizer_step(7)
        expected = list(itertools.islice(iter(expected_sampler), 10))

        sampler = TopologyIndependentSampler(
            rank=2,
            world_size=4,
            batch_size=2,
            gradient_accumulation_steps=16,
            samples_per_step=128,
        )
        loader = TopologyIndependentDataLoader(
            dataset=_IndexDataset(),
            sampler=sampler,
            data_plan={"test": "multiworker-order"},
            batch_size=2,
            collate_fn=_index_collate,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=False,
            snapshot_every_n_steps=1,
        )
        loader.set_optimizer_step(7)
        iterator = iter(loader)
        actual = torch.cat([next(iterator) for _ in range(5)]).tolist()
        del iterator

        self.assertEqual(actual, expected)
        state = loader.state_dict()
        self.assertEqual(set(state), {"data_plan"})
        loader.load_state_dict(state)

    def test_memmap_store_returns_exact_uint16_parent_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = np.lib.format.open_memmap(
                root / "tokens.npy",
                mode="w+",
                dtype=np.uint16,
                shape=(PARENTS_PER_STEP, PARENT_SEQ_LEN),
            )
            positions = np.arange(PARENT_SEQ_LEN, dtype=np.uint16)
            for parent in range(PARENTS_PER_STEP):
                tokens[parent] = positions ^ np.uint16(parent)
            tokens.flush()
            del tokens
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "seq_len": PARENT_SEQ_LEN,
                        "num_rows": PARENTS_PER_STEP,
                        "tokens_file": "tokens.npy",
                    }
                )
            )

            parents = MemmapTokenBlockDataset(root)
            virtual = DeterministicParentBlockDataset(
                parent_dataset=parents,
                seq_len=256,
                parent_seq_len=PARENT_SEQ_LEN,
                parent_blocks_per_step=PARENTS_PER_STEP,
                seed=91,
            )
            parent, child, epoch = virtual.parent_and_child(0)
            sample = virtual[0]["input_ids"]
            expected = torch.from_numpy(
                ((positions[:256] ^ np.uint16(parent)).astype(np.int64))
            )

            self.assertEqual(child, 0)
            self.assertEqual(epoch, 0)
            torch.testing.assert_close(sample, expected, rtol=0, atol=0)

    def test_memmap_store_optional_payload_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens_path = root / "tokens.npy"
            np.save(
                tokens_path,
                np.arange(32, dtype=np.uint16).reshape(4, 8),
                allow_pickle=False,
            )
            manifest = {
                "seq_len": 8,
                "num_rows": 4,
                "tokens_payload_sha256": _npy_payload_sha256(tokens_path),
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            MemmapTokenBlockDataset(root, verify_payload=True)

            manifest["tokens_payload_sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "payload SHA256 mismatch"):
                MemmapTokenBlockDataset(root, verify_payload=True)

    def test_fixed_validation_loader_casts_uint16_rows_to_int64(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = np.arange(32, dtype=np.uint16).reshape(4, 8)
            np.save(root / "tokens.npy", source, allow_pickle=False)
            (root / "manifest.json").write_text(
                json.dumps({"seq_len": 8, "num_rows": 4})
            )

            dataset = Int64TokenBlockDatasetView(
                MemmapTokenBlockDataset(root)
            )
            sampler = FixedValidationSampler(
                rank=1, world_size=2, batch_size=1, num_samples=len(dataset)
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=1,
                collate_fn=DataCollatorForLanguageModeling(
                    tokenizer=_IdentityTokenizer(), context_len=8, varlen=False
                ),
                num_workers=0,
            )

            batches = list(loader)
            self.assertEqual(len(batches), 2)
            for batch, row in zip(batches, (1, 3)):
                self.assertEqual(batch["input_ids"].dtype, torch.int64)
                self.assertEqual(batch["labels"].dtype, torch.int64)
                torch.testing.assert_close(
                    batch["input_ids"][0],
                    torch.from_numpy(source[row].astype(np.int64)),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    batch["labels"], batch["input_ids"], rtol=0, atol=0
                )

    def test_memmap_store_without_valid_lengths_preserves_legacy_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = np.arange(16, dtype=np.uint16).reshape(2, 8)
            np.save(root / "tokens.npy", source, allow_pickle=False)
            (root / "manifest.json").write_text(
                json.dumps({"seq_len": 8, "num_rows": 2})
            )

            dataset = MemmapTokenBlockDataset(root)
            self.assertEqual(dataset.column_names, ["input_ids"])
            self.assertEqual(set(dataset[0]), {"input_ids"})
            self.assertEqual(
                set(Int64TokenBlockDatasetView(dataset)[0]), {"input_ids"}
            )

    def test_valid_lengths_mask_full_and_padded_tail_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = np.arange(16, dtype=np.uint16).reshape(2, 8)
            np.save(root / "tokens.npy", source, allow_pickle=False)
            np.save(
                root / "valid_lengths.npy",
                np.array([8, 5], dtype=np.uint32),
                allow_pickle=False,
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "seq_len": 8,
                        "num_rows": 2,
                        "valid_lengths_file": "valid_lengths.npy",
                    }
                )
            )

            memmap_dataset = MemmapTokenBlockDataset(root)
            self.assertEqual(
                memmap_dataset.column_names, ["input_ids", "valid_length"]
            )
            self.assertEqual(memmap_dataset[0]["valid_length"], 8)
            self.assertEqual(memmap_dataset[1]["valid_length"], 5)
            self.assertEqual(memmap_dataset.num_valid_tokens, 13)

            dataset = Int64TokenBlockDatasetView(memmap_dataset)
            torch.testing.assert_close(
                dataset[0]["attention_mask"], torch.ones(8, dtype=torch.long)
            )
            torch.testing.assert_close(
                dataset[1]["attention_mask"],
                torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
            )

            batch = DataCollatorForLanguageModeling(
                tokenizer=_IdentityTokenizer(), context_len=8, varlen=False
            )([dataset[0], dataset[1]])
            torch.testing.assert_close(
                batch["attention_mask"],
                torch.tensor(
                    [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]]
                ),
            )
            torch.testing.assert_close(
                batch["labels"][1],
                torch.tensor([8, 9, 10, 11, 12, -100, -100, -100]),
            )

            manifest = memmap_dataset.manifest
            manifest["tokens_payload_sha256"] = _npy_payload_sha256(
                root / "tokens.npy"
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                ValueError, "no valid-length payload checksum"
            ):
                MemmapTokenBlockDataset(root, verify_payload=True)

            manifest["valid_lengths_payload_sha256"] = _npy_payload_sha256(
                root / "valid_lengths.npy"
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            MemmapTokenBlockDataset(root, verify_payload=True)

            manifest["valid_lengths_payload_sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                ValueError, "Valid-length payload SHA256 mismatch"
            ):
                MemmapTokenBlockDataset(root, verify_payload=True)

    def test_loss_start_position_preserves_boundary_and_tail_mask(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=_IdentityTokenizer(),
            context_len=8,
            varlen=False,
            loss_start_position=4,
        )
        batch = collator(
            [
                {
                    "input_ids": torch.arange(8),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 0, 0]),
                },
                {
                    "input_ids": torch.arange(10, 18),
                    "attention_mask": torch.ones(8, dtype=torch.long),
                },
            ]
        )

        torch.testing.assert_close(
            batch["labels"],
            torch.tensor(
                [
                    [-100, -100, -100, -100, 4, 5, -100, -100],
                    [-100, -100, -100, -100, 14, 15, 16, 17],
                ]
            ),
        )
        # Models shift labels internally. The first retained target is token
        # position 4, paired with the logits produced at position 3.
        torch.testing.assert_close(
            batch["labels"][..., 1:],
            torch.tensor(
                [
                    [-100, -100, -100, 4, 5, -100, -100],
                    [-100, -100, -100, 14, 15, 16, 17],
                ]
            ),
        )

    def test_loss_start_position_masks_varlen_labels(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=_IdentityTokenizer(),
            context_len=8,
            varlen=True,
            loss_start_position=3,
        )
        batch = collator([{"input_ids": torch.arange(8), "cu_seqlens": [0, 8]}])
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([[-100, -100, -100, 3, 4, 5, 6, 7]]),
        )

    def test_loss_start_position_rejects_negative_value(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=_IdentityTokenizer(), loss_start_position=-1
        )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            collator([torch.arange(8)])

    def test_masked_parent_rejects_shorter_child_views(self):
        class _Parents(Dataset):
            valid_lengths_path = Path("valid_lengths.npy")

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {
                    "input_ids": np.arange(8, dtype=np.uint16),
                    "valid_length": 6,
                }

        with self.assertRaisesRegex(
            ValueError, "cannot be split into shorter children"
        ):
            DeterministicParentBlockDataset(
                parent_dataset=_Parents(),
                seq_len=4,
                parent_seq_len=8,
                parent_blocks_per_step=1,
                seed=42,
            )

        dataset = DeterministicParentBlockDataset(
            parent_dataset=_Parents(),
            seq_len=8,
            parent_seq_len=8,
            parent_blocks_per_step=1,
            seed=42,
        )
        child = dataset[0]
        torch.testing.assert_close(
            child["attention_mask"], torch.tensor([1, 1, 1, 1, 1, 1, 0, 0])
        )
        torch.testing.assert_close(
            child["input_ids"], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        )

    def test_varlen_collator_rejects_attention_masks(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=_IdentityTokenizer(), context_len=8, varlen=True
        )
        with self.assertRaisesRegex(
            ValueError, "Masked token blocks are not supported with varlen=True"
        ):
            collator(
                [
                    {
                        "input_ids": torch.arange(8),
                        "attention_mask": torch.tensor(
                            [1, 1, 1, 1, 1, 0, 0, 0]
                        ),
                    }
                ]
            )

    def test_sampler_rejects_non_exact_logical_batch(self):
        with self.assertRaisesRegex(ValueError, "one logical parent cohort"):
            TopologyIndependentSampler(
                rank=0,
                world_size=16,
                batch_size=1,
                gradient_accumulation_steps=1,
                samples_per_step=32,
            )
        with self.assertRaisesRegex(ValueError, "rank must be"):
            TopologyIndependentSampler(
                rank=8,
                world_size=8,
                batch_size=1,
                gradient_accumulation_steps=4,
                samples_per_step=32,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            TopologyIndependentSampler(
                rank=0,
                world_size=0,
                batch_size=1,
                gradient_accumulation_steps=4,
                samples_per_step=32,
            )


if __name__ == "__main__":
    unittest.main()
