import random
import time

import torch
from torch.utils.data import Dataset, IterableDataset


def _worker_shard(n: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    """Compute the [start, end) range for a worker such that shard sizes differ by at most 1.

    The first (n % num_workers) workers each get one extra item.

    Example: n=10, num_workers=3
        Worker 0: [0, 4)  -> 4 items
        Worker 1: [4, 7)  -> 3 items
        Worker 2: [7, 10) -> 3 items
    """
    base, remainder = divmod(n, num_workers)
    if worker_id < remainder:
        start = worker_id * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (worker_id - remainder) * base
        end = start + base
    return start, end


class IntegerMapDataset(Dataset):
    """Map-style dataset of integers 0..n-1."""

    def __init__(self, n: int, max_sleep: float = 0.0):
        self.data = list(range(n))
        self.max_sleep = max_sleep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.max_sleep > 0:
            time.sleep(random.random() * self.max_sleep)
        return self.data[idx]


class IntegerIterableDataset(IterableDataset):
    """Iterable-style dataset that yields 0..n-1.

    No sharding -- every worker yields the full dataset.
    This is intentionally naive to demonstrate the data duplication problem.
    """

    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        yield from range(self.n)


class ShardedIntegerIterableDataset(IterableDataset):
    """Iterable-style dataset that correctly shards 0..n-1 across workers.

    Uses get_worker_info() to partition data so each worker yields
    a non-overlapping slice. Shard sizes differ by at most 1.
    """

    def __init__(self, n: int, max_sleep: float = 0.0):
        self.n = n
        self.max_sleep = max_sleep

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self.n
        else:
            start, end = _worker_shard(self.n, worker_info.id, worker_info.num_workers)
        for i in range(start, end):
            if self.max_sleep > 0:
                time.sleep(random.random() * self.max_sleep)
            yield i


class ShuffleableIterableDataset(IterableDataset):
    """Sharded iterable dataset with optional shuffling.

    Demonstrates how to shuffle inside __iter__ using per-worker seeding.
    The seed can be set externally (e.g. per-epoch) for reproducibility.
    """

    def __init__(self, n: int, shuffle: bool = False, seed: int | None = None):
        self.n = n
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self.n
        else:
            start, end = _worker_shard(self.n, worker_info.id, worker_info.num_workers)

        indices = list(range(start, end))

        if self.shuffle:
            if self.seed is not None:
                worker_seed = self.seed + worker_info.id if worker_info is not None else self.seed
                rng = random.Random(worker_seed)
            else:
                rng = random.Random()
            rng.shuffle(indices)

        yield from indices
