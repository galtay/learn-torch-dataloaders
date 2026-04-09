"""Script 03: Map-Style Dataset with Multiple Workers

Demonstrates:
- How the DataLoader distributes index fetching across worker processes
- The batch output queue ensures batches arrive in order
- Worker ID tracking via returned data (reliable, unlike printing from subprocesses)

Run: uv run python scripts/03_map_style_multiworker.py
"""

import torch
from torch.utils.data import DataLoader, Dataset

from learn_torch_dataloaders.utils import section_header


class IntegerMapDatasetWithWorkerInfo(Dataset):
    """Returns both the value and the worker ID that fetched it."""

    def __init__(self, n: int):
        self.data = list(range(n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1
        return {"value": self.data[idx], "worker_id": worker_id}


def run_with_workers(n: int, batch_size: int, num_workers: int):
    section_header(f"Map-Style, {num_workers} Workers, batch_size={batch_size}")

    dataset = IntegerMapDatasetWithWorkerInfo(n=n)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
    )

    print(f"Dataset size: {n}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}\n")

    for batch_idx, batch in enumerate(loader):
        values = batch["value"].tolist()
        workers = batch["worker_id"].tolist()
        items_str = ", ".join(f"{v}(w{w})" for v, w in zip(values, workers, strict=True))
        print(f"Batch {batch_idx}: [{items_str}]")

    print()


def main():
    n = 50
    batch_size = 8

    run_with_workers(n=n, batch_size=batch_size, num_workers=2)

    print("--- How it works ---")
    print(f"1. The sampler generates indices [0, 1, 2, ..., {n - 1}]")
    print(f"2. Indices are grouped into batches of {batch_size}")
    print("3. Batches are distributed to workers in round-robin fashion")
    print("4. Worker 0 fetches batch 0, Worker 1 fetches batch 1, etc.")
    print("5. The output queue reassembles batches in the original order")
    print("6. So batches ALWAYS come out in order, even though workers run in parallel\n")

    run_with_workers(n=n, batch_size=batch_size, num_workers=4)

    print("--- Key Takeaways ---")
    print("1. Map-style + DataLoader = automatic index distribution to workers")
    print("2. Batch ORDER is always preserved (the output queue handles this)")
    print("3. Each item shows which worker fetched it -- workers share the load")
    print("4. More workers doesn't change the data, just who fetches what")


if __name__ == "__main__":
    main()
