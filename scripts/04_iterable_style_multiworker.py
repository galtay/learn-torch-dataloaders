"""Script 04: Iterable-Style Dataset with Multiple Workers -- The Duplication Bug

Demonstrates:
- The critical pitfall: each worker gets its own copy of the iterator
- Without sharding, every worker yields the FULL dataset
- This means data is duplicated num_workers times

Run: uv run python scripts/04_iterable_style_multiworker.py
"""

from collections import Counter

from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import IntegerIterableDataset
from learn_torch_dataloaders.utils import section_header


def main():
    n = 50
    batch_size = 8
    num_workers = 2

    section_header(f"Iterable-Style, {num_workers} Workers, NO SHARDING (BUG!)")
    print(f"Dataset yields values 0..{n - 1}")
    print(f"Expected total items: {n}")
    print(f"Batch size: {batch_size}\n")

    dataset = IntegerIterableDataset(n=n)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    all_items = []
    for batch_idx, batch in enumerate(loader):
        items = batch.tolist()
        all_items.extend(items)
        print(f"Batch {batch_idx}: {items} (size={len(items)})")

    print(f"\nTotal items seen: {len(all_items)} (expected {n})")

    counts = Counter(all_items)
    duplicated = {v: c for v, c in sorted(counts.items()) if c > 1}
    print(f"Items appearing more than once: {len(duplicated)} out of {n}")
    print(f"Every item appears {num_workers} times!\n")

    print("--- Why this happens ---")
    print(f"1. DataLoader spawns {num_workers} worker processes")
    print("2. Each worker gets a COPY of the dataset object")
    print("3. Each worker calls __iter__() independently")
    print(f"4. Since our __iter__ yields 0..{n - 1} with no sharding logic,")
    print(f"   EACH worker yields all {n} items")
    print(f"5. Result: {n} x {num_workers} = {n * num_workers} total items\n")

    print("--- Fix ---")
    print("Use get_worker_info() inside __iter__ to shard data across workers.")
    print("See script 05 for the solution.")


if __name__ == "__main__":
    main()
