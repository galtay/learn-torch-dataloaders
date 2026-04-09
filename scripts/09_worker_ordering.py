"""Script 09: Deterministic Ordering -- Map vs Iterable Under Variable Load

Demonstrates:
- Map-style datasets guarantee the same sample order across runs, even when
  workers take random amounts of time to process items
- Iterable datasets do NOT guarantee the same sample order when workers
  have variable processing times, because the interleaving depends on
  which worker produces items first

We simulate variable processing time with a small random sleep in each
dataset's __getitem__ / __iter__.

Run: uv run python scripts/09_worker_ordering.py
"""

import torch
from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import IntegerMapDataset, ShardedIntegerIterableDataset
from learn_torch_dataloaders.utils import section_header


def collect_all_items(loader):
    """Collect all items from a loader in order, returning a flat list."""
    items = []
    for batch in loader:
        items.extend(batch.tolist())
    return items


def main():
    n = 50
    batch_size = 8
    num_workers = 3
    max_sleep = 0.01  # up to 10ms random sleep per item
    num_runs = 4

    # =========================================================================
    # Part A: Map-style with random sleep
    # =========================================================================
    section_header("Part A: Map-Style with Random Sleep")
    print(f"n={n}, batch_size={batch_size}, num_workers={num_workers}")
    print(f"max_sleep={max_sleep}s per item\n")

    map_results = []
    for run in range(num_runs):
        dataset = IntegerMapDataset(n=n, max_sleep=max_sleep)
        g = torch.Generator()
        g.manual_seed(42)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=g,
        )
        items = collect_all_items(loader)
        map_results.append(items)
        print(f"Run {run}: {items}")

    map_all_same = all(r == map_results[0] for r in map_results)
    print(f"\nAll {num_runs} runs identical? {map_all_same}")
    if map_all_same:
        print("-> YES! Map-style order is determined by the sampler, not worker timing.")
        print("   Workers are just fetchers. The output queue reassembles batches in")
        print("   sampler order regardless of which worker finishes first.")

    # =========================================================================
    # Part B: Iterable-style with random sleep
    # =========================================================================
    section_header("Part B: Iterable-Style (Sharded) with Random Sleep")
    print(f"n={n}, batch_size={batch_size}, num_workers={num_workers}")
    print(f"max_sleep={max_sleep}s per item\n")

    iter_results = []
    for run in range(num_runs):
        dataset = ShardedIntegerIterableDataset(n=n, max_sleep=max_sleep)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        items = collect_all_items(loader)
        iter_results.append(items)
        print(f"Run {run}: {items}")

    iter_all_same = all(r == iter_results[0] for r in iter_results)
    unique_orderings = len(set(tuple(r) for r in iter_results))
    print(f"\nAll {num_runs} runs identical? {iter_all_same}")
    print(f"Unique orderings: {unique_orderings} out of {num_runs} runs")
    if not iter_all_same:
        print("-> NO! Iterable-style order depends on worker timing.")
        print("   Each worker yields items from its shard independently.")
        print("   Random sleep causes workers to produce items at different rates,")
        print("   changing which worker's items end up in which batch.")

    # =========================================================================
    # Part C: Verify same data, different order
    # =========================================================================
    section_header("Part C: Same Data, Different Order")
    print("Even though order differs, all runs see the same set of items:\n")
    for run, items in enumerate(iter_results):
        print(f"Run {run}: sorted={sorted(items)}, count={len(items)}")

    print("\n--- Key Takeaway ---")
    print("Map-style:     sample order = sampler order (workers are transparent)")
    print("Iterable-style: sample order = interleaving of worker iterators (timing-dependent)")
    print("\nThis is why map-style datasets are preferred when deterministic ordering matters.")


if __name__ == "__main__":
    main()
