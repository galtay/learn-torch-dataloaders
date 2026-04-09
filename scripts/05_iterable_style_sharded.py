"""Script 05: Iterable-Style Dataset with Proper Sharding

Demonstrates:
- Using get_worker_info() to partition data across workers
- Each worker yields a non-overlapping slice of the data
- Balanced sharding: worker shard sizes differ by at most 1
- The partial-batch effect: only the very last batch can be partial,
  but more workers means more potential for partial final batches

Run: uv run python scripts/05_iterable_style_sharded.py
"""

from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import ShardedIntegerIterableDataset, _worker_shard
from learn_torch_dataloaders.utils import print_batch, section_header


def run_sharded(n: int, batch_size: int, num_workers: int):
    section_header(f"Sharded Iterable, n={n}, batch_size={batch_size}, {num_workers} Workers")

    # Show the shard assignment
    if num_workers > 0:
        print("Shard assignment:")
        for wid in range(num_workers):
            start, end = _worker_shard(n, wid, num_workers)
            print(f"  Worker {wid}: items [{start}, {end}) -> {end - start} items")
        print()

    dataset = ShardedIntegerIterableDataset(n=n)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    all_items = []
    batch_sizes = []
    for batch_idx, batch in enumerate(loader):
        items = batch.tolist()
        all_items.extend(items)
        batch_sizes.append(len(items))
        print_batch(batch_idx, batch)

    print(f"\nTotal items: {len(all_items)} (expected {n})")
    unique = set(all_items)
    print(f"Unique items: {len(unique)}, Duplicates: {len(all_items) - len(unique)}")
    print(f"Batch sizes: {batch_sizes}")


def main():
    n = 50
    batch_size = 8

    # Single worker (baseline)
    run_sharded(n=n, batch_size=batch_size, num_workers=0)

    # 2 workers
    run_sharded(n=n, batch_size=batch_size, num_workers=2)

    # 3 workers
    run_sharded(n=n, batch_size=batch_size, num_workers=3)

    print("\n")
    section_header("How Sharding Works")
    print("1. Inside __iter__, call get_worker_info()")
    print("2. If None -> single worker mode, yield everything")
    print("3. Otherwise, compute this worker's slice using divmod:")
    print("   base, remainder = divmod(n, num_workers)")
    print("   First `remainder` workers get (base + 1) items, rest get base")
    print("4. This ensures shard sizes differ by at most 1\n")

    section_header("Partial Batch Behavior")
    print("With a single worker, only the very last batch can be partial")
    print(
        f"(here: {n} items / {batch_size} batch_size = "
        f"{n // batch_size} full + 1 partial of {n % batch_size}).\n"
    )
    print("With multiple workers, each worker's shard is yielded independently.")
    print("When a worker exhausts its shard, it can cause a partial batch")
    print("at the boundary. More workers = more boundaries = more potential")
    print("for partial batches. This is an inherent tradeoff of iterable-style")
    print("datasets with multiprocessing.")


if __name__ == "__main__":
    main()
