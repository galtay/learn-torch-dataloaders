"""Script 07: Partial Batch Handling with drop_last

Demonstrates:
- drop_last=False (default): the last batch may be smaller than batch_size
- drop_last=True: the incomplete final batch is discarded
- Behavior with both map-style and iterable-style datasets

Run: uv run python scripts/07_drop_last.py
"""

from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import IntegerMapDataset, ShardedIntegerIterableDataset
from learn_torch_dataloaders.utils import print_batch, section_header


def run_drop_last_demo(dataset, dataset_name: str, batch_size: int, drop_last: bool):
    label = f"{dataset_name}, batch_size={batch_size}, drop_last={drop_last}"
    section_header(label)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=drop_last)

    total_items = 0
    num_batches = 0
    for batch_idx, batch in enumerate(loader):
        print_batch(batch_idx, batch)
        total_items += len(batch)
        num_batches += 1

    print(f"\nTotal batches: {num_batches}")
    print(f"Total items: {total_items}")
    return num_batches, total_items


def main():
    n = 50
    batch_size = 8

    print(f"Dataset: integers 0..{n - 1} (size={n})")
    print(f"Batch size: {batch_size}")
    print(f"Full batches: {n // batch_size}, Remainder: {n % batch_size} items\n")

    # --- Map-style ---
    map_dataset = IntegerMapDataset(n=n)

    _, items_kept = run_drop_last_demo(map_dataset, "Map-Style", batch_size, drop_last=False)
    _, items_dropped = run_drop_last_demo(map_dataset, "Map-Style", batch_size, drop_last=True)

    section_header("Comparison")
    print(f"drop_last=False: {items_kept} items across all batches")
    print(f"drop_last=True:  {items_dropped} items across all batches")
    print(f"Items dropped: {items_kept - items_dropped}")

    # --- Iterable-style ---
    iter_dataset = ShardedIntegerIterableDataset(n=n)
    run_drop_last_demo(iter_dataset, "Iterable-Style", batch_size, drop_last=False)
    run_drop_last_demo(iter_dataset, "Iterable-Style", batch_size, drop_last=True)

    print("\n--- Key Takeaways ---")
    print("1. drop_last=False (default): keeps the partial final batch")
    print("2. drop_last=True: discards it -- some data is never seen")
    print("3. This works the same for map-style and iterable-style datasets")
    print("4. Common use case: drop_last=True during training to ensure")
    print("   uniform batch sizes (important for batch normalization)")
    print("5. Usually set drop_last=False for validation/test to see all data")


if __name__ == "__main__":
    main()
