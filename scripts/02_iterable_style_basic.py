"""Script 02: Iterable-Style Dataset Basics

Demonstrates:
- The IterableDataset protocol: __iter__
- How DataLoader consumes items from the iterator
- With a single worker, behavior looks identical to map-style

Run: uv run python scripts/02_iterable_style_basic.py
"""

from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import IntegerIterableDataset
from learn_torch_dataloaders.utils import print_batch, section_header


def main():
    n = 50
    batch_size = 8

    section_header("Iterable-Style Dataset, Single Worker (num_workers=0)")

    dataset = IntegerIterableDataset(n=n)
    print(f"Dataset yields values 0..{n - 1}")
    print(f"Batch size: {batch_size}\n")

    # Note: len() is not available for iterable datasets
    try:
        print(f"len(dataset) = {len(dataset)}")
    except TypeError as e:
        print(f"len(dataset) -> TypeError: {e}")
        print("(Iterable datasets don't support __len__)\n")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    total_items = 0
    for batch_idx, batch in enumerate(loader):
        print_batch(batch_idx, batch)
        total_items += len(batch)

    print(f"\nTotal batches: {batch_idx + 1}")
    print(f"Total items: {total_items}")

    print("\n--- Key Takeaways ---")
    print("1. IterableDataset uses __iter__ instead of __getitem__")
    print("2. The DataLoader pulls items from the iterator and batches them")
    print("3. With a single worker, output looks the same as map-style")
    print("4. len() is not available -- the DataLoader doesn't know the size upfront")
    print("5. The difference becomes critical with multiple workers (see script 04)")


if __name__ == "__main__":
    main()
