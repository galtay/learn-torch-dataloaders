"""Script 01: Map-Style Dataset Basics

Demonstrates:
- The Dataset protocol: __len__ and __getitem__
- How DataLoader calls __getitem__ with sequential indices
- Default batching behavior
- Partial last batch when dataset size isn't divisible by batch_size

Run: uv run python scripts/01_map_style_basic.py
"""

from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import IntegerMapDataset
from learn_torch_dataloaders.utils import print_batch, section_header


def main():
    n = 50
    batch_size = 8

    section_header("Map-Style Dataset, Single Worker (num_workers=0)")

    dataset = IntegerMapDataset(n=n)
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Expected full batches: {n // batch_size}, remainder: {n % batch_size}\n")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    total_items = 0
    for batch_idx, batch in enumerate(loader):
        print_batch(batch_idx, batch)
        total_items += len(batch)

    print(f"\nTotal batches: {batch_idx + 1}")
    print(f"Total items: {total_items}")

    print("\n--- Key Takeaways ---")
    print("1. DataLoader calls __getitem__(0), __getitem__(1), ... sequentially")
    print("2. Items are grouped into batches of size batch_size")
    print(f"3. The last batch has only {n % batch_size} items (n % batch_size)")
    print("4. With num_workers=0, everything runs in the main process")


if __name__ == "__main__":
    main()
