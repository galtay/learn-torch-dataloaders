"""Script 06: Shuffling Behavior -- Map vs Iterable

Demonstrates:
- Map-style: shuffle=True permutes indices via RandomSampler
- Map-style: shuffle order changes each epoch
- Iterable-style: shuffle=True on the DataLoader raises an error
- Iterable-style: shuffling must be implemented inside __iter__

Run: uv run python scripts/06_shuffle_behavior.py
"""

from torch.utils.data import DataLoader

from learn_torch_dataloaders.datasets import (
    IntegerIterableDataset,
    IntegerMapDataset,
    ShuffleableIterableDataset,
)
from learn_torch_dataloaders.utils import section_header


def collect_batches(loader):
    """Collect all batches as lists."""
    batches = []
    for batch in loader:
        batches.append(batch.tolist())
    return batches


def main():
    n = 50
    batch_size = 8

    # --- Part A: Map-style, no shuffle ---
    section_header("Part A: Map-Style, shuffle=False")
    dataset = IntegerMapDataset(n=n)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    for epoch in range(2):
        batches = collect_batches(loader)
        flat = [item for b in batches for item in b]
        print(f"Epoch {epoch}: {flat}")
    print("-> Same order every epoch\n")

    # --- Part B: Map-style, shuffle ---
    section_header("Part B: Map-Style, shuffle=True")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    for epoch in range(3):
        batches = collect_batches(loader)
        flat = [item for b in batches for item in b]
        print(f"Epoch {epoch}: {flat}")
    print("-> Different order each epoch (RandomSampler permutes indices)\n")

    # --- Part C: Iterable-style, shuffle raises error ---
    section_header("Part C: Iterable-Style, shuffle is NOT supported")
    print("DataLoader(IterableDataset, shuffle=True) is not allowed.")
    print("PyTorch raises an error because the DataLoader cannot shuffle")
    print("an iterator it doesn't control.\n")

    iter_dataset = IntegerIterableDataset(n=n)
    try:
        loader = DataLoader(iter_dataset, batch_size=batch_size, shuffle=True)
        for _batch in loader:
            pass
    except ValueError as e:
        print(f"ValueError: {e}\n")

    # --- Part D: Shuffling inside __iter__ ---
    section_header("Part D: Shuffleable Iterable Dataset (correct approach)")
    print("Shuffling must happen INSIDE __iter__, controlled by the dataset.\n")

    for seed in [42, 42, 99]:
        shuf_dataset = ShuffleableIterableDataset(n=n, shuffle=True, seed=seed)
        loader = DataLoader(shuf_dataset, batch_size=batch_size, num_workers=0)
        batches = collect_batches(loader)
        flat = [item for b in batches for item in b]
        print(f"seed={seed}: {flat}")

    print("\n-> Same seed = same order (reproducible)")
    print("-> Different seed = different order")

    print("\n--- Key Takeaways ---")
    print("1. Map-style: shuffle=True uses RandomSampler to permute indices")
    print("2. Iterable-style: shuffle=True raises an error")
    print("3. For iterable datasets, implement shuffling inside __iter__")
    print("4. Use a seed parameter to make shuffling reproducible")


if __name__ == "__main__":
    main()
