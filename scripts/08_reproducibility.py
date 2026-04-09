"""Script 08: Reproducibility -- Seeding and Determinism (Capstone)

Demonstrates:
- Without seeding, shuffled order changes each run
- Using generator= to make the sampler deterministic
- Using worker_init_fn to seed worker subprocesses
- The full reproducibility recipe

Run: uv run python scripts/08_reproducibility.py
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from learn_torch_dataloaders.utils import section_header


class IntegerMapDatasetWithNoise(Dataset):
    """Map-style dataset that adds random noise to demonstrate RNG seeding.

    Each item returns (value, noise) where noise comes from random.random(),
    showing whether worker RNG state is properly controlled.
    """

    def __init__(self, n: int):
        self.data = list(range(n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value = self.data[idx]
        noise = random.random()
        return {"value": value, "noise": round(noise, 4)}


def seed_worker(worker_id):
    """Worker init function that seeds all RNG libraries deterministically.

    PyTorch sets a unique seed for each worker based on:
        base_seed + worker_id
    We can read this via torch.initial_seed() and use it to seed other libraries.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print(f"  Worker {worker_id} seeded with {worker_seed}", flush=True)


def collect_results(loader):
    """Run one epoch and return (values, noises) in order."""
    values = []
    noises = []
    for batch in loader:
        values.extend(batch["value"].tolist())
        noises.extend(batch["noise"].tolist())
    return values, noises


def main():
    n = 50
    batch_size = 8

    # --- Part A: No seeding (non-reproducible) ---
    section_header("Part A: No Seeding -- Non-Reproducible")

    dataset = IntegerMapDatasetWithNoise(n=n)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for run in range(2):
        values, noises = collect_results(loader)
        print(f"Run {run}: first 10 values={values[:10]}")
        print(f"         first 10 noises={noises[:10]}")

    print("\n-> Values in different order, noises differ between runs")
    print("   (shuffle uses a random permutation, random.random() is unseeded)")

    # --- Part B: Seeding the sampler only ---
    section_header("Part B: Seeded Generator -- Reproducible Shuffle Order")

    for run in range(2):
        g = torch.Generator()
        g.manual_seed(42)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g
        )
        values, noises = collect_results(loader)
        print(f"Run {run}: first 10 values={values[:10]}")
        print(f"         first 10 noises={noises[:10]}")

    print("\n-> Values in SAME order (generator seeds the RandomSampler)")
    print("   But noises still differ (random.random() is not seeded by generator)")

    # --- Part C: Full reproducibility with worker_init_fn ---
    section_header("Part C: Full Reproducibility Recipe")
    print("Combining: torch.manual_seed + generator + worker_init_fn\n")

    for run in range(2):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        g = torch.Generator()
        g.manual_seed(42)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=False,
        )

        values, noises = collect_results(loader)
        print(f"Run {run}: first 10 values={values[:10]}")
        print(f"         first 10 noises={noises[:10]}\n")

    print("-> Both values AND noises are identical across runs!")

    # --- Part D: What each piece controls ---
    section_header("Summary: The Reproducibility Recipe")

    print("""
To get fully reproducible DataLoader output:

    torch.manual_seed(42)          # Seeds PyTorch ops in main process
    random.seed(42)                # Seeds Python random in main process
    np.random.seed(42)             # Seeds NumPy random in main process

    g = torch.Generator()
    g.manual_seed(42)              # Seeds the sampler (shuffle order)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=N,
        generator=g,               # Controls shuffle order
        worker_init_fn=seed_worker, # Seeds each worker's RNG
    )

What each piece controls:
- generator      -> shuffle order (which indices go in which batch)
- worker_init_fn -> per-worker RNG (any randomness inside __getitem__)
- torch/random/np.manual_seed -> main process RNG (for num_workers=0)

Without ALL of these, some source of randomness remains unseeded.
""")


if __name__ == "__main__":
    main()
