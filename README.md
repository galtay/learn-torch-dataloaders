# learn-torch-dataloaders

Learn how PyTorch DataLoaders work through simple, reproducible examples. Each numbered script builds on previous concepts, using integer datasets (0..N-1) so you can see exactly what's happening at every step.

## Setup

```bash
uv sync
```

## Running Scripts

```bash
uv run python scripts/01_map_style_basic.py
```

## Learning Progression

| Script | Topic | What You'll Learn |
|--------|-------|-------------------|
| 01 | Map-style basics | `__len__`/`__getitem__` protocol, sequential batching |
| 02 | Iterable-style basics | `__iter__` protocol, contrast with map-style |
| 03 | Map-style multi-worker | Worker processes, batch queue ordering, worker ID tracking |
| 04 | Iterable multi-worker (bug) | Data duplication when iterables aren't sharded |
| 05 | Iterable sharding (fix) | Using `get_worker_info()` to partition data across workers |
| 06 | Shuffle behavior | Map-style shuffle vs iterable-style (must implement yourself) |
| 07 | drop_last | Partial batch handling, `drop_last=True` vs `False` |
| 08 | Reproducibility | Seeding the sampler, worker RNG, full reproducibility recipe |

## Key Concepts

- **Map-style datasets** implement `__len__` + `__getitem__`. The DataLoader controls indexing, shuffling, and distribution to workers.
- **Iterable-style datasets** implement `__iter__`. The dataset controls iteration order. With multiple workers, you must shard data manually.
- **Reproducibility** requires seeding at three levels: the DataLoader's sampler (`generator`), each worker subprocess (`worker_init_fn`), and the main process (`torch.manual_seed`).
