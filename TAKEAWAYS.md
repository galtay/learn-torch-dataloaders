# Mildly Surprising Facts About PyTorch DataLoaders

A collection of non-obvious behaviors discovered while building this repo.

---

### 1. Map-style datasets always have at most one partial batch, regardless of worker count

With map-style datasets, the `DataLoader` controls batching centrally. The sampler generates a global list of indices, groups them into batches, and distributes complete batches to workers for fetching. Workers are just fetchers — they don't decide what goes into a batch. So no matter how many workers you use, you only get one partial batch (the last one, if `n % batch_size != 0`).

### 2. Iterable datasets can produce up to `num_workers` partial batches

With iterable datasets, each worker runs its own iterator independently. The DataLoader collects items from workers as they arrive and groups them into batches. When a worker exhausts its shard, whatever items it contributed to the current in-progress batch get flushed — potentially as a partial batch. Since each worker can finish at a different time, you can get up to `num_workers` partial batches scattered through the output, not just one at the end.

### 3. `shuffle=True` raises an error for iterable datasets

You might expect the DataLoader to silently ignore `shuffle=True` for iterable datasets or buffer and shuffle the output. Instead, PyTorch raises a `ValueError`. Shuffling must be implemented inside `__iter__` by the dataset itself, because the DataLoader has no way to shuffle an iterator it doesn't control.

### 4. Each worker gets a full copy of the dataset object

With `num_workers > 0`, the dataset is pickled and sent to each worker subprocess. For iterable datasets without sharding logic, this means every worker independently iterates the entire dataset — silently duplicating all your data. There's no warning. You just train on `n * num_workers` samples per epoch instead of `n`.

### 5. Reproducibility requires seeding at three separate levels

Setting `torch.manual_seed(42)` is not enough. Full reproducibility requires:
- A `generator` on the DataLoader (controls shuffle order in the sampler)
- A `worker_init_fn` (seeds each worker subprocess's RNG — Python `random`, NumPy, etc.)
- Global seeds in the main process (`torch.manual_seed`, `random.seed`, `np.random.seed`)

Miss any one of these and some source of randomness remains unseeded.

### 6. Worker prints may not appear where you expect

When `num_workers > 0`, dataset code (`__getitem__` or `__iter__`) runs in forked/spawned subprocesses. Print statements from workers go through subprocess stdout, which may be buffered, interleaved, or appear out of order relative to the main process output. For reliable per-item worker tracking, return the worker ID as part of the data rather than printing it.

### 7. Map-style datasets guarantee deterministic sample order with multiple workers; iterable datasets do not

With map-style datasets, the sampler produces a global index order *before* any workers are involved. Batches are pre-determined (e.g. `[0,1,2], [3,4,5], ...`) and then handed to workers purely for fetching. The DataLoader's output queue (controlled by `in_order=True`, the default) ensures batches are yielded in exactly the sampler's order — even if worker 1 finishes its batch before worker 0. Workers are transparent; they cannot affect sample order.

With iterable datasets, each worker runs its own iterator and produces items independently. The DataLoader assigns tasks to workers in round-robin order and reassembles results in that task order (also via `in_order=True`). However, the *content* of each task depends on each worker's iterator state and shard boundaries. When workers have uneven shard sizes or finish at different times, the interleaving of items across workers can change the sample order. In practice this means: even with deterministic sharding and seeding, the batch contents from iterable datasets can depend on timing and worker scheduling in ways that map-style datasets never do.
