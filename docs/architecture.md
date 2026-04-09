# DataLoader Architecture

Visual guide to how PyTorch DataLoaders work internally.

---

## Component Overview

```mermaid
graph LR
    DS["Dataset"] --> F["Fetcher"]
    S["Sampler"] --> BS["BatchSampler"]
    BS --> F
    F --> CF["CollateFunction"]
    CF --> WQ["WorkerQueues"]
    WQ --> OUT["Batches"]
```

### Map-Style Components

| Component | Class | Role |
|-----------|-------|------|
| **Dataset** | User-provided | Defines `__getitem__` and `__len__` |
| **Sampler** | `SequentialSampler` (`RandomSampler` if `shuffle=True`) | Produces indices that control access order |
| **BatchSampler** | `BatchSampler` | Groups sampler indices into lists of `batch_size` |
| **Fetcher** | `_MapDatasetFetcher` | Calls `dataset[idx]` for each index in a batch |
| **CollateFunction** | `default_collate` | Converts list of samples into a batch tensor |
| **WorkerQueues** | `index_queue` + `worker_result_queue` | Distributes work to and collects results from workers |

### Iterable-Style Components

| Component | Class | Role |
|-----------|-------|------|
| **Dataset** | User-provided | Defines `__iter__` |
| **Sampler** | `_InfiniteConstantSampler` | Dummy sampler that yields `None` forever |
| **BatchSampler** | `BatchSampler` | Counts how many items to pull per batch |
| **Fetcher** | `_IterableDatasetFetcher` | Calls `next(dataset_iter)` x `batch_size` to fill a batch |
| **CollateFunction** | `default_collate` | Converts list of samples into a batch tensor |
| **WorkerQueues** | `index_queue` + `worker_result_queue` | Distributes work to and collects results from workers |

---

## Map-Style: Single Worker (`num_workers=0`)

Everything runs in the main process. The Sampler controls access order.

```mermaid
flowchart TD
    S["Sampler"] -->|"indices: [4, 2, 0, ...]"| BS["BatchSampler"]
    BS -->|"[4, 2, 0]"| F["Fetcher"]
    DS["Dataset"] --> F
    F -->|"dataset[4], dataset[2], dataset[0]"| CF["CollateFunction"]
    CF --> OUT["Batch"]
```

**Key point:** The Sampler decides the order. The Fetcher just calls `dataset[idx]` for each index in the batch.

---

## Map-Style: Multiple Workers (`num_workers>0`)

Workers fetch data in parallel, but the output queue preserves Sampler order.

```mermaid
flowchart TD
    S["Sampler"] --> BS["BatchSampler"]

    subgraph "Worker 0"
        direction LR
        IQ0["index_queue"] --> DS0["Dataset"] --> F0["Fetcher"]
        F0 -->|"dataset[idx]"| CF0["CollateFunction"]
    end

    subgraph "Worker 1"
        direction LR
        IQ1["index_queue"] --> DS1["Dataset"] --> F1["Fetcher"]
        F1 -->|"dataset[idx]"| CF1["CollateFunction"]
    end

    BS -->|"batch 0 indices"| IQ0
    BS -->|"batch 1 indices"| IQ1

    CF0 -->|"(0, batch_data)"| RQ["worker_result_queue"]
    CF1 -->|"(1, batch_data)"| RQ

    RQ -->|"reorder by batch number"| OUT["Batches"]
```

**Key point:** Each worker has its own `index_queue` and a copy of the Dataset. Each batch is tagged with a batch number. The main process yields batches in order of that number via the shared `worker_result_queue`, even if Worker 1 finishes before Worker 0. This is why map-style ordering is deterministic regardless of worker timing.

---

## Iterable-Style: Single Worker (`num_workers=0`)

The Dataset controls iteration. The Sampler is a dummy that just counts batch slots.

```mermaid
flowchart TD
    S["Sampler"] -->|"[None, None, None]<br/>(batch_size times)"| F["Fetcher"]
    DS["Dataset"] -->|"iter(dataset)"| F
    F -->|"next(), next(), next()"| CF["CollateFunction"]
    CF --> OUT["Batch"]
```

**Key point:** No real Sampler. The Fetcher calls `next()` on the Dataset's iterator `batch_size` times to fill each batch. The Dataset decides what comes out.

---

## Iterable-Style: Multiple Workers (`num_workers>0`)

Each worker gets its own copy of the Dataset and its own iterator.

```mermaid
flowchart TD
    S["Sampler"] --> BS["BatchSampler"]

    subgraph "Worker 0"
        direction LR
        IQ0["index_queue"] --> DS0["Dataset"] -->|"iter(dataset)"| F0["Fetcher"]
        F0 -->|"next() x batch_size"| CF0["CollateFunction"]
    end

    subgraph "Worker 1"
        direction LR
        IQ1["index_queue"] --> DS1["Dataset"] -->|"iter(dataset)"| F1["Fetcher"]
        F1 -->|"next() x batch_size"| CF1["CollateFunction"]
    end

    BS -->|"task 0"| IQ0
    BS -->|"task 1"| IQ1

    CF0 -->|"(0, batch_data)"| RQ["worker_result_queue"]
    CF1 -->|"(1, batch_data)"| RQ

    RQ -->|"reorder by task number"| OUT["Batches"]
```

**Key point:** Each worker has an independent copy of the Dataset and iterator. Without sharding logic in `__iter__`, every worker yields the full dataset (data duplication). The task-number reordering preserves the round-robin assignment order, but the *content* depends on what each worker's iterator yields — which can be timing-dependent.

---

## Map-Style vs Iterable-Style: Summary

|  | Map-Style | Iterable-Style |
|--|-----------|----------------|
| **Who controls order?** | Sampler (external to Dataset) | Dataset's `__iter__` (internal) |
| **Shuffling** | `shuffle=True` on DataLoader | Must implement in `__iter__` |
| **Multi-worker** | Automatic index distribution | Must shard manually |
| **Deterministic order** | Yes (Sampler + output queue) | Depends on worker timing |
| **`len()` support** | Yes (`__len__`) | No |
