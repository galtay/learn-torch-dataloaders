# DataLoader Architecture

Visual guide to how PyTorch DataLoaders work internally.

---

## Component Overview

The DataLoader is composed of several pluggable components. Which ones are active depends on the dataset type and configuration.

```mermaid
graph LR
    subgraph User Configurable
        DS[Dataset]
        S[Sampler]
        BS[BatchSampler]
        CF[collate_fn]
    end

    subgraph DataLoader Internals
        IT[Iterator]
        F[Fetcher]
        Q[Worker Queues]
        PM[pin_memory]
    end

    DS --> F
    S --> BS
    BS --> IT
    IT --> F
    F --> CF
    CF --> Q
    Q --> PM
    PM --> OUT[Batches]
```

| Component | Role | Default |
|-----------|------|---------|
| **Dataset** | Holds or generates the data | User-provided |
| **Sampler** | Produces indices (map-style only) | `SequentialSampler` or `RandomSampler` |
| **BatchSampler** | Groups sampler indices into batches | `BatchSampler(sampler, batch_size)` |
| **Fetcher** | Retrieves data from dataset | `_MapDatasetFetcher` or `_IterableDatasetFetcher` |
| **collate_fn** | Converts list of samples into a batch | `default_collate` (stacks tensors) |
| **pin_memory** | Copies batch to pinned (page-locked) memory for faster GPU transfer | Disabled |

---

## Map-Style: Single Worker (`num_workers=0`)

Everything runs in the main process. The sampler controls access order.

```mermaid
flowchart TD
    A["DataLoader.__iter__()"] --> B["_SingleProcessDataLoaderIter"]

    B --> C["Sampler<br/>(SequentialSampler or RandomSampler)"]
    C -->|"indices: [4, 2, 0, ...]"| D["BatchSampler"]
    D -->|"batch indices: [4, 2, 0]"| E["_MapDatasetFetcher"]

    E --> F["dataset[4]"]
    E --> G["dataset[2]"]
    E --> H["dataset[0]"]

    F --> I["collate_fn([s4, s2, s0])"]
    G --> I
    H --> I

    I --> J["Batch Tensor"]

    style C fill:#4a9eff,color:#fff
    style E fill:#ff9f43,color:#fff
    style I fill:#2ed573,color:#fff
```

**Key point:** The Sampler decides the order. The Fetcher just calls `dataset[idx]` for each index in the batch.

---

## Map-Style: Multiple Workers (`num_workers>0`)

Workers fetch data in parallel, but the output queue preserves sampler order.

```mermaid
flowchart TD
    A["DataLoader.__iter__()"] --> B["_MultiProcessingDataLoaderIter"]

    B --> C["Sampler → BatchSampler"]
    C -->|"(0, [idx, idx, idx])<br/>(1, [idx, idx, idx])<br/>..."| D["index_queues<br/>(one per worker)"]

    subgraph "Worker 0 (subprocess)"
        W0["_worker_loop"]
        F0["_MapDatasetFetcher"]
        W0 --> F0
        F0 -->|"dataset[idx]"| C0["collate_fn"]
    end

    subgraph "Worker 1 (subprocess)"
        W1["_worker_loop"]
        F1["_MapDatasetFetcher"]
        W1 --> F1
        F1 -->|"dataset[idx]"| C1["collate_fn"]
    end

    D -->|"batch 0 indices"| W0
    D -->|"batch 1 indices"| W1

    C0 -->|"(0, batch_data)"| RQ["worker_result_queue<br/>(shared)"]
    C1 -->|"(1, batch_data)"| RQ

    RQ --> REORDER["Reorder by batch number<br/>(in_order=True)"]
    REORDER --> OUT["Batches in sampler order"]

    style C fill:#4a9eff,color:#fff
    style REORDER fill:#ff6b6b,color:#fff
    style RQ fill:#ffa502,color:#fff
```

**Key point:** Each batch is tagged with a batch number. The main process yields batches in order of that number, even if Worker 1 finishes before Worker 0. This is why map-style ordering is deterministic regardless of worker timing.

---

## Iterable-Style: Single Worker (`num_workers=0`)

The dataset controls iteration. The sampler is a dummy that just counts batch slots.

```mermaid
flowchart TD
    A["DataLoader.__iter__()"] --> B["_SingleProcessDataLoaderIter"]

    B --> C["_InfiniteConstantSampler<br/>(yields None forever)"]
    C -->|"[None, None, None]<br/>(batch_size times)"| D["_IterableDatasetFetcher"]

    D --> E["dataset_iter = iter(dataset)"]

    E --> F["next(dataset_iter)"]
    E --> G["next(dataset_iter)"]
    E --> H["next(dataset_iter)"]

    F --> I["collate_fn([s0, s1, s2])"]
    G --> I
    H --> I

    I --> J["Batch Tensor"]

    style C fill:#a55eea,color:#fff
    style D fill:#ff9f43,color:#fff
    style I fill:#2ed573,color:#fff
```

**Key point:** No real sampler. The Fetcher just calls `next()` on the dataset's iterator `batch_size` times to fill each batch. The dataset decides what comes out.

---

## Iterable-Style: Multiple Workers (`num_workers>0`)

Each worker gets its own copy of the dataset and its own iterator.

```mermaid
flowchart TD
    A["DataLoader.__iter__()"] --> B["_MultiProcessingDataLoaderIter"]

    B --> C["_InfiniteConstantSampler<br/>→ BatchSampler"]
    C -->|"(0, [None, None, None])<br/>(1, [None, None, None])<br/>..."| D["index_queues"]

    subgraph "Worker 0 (subprocess)"
        W0["_worker_loop"]
        DS0["iter(dataset_copy_0)"]
        F0["_IterableDatasetFetcher"]
        W0 --> F0
        F0 -->|"next() × batch_size"| DS0
        DS0 -->|"items from shard 0"| C0["collate_fn"]
    end

    subgraph "Worker 1 (subprocess)"
        W1["_worker_loop"]
        DS1["iter(dataset_copy_1)"]
        F1["_IterableDatasetFetcher"]
        W1 --> F1
        F1 -->|"next() × batch_size"| DS1
        DS1 -->|"items from shard 1"| C1["collate_fn"]
    end

    D -->|"task 0"| W0
    D -->|"task 1"| W1

    C0 -->|"(0, batch_data)"| RQ["worker_result_queue<br/>(shared)"]
    C1 -->|"(1, batch_data)"| RQ

    RQ --> REORDER["Reorder by task number"]
    REORDER --> OUT["Batches"]

    style C fill:#a55eea,color:#fff
    style DS0 fill:#ff6348,color:#fff
    style DS1 fill:#ff6348,color:#fff
    style RQ fill:#ffa502,color:#fff
```

**Key point:** Each worker has an independent copy of the dataset and iterator. Without sharding logic in `__iter__`, every worker yields the full dataset (data duplication). The task-number reordering preserves the round-robin assignment order, but the *content* depends on what each worker's iterator yields — which can be timing-dependent.

---

## Map-Style vs Iterable-Style: Summary

```mermaid
flowchart LR
    subgraph Map-Style
        direction TB
        MS_S["Sampler<br/>controls order"] --> MS_BS["BatchSampler<br/>groups indices"]
        MS_BS --> MS_F["Fetcher<br/>calls dataset[idx]"]
        MS_F --> MS_C["collate_fn"]
    end

    subgraph Iterable-Style
        direction TB
        IS_S["_InfiniteConstantSampler<br/>(dummy)"] --> IS_BS["BatchSampler<br/>counts slots"]
        IS_BS --> IS_F["Fetcher<br/>calls next(iter)"]
        IS_F --> IS_C["collate_fn"]
    end

    style MS_S fill:#4a9eff,color:#fff
    style IS_S fill:#a55eea,color:#fff
```

|  | Map-Style | Iterable-Style |
|--|-----------|----------------|
| **Who controls order?** | Sampler (external to dataset) | Dataset's `__iter__` (internal) |
| **Shuffling** | `shuffle=True` on DataLoader | Must implement in `__iter__` |
| **Multi-worker** | Automatic index distribution | Must shard manually |
| **Deterministic order** | Yes (sampler + output queue) | Depends on worker timing |
| **`len()` support** | Yes (`__len__`) | No |
