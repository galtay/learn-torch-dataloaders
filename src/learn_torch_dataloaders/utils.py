def section_header(title: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'=' * 60}\n", flush=True)


def print_batch(batch_idx: int, batch, epoch: int | None = None) -> None:
    items = batch.tolist() if hasattr(batch, "tolist") else list(batch)
    prefix = f"Epoch {epoch} | " if epoch is not None else ""
    print(f"{prefix}Batch {batch_idx}: {items} (size={len(items)})", flush=True)
