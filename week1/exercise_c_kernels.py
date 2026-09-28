"""Compare matmul, element-wise, and embedding lookup workloads.

The printed throughput is an estimate based on useful operations/bytes. It is
intended for relative comparison rather than a hardware peak specification.

python week1/exercise_c_kernels.py \
  --devices cpu mps \
  --dtype fp32 \
  --matmul-size 1024 \
  --element-count 1048576 \
  --embedding-rows 16384 \
  --embedding-dim 128 \
  --lookup-count 8192 \
  --warmup 1 \
  --trials 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--element-count", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--embedding-rows", type=int, default=131072)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lookup-count", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "kernels.csv")
    return parser.parse_args()


def devices(requested: list[str] | None) -> list[torch.device]:
    names = requested or ["cpu", "mps"]
    output = []
    for name in names:
        if name == "mps" and not torch.backends.mps.is_available():
            print("Skip mps: unavailable")
        elif name == "cuda" and not torch.cuda.is_available():
            print("Skip cuda: unavailable")
        else:
            output.append(torch.device(name))
    return output


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def time_operation(operation: Callable[[], torch.Tensor], device: torch.device, warmup: int, trials: int) -> float:
    result = None
    for _ in range(warmup):
        result = operation()
    sync(device)

    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        result = operation()
        sync(device)
        samples.append(time.perf_counter() - start)
    del result
    return statistics.median(samples)


def benchmark_matmul(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    n = args.matmul_size
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    seconds = time_operation(lambda: a @ b, device, args.warmup, args.trials)
    row = {
        "kernel": "matmul",
        "device": device.type,
        "dtype": args.dtype,
        "shape": f"{n}x{n} @ {n}x{n}",
        "median_ms": seconds * 1000,
        "metric": "TFLOPS",
        "throughput": 2 * n**3 / seconds / 1e12,
        "prediction": "compute-bound when sufficiently large and square",
    }
    del a, b
    return row


def benchmark_elementwise(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    count = args.element_count
    x = torch.randn(count, device=device, dtype=dtype)
    y = torch.randn_like(x)
    seconds = time_operation(lambda: torch.relu(x + y), device, args.warmup, args.trials)
    # Eager add + ReLU: read x/y, write/read intermediate, write output.
    estimated_bytes = 5 * count * x.element_size()
    row = {
        "kernel": "add_relu",
        "device": device.type,
        "dtype": args.dtype,
        "shape": str(count),
        "median_ms": seconds * 1000,
        "metric": "estimated_GB/s",
        "throughput": estimated_bytes / seconds / 1e9,
        "prediction": "memory-bound when large; launch-bound when small",
    }
    del x, y
    return row


def benchmark_embedding(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    table = torch.randn(args.embedding_rows, args.embedding_dim, device=device, dtype=dtype)
    ids = torch.randint(0, args.embedding_rows, (args.lookup_count,), device=device)
    seconds = time_operation(lambda: table[ids], device, args.warmup, args.trials)
    output_bytes = args.lookup_count * args.embedding_dim * table.element_size()
    estimated_bytes = 2 * output_bytes + args.lookup_count * ids.element_size()
    row = {
        "kernel": "random_embedding_lookup",
        "device": device.type,
        "dtype": args.dtype,
        "shape": f"table={args.embedding_rows}x{args.embedding_dim}; lookups={args.lookup_count}",
        "median_ms": seconds * 1000,
        "metric": "estimated_GB/s",
        "throughput": estimated_bytes / seconds / 1e9,
        "prediction": "memory-bound; sensitive to cache locality and gather pattern",
    }
    del table, ids
    return row


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    rows: list[dict[str, object]] = []

    print(f"PyTorch {torch.__version__}; dtype={args.dtype}")
    for device in devices(args.devices):
        for benchmark in (benchmark_matmul, benchmark_elementwise, benchmark_embedding):
            try:
                row = benchmark(args, device, dtype)
                rows.append(row)
                print(
                    f"{row['kernel']:24s} device={device.type:4s} "
                    f"time={row['median_ms']:10.3f} ms "
                    f"{row['metric']}={row['throughput']:.3f}"
                )
            except (RuntimeError, MemoryError) as error:
                print(f"{benchmark.__name__} on {device}: {type(error).__name__}: {error}")
            finally:
                cleanup(device)

    if not rows:
        raise SystemExit("No benchmark completed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
