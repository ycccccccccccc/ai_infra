"""CPU/GPU matrix multiplication benchmark.

Examples:
    python exercise_cpu_mps_matmul.py
    python exercise_cpu_mps_matmul.py --sizes 128 512 2048
    python exercise_cpu_mps_matmul.py --devices cpu mps
    python exercise_cpu_mps_matmul.py --devices cpu cuda

This is a compute-only benchmark: matrices are created on the target device before
timing starts. GPU transfer time is therefore excluded.

python week1/exercise_cpu_mps_matmul.py \
  --devices cpu mps \
  --sizes 128 512 \
  --warmup 1 \
  --trials 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import statistics
import time
from pathlib import Path

import torch


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "matmul.csv")
    return parser.parse_args()


def available_devices(requested: list[str] | None) -> list[torch.device]:
    names = requested or ["cpu", "mps"]
    devices: list[torch.device] = []
    for name in names:
        if name == "mps" and not torch.backends.mps.is_available():
            print("Skip mps: MPS is unavailable")
            continue
        if name == "cuda" and not torch.cuda.is_available():
            print("Skip cuda: CUDA is unavailable")
            continue
        devices.append(torch.device(name))
    return devices


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple GPU via MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def inner_loops(size: int) -> int:
    # Tiny operations need more repetitions per timing sample.
    if size <= 256:
        return 500
    if size <= 512:
        return 100
    if size <= 1024:
        return 20
    if size <= 2048:
        return 5
    return 1


@torch.inference_mode()
def benchmark(size: int, device: torch.device, warmup: int, trials: int) -> dict[str, object]:
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    for _ in range(warmup):
        _ = a @ b
    sync(device)

    loops = inner_loops(size)
    samples: list[float] = []
    result = None

    for _ in range(trials):
        sync(device)
        start = time.perf_counter()
        for _ in range(loops):
            result = a @ b
        sync(device)
        samples.append((time.perf_counter() - start) / loops)

    median_sec = statistics.median(samples)
    gflops = (2 * size**3) / median_sec / 1e9

    del a, b, result
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "matrix_size": size,
        "device": device.type,
        "device_name": device_name(device),
        "dtype": "float32",
        "median_ms": median_sec * 1000,
        "achieved_gflops": gflops,
        "trials": trials,
        "inner_loops": loops,
    }


def main() -> None:
    args = parse_args()
    devices = available_devices(args.devices)
    if not devices:
        raise SystemExit("No requested device is available")

    print(f"PyTorch {torch.__version__}")
    rows: list[dict[str, object]] = []

    for size in args.sizes:
        by_device: dict[str, float] = {}
        for device in devices:
            row = benchmark(size, device, args.warmup, args.trials)
            rows.append(row)
            by_device[device.type] = float(row["median_ms"])
            print(
                f"size={size:5d} device={device.type:4s} "
                f"time={row['median_ms']:10.4f} ms "
                f"performance={row['achieved_gflops']:10.2f} GFLOPS"
            )

        accelerator = "mps" if "mps" in by_device else "cuda"
        if "cpu" in by_device and accelerator in by_device:
            speedup = by_device["cpu"] / by_device[accelerator]
            print(f"{accelerator.upper()} speedup over CPU: {speedup:.2f}x")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
