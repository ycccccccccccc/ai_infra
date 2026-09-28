"""Device-aware batched matrix multiplication benchmark."""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


PRECISIONS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
WEEK1_ROOT = Path(__file__).resolve().parents[3]
WEEK1_RESULTS_DIR = WEEK1_ROOT / "results"


@dataclass
class BenchmarkResult:
    device: str
    device_name: str
    precision: str
    batch: int
    matrix_size: int
    median_runtime_ms: float | None
    throughput_tflops: float | None
    memory_used_mib: float | None
    memory_measurement: str
    status: str


def add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device; auto prefers CUDA, then MPS, then CPU",
    )
    parser.add_argument(
        "--compare-cpu",
        action="store_true",
        help="Also run the same cases on CPU when the selected device is an accelerator",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=list(PRECISIONS),
        default=None,
        help="Defaults to fp32/fp16 on accelerators and fp32 on CPU",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--output", type=Path, default=WEEK1_RESULTS_DIR / "aip_gpu.csv")


def validate_positive(values: list[int], label: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{label} must contain positive integers")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple GPU via MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def default_precisions(device: torch.device) -> list[str]:
    if device.type == "cpu":
        return ["fp32"]
    values = ["fp32", "fp16"]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        values.append("bf16")
    return values


def memory_usage(device: torch.device, tensor_bytes: int) -> tuple[float, str]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20, "peak PyTorch allocation"
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20, "MPS allocation snapshot"
    return tensor_bytes / 2**20, "estimated input/output tensors"


@torch.inference_mode()
def benchmark_case(
    device: torch.device,
    precision: str,
    batch: int,
    size: int,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    dtype = PRECISIONS[precision]
    name = get_device_name(device)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        left = torch.randn(batch, size, size, device=device, dtype=dtype)
        right = torch.randn(batch, size, size, device=device, dtype=dtype)

        for _ in range(warmup):
            output = torch.bmm(left, right)
        sync(device)

        samples: list[float] = []
        for _ in range(iterations):
            sync(device)
            start = time.perf_counter()
            output = torch.bmm(left, right)
            sync(device)
            samples.append(time.perf_counter() - start)

        median_sec = statistics.median(samples)
        operations = 2 * batch * size**3
        tensor_bytes = (left.numel() + right.numel() + output.numel()) * left.element_size()
        memory_mib, memory_kind = memory_usage(device, tensor_bytes)
        result = BenchmarkResult(
            device=device.type,
            device_name=name,
            precision=precision,
            batch=batch,
            matrix_size=size,
            median_runtime_ms=median_sec * 1000,
            throughput_tflops=operations / median_sec / 1e12,
            memory_used_mib=memory_mib,
            memory_measurement=memory_kind,
            status="ok",
        )
        del left, right, output
        clear_memory(device)
        return result
    except (RuntimeError, MemoryError) as error:
        clear_memory(device)
        return BenchmarkResult(
            device=device.type,
            device_name=name,
            precision=precision,
            batch=batch,
            matrix_size=size,
            median_runtime_ms=None,
            throughput_tflops=None,
            memory_used_mib=None,
            memory_measurement="unavailable",
            status=f"{type(error).__name__}: {str(error).splitlines()[0][:160]}",
        )


def format_number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'DEVICE':<7} {'PREC':<5} {'BATCH':>5} {'SIZE':>6} "
        f"{'RUNTIME ms':>12} {'TFLOPS':>10} {'MEM MiB':>10}  STATUS"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.device:<7} {result.precision:<5} {result.batch:>5} "
            f"{result.matrix_size:>6} {format_number(result.median_runtime_ms):>12} "
            f"{format_number(result.throughput_tflops):>10} "
            f"{format_number(result.memory_used_mib, 1):>10}  {result.status}"
        )


def save_results(results: list[BenchmarkResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def run_gpu_command(args: argparse.Namespace) -> int:
    try:
        validate_positive(args.sizes, "sizes")
        validate_positive(args.batches, "batches")
        if args.warmup < 0 or args.iterations <= 0:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        primary = resolve_device(args.device)
    except ValueError as error:
        print(f"error: {error}")
        return 2

    selected_devices = [primary]
    if args.compare_cpu and primary.type != "cpu":
        selected_devices.insert(0, torch.device("cpu"))

    results: list[BenchmarkResult] = []
    print(f"PyTorch: {torch.__version__}")
    for device in selected_devices:
        precisions = args.precisions or default_precisions(device)
        print(f"Device: {get_device_name(device)}; precisions: {', '.join(precisions)}")
        for precision in precisions:
            for batch in args.batches:
                for size in args.sizes:
                    results.append(
                        benchmark_case(
                            device=device,
                            precision=precision,
                            batch=batch,
                            size=size,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                    )

    print()
    print_results(results)
    save_results(results, args.output)
    print(f"\nSaved: {args.output}")
    return 0 if any(result.status == "ok" for result in results) else 1
