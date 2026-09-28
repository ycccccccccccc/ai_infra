"""Benchmark a small Transformer across batch, sequence length, and dtype.

The benchmark measures inference only. It does not include data loading,
backward, or optimizer time. OOM/unsupported configurations are written to CSV
instead of terminating the full experiment.

python week1/exercise_b_transformer.py \
  --device mps \
  --batches 1 8 \
  --sequences 128 512 \
  --dtypes fp32 fp16 \
  --warmup 1 \
  --iterations 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--sequences", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--dtypes", nargs="+", choices=list(DTYPES), default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--profile", action="store_true", help="Print a profiler table after benchmarks")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "transformer.csv")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
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


def memory_gib(device: torch.device) -> tuple[float, str]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30, "peak_tensor_allocated"
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**30, "snapshot_tensor_allocated"
    return 0.0, "not_measured"


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence: int,
        d_model: int,
        layers: int,
        heads: int,
        ffn: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_sequence, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence = token_ids.shape[1]
        positions = torch.arange(sequence, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        return self.norm(self.encoder(hidden))


def default_dtypes(device: torch.device) -> list[str]:
    if device.type == "cpu":
        return ["fp32"]
    if device.type == "mps":
        return ["fp32", "fp16"]
    # BF16 is useful only when the CUDA GPU reports support.
    values = ["fp32", "fp16"]
    if torch.cuda.is_bf16_supported():
        values.append("bf16")
    return values


@torch.inference_mode()
def run_case(args: argparse.Namespace, device: torch.device, batch: int, sequence: int, dtype_name: str) -> dict[str, object]:
    dtype = DTYPES[dtype_name]
    model = TinyTransformer(
        vocab_size=args.vocab_size,
        max_sequence=max(args.sequences),
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ffn=args.ffn,
    ).to(device=device, dtype=dtype).eval()
    token_ids = torch.randint(0, args.vocab_size, (batch, sequence), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        output = model(token_ids)
    sync(device)

    samples: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = model(token_ids)
        sync(device)
        samples.append(time.perf_counter() - start)

    latency_sec = statistics.median(samples)
    memory, memory_kind = memory_gib(device)
    result = {
        "device": device.type,
        "batch": batch,
        "sequence": sequence,
        "dtype": dtype_name,
        "median_latency_ms": latency_sec * 1000,
        "tokens_per_second": batch * sequence / latency_sec,
        "memory_gib": memory,
        "memory_measurement": memory_kind,
        "status": "ok",
    }

    del output, token_ids, model
    clear_memory(device)
    return result


def error_row(device: torch.device, batch: int, sequence: int, dtype_name: str, error: Exception) -> dict[str, object]:
    return {
        "device": device.type,
        "batch": batch,
        "sequence": sequence,
        "dtype": dtype_name,
        "median_latency_ms": "",
        "tokens_per_second": "",
        "memory_gib": "",
        "memory_measurement": "",
        "status": f"{type(error).__name__}: {str(error).splitlines()[0][:160]}",
    }


@torch.inference_mode()
def print_profile(args: argparse.Namespace, device: torch.device, dtype_name: str) -> None:
    batch = min(args.batches)
    sequence = min(args.sequences)
    dtype = DTYPES[dtype_name]
    model = TinyTransformer(
        args.vocab_size,
        max(args.sequences),
        args.d_model,
        args.layers,
        args.heads,
        args.ffn,
    ).to(device=device, dtype=dtype).eval()
    token_ids = torch.randint(0, args.vocab_size, (batch, sequence), device=device)

    activities = [ProfilerActivity.CPU]
    sort_key = "self_cpu_time_total"
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        sort_key = "self_cuda_time_total"

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        model(token_ids)
        sync(device)

    print("\nProfiler representative case:", batch, sequence, dtype_name)
    if device.type == "mps":
        print("MPS note: this table measures PyTorch host operators, not Metal kernel duration.")
        print("Use torch.mps.profiler with Xcode Instruments for a Metal timeline.")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype_names = args.dtypes or default_dtypes(device)
    print(f"PyTorch {torch.__version__}; device={device}; dtypes={dtype_names}")

    rows: list[dict[str, object]] = []
    for batch in args.batches:
        for sequence in args.sequences:
            for dtype_name in dtype_names:
                try:
                    row = run_case(args, device, batch, sequence, dtype_name)
                    print(
                        f"batch={batch:2d} seq={sequence:4d} dtype={dtype_name:4s} "
                        f"latency={row['median_latency_ms']:9.3f} ms "
                        f"throughput={row['tokens_per_second']:12.1f} tokens/s "
                        f"memory={row['memory_gib']:.3f} GiB"
                    )
                except (RuntimeError, MemoryError) as error:
                    row = error_row(device, batch, sequence, dtype_name, error)
                    print(f"batch={batch} seq={sequence} dtype={dtype_name}: {row['status']}")
                    clear_memory(device)
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")

    if args.profile:
        print_profile(args, device, dtype_names[0])


if __name__ == "__main__":
    main()
