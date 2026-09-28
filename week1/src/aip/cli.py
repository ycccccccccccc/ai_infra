"""Command-line entry point for aip."""

from __future__ import annotations

import argparse

from aip.benchmarks.gpu import add_gpu_arguments, run_gpu_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aip", description="AI infrastructure learning tools")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    gpu = benchmark_commands.add_parser(
        "gpu",
        help="Benchmark batched matrix multiplication by device, precision, size, and batch",
    )
    add_gpu_arguments(gpu)
    gpu.set_defaults(handler=run_gpu_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()

