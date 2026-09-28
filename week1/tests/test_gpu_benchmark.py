from __future__ import annotations

import unittest

import torch

from aip.benchmarks.gpu import benchmark_case, resolve_device, validate_positive


class GpuBenchmarkTests(unittest.TestCase):
    def test_auto_device_is_valid(self) -> None:
        self.assertIn(resolve_device("auto").type, {"cpu", "mps", "cuda"})

    def test_positive_validation(self) -> None:
        validate_positive([1, 8, 32], "batch")
        with self.assertRaises(ValueError):
            validate_positive([0], "batch")

    def test_tiny_cpu_case(self) -> None:
        result = benchmark_case(
            device=torch.device("cpu"),
            precision="fp32",
            batch=1,
            size=16,
            warmup=1,
            iterations=2,
        )
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.median_runtime_ms or 0, 0)
        self.assertGreater(result.memory_used_mib or 0, 0)


if __name__ == "__main__":
    unittest.main()

