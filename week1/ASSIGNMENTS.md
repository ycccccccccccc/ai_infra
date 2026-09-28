# Week 1 Implementation Assignments

## 1. Build CPU vs GPU matrix benchmark

**Goal:** Implement matrix multiplication benchmark across CPU/GPU and compare execution time by matrix size.

**Implementation:** `exercise_cpu_mps_matmul.py`

Run locally on Mac:

```bash
python exercise_cpu_mps_matmul.py --devices cpu mps
```

Run on Kaggle/Colab:

```bash
python exercise_cpu_mps_matmul.py --devices cpu cuda
```

**Expected output:** `week1/results/matmul.csv`（從 repository root 觀察）

**Acceptance criteria:**

- Test at least six matrix sizes.
- Warm up before measurement and synchronize GPU execution.
- Report median runtime, achieved GFLOPS, and accelerator speedup over CPU.
- Identify the CPU/GPU crossover point.

## 2. Benchmark precision and batch size

**Goal:** Compare FP32 with FP16 or BF16 and multiple batch sizes; record latency, throughput, and memory.

**Implementation:** `exercise_b_transformer.py` and `aip benchmark gpu`

Transformer experiment:

```bash
python exercise_b_transformer.py --batches 1 8 32 --dtypes fp32 fp16 --profile
```

Controlled batched-matmul experiment:

```bash
aip benchmark gpu --compare-cpu --batches 1 8 32 --precisions fp32 fp16
```

**Expected output:** `week1/results/transformer.csv` and `week1/results/aip_gpu.csv`

**Acceptance criteria:**

- Keep model/shape constant while comparing precision.
- Keep precision/shape constant while comparing batch.
- Report median latency, tokens/s or TFLOPS, and memory.
- Record unsupported dtype and OOM as results rather than silently removing them.

## 3. Build GPU Workload Profiler CLI

**Goal:** Create `aip benchmark gpu` that reports device, precision, batch, memory use, and runtime.

Install from repository root:

```bash
python -m pip install -e ./week1
aip benchmark gpu --help
```

Quick Mac run:

```bash
aip benchmark gpu \
  --device auto \
  --compare-cpu \
  --sizes 128 512 1024 \
  --batches 1 8 32 \
  --precisions fp32 fp16
```

The CLI uses CUDA when available, otherwise MPS, otherwise CPU. On CPU, memory is estimated from input/output tensors; CUDA reports peak PyTorch allocation; MPS reports an allocator snapshot. The CSV includes `memory_measurement` so these different definitions are not accidentally treated as identical.

## 4. Document GPU performance findings

**Goal:** Explain why utilization and runtime change across experiments.

**Implementation:** `GPU_PERFORMANCE_FINDINGS.md`

After running the experiments:

1. Copy key measurements from the CSV files into the report.
2. Describe whether latency grows linearly with batch.
3. Explain whether throughput improves with batch.
4. Compare FP32 with FP16/BF16 memory and performance.
5. Separate compute-bound, memory-bound, launch-bound, and transfer-bound explanations.
6. State limitations of Mac MPS versus NVIDIA CUDA/HBM measurements.
