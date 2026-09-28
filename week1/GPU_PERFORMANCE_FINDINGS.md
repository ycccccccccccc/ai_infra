# GPU Performance Findings

## Scope

The local experiment compares the M5 CPU with the Apple GPU through PyTorch MPS. It is a compute-only benchmark: matrices are created on the target device before timing, so device-transfer time is excluded. These measurements demonstrate workload behavior but do not measure CUDA, NVIDIA Tensor Cores, HBM, or NVLink.

## Matrix-size findings

| Matrix size | CPU time | MPS time | MPS speedup |
|---:|---:|---:|---:|
| 128 | 0.005 ms | 0.327 ms | 0.02× |
| 256 | 0.026 ms | 0.569 ms | 0.05× |
| 512 | 0.185 ms | 0.848 ms | 0.22× |
| 1024 | 1.117 ms | 1.582 ms | 0.71× |
| 2048 | 8.128 ms | 4.895 ms | 1.66× |
| 4096 | 67.864 ms | 39.187 ms | 1.73× |

The observed crossover point is between matrix sizes 1024 and 2048. Below that point, CPU execution is faster. Above it, the GPU provides better throughput.

Small matrix multiplication does too little work to amortize framework dispatch, Metal command submission, kernel launch, and synchronization. CPU libraries can also keep small operands in cache and use optimized vector instructions. As matrix size grows, the amount of work grows as O(N³), while the matrix storage grows as O(N²). Arithmetic intensity therefore increases and the GPU has enough independent work to use more parallel execution resources.

At sizes 2048 and 4096, the MPS result is approximately 3.51 TFLOPS, while the CPU result is approximately 2.0–2.1 TFLOPS. The similar achieved throughput across the two largest sizes suggests that both implementations are approaching a steady operating region for this shape, dtype, framework version, and system. These are achieved benchmark values, not hardware peak specifications.

The extremely short CPU measurements for sizes 128 and 256 are sensitive to timer resolution, cache state, thread-pool behavior, and background activity. They should be measured with many inner iterations and interpreted as latency observations rather than peak-FLOPS measurements.

## Precision findings

FP16 stores each element in half the bytes of FP32, so the input and output tensor footprint should be approximately halved for identical shapes. Lower precision can also increase compute throughput when the backend has an optimized low-precision matrix kernel.

However, lower precision does not guarantee lower end-to-end latency. Conversion overhead, unsupported operators, small workloads, non-optimal matrix alignment, and memory or launch bottlenecks can hide the theoretical benefit. On NVIDIA hardware, Tensor Core availability and supported dtype are especially important. On Apple MPS, the implementation follows Metal/MPS kernels rather than CUDA Tensor Core behavior.

Measured precision results are intentionally left to the generated CSV instead of being invented here. After running the experiment, record:

| Device | Size/batch | FP32 runtime | FP16/BF16 runtime | FP32 memory | FP16/BF16 memory |
|---|---|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Batch-size findings

Increasing batch size gives the accelerator more independent matrix operations or tokens to process. Throughput often improves because fixed launch costs are amortized and more execution units can remain busy. Per-batch latency nevertheless rises because the batch contains more work. A useful result is therefore not merely “GPU utilization increased,” but whether throughput rose faster than latency and whether the latency remains acceptable.

Memory use should increase approximately with batch for batched input, output, and activation tensors. Transformer attention can grow more aggressively with sequence length, and inference KV cache grows with active sequences and processed tokens. Eventually a larger batch can stop helping because compute, memory bandwidth, capacity, or scheduling is saturated.

After running `exercise_b_transformer.py`, add representative measurements:

| Precision | Batch | Sequence | Latency | Tokens/s | Memory |
|---|---:|---:|---:|---:|---:|
| TBD | 1 | TBD | TBD | TBD | TBD |
| TBD | 8 | TBD | TBD | TBD | TBD |
| TBD | 32 | TBD | TBD | TBD | TBD |

## Why utilization and runtime can move differently

Utilization indicates that the GPU was active during a sampling window; it does not show how efficiently the useful model work was completed. Runtime and utilization may diverge for several reasons:

- A larger batch can increase utilization and throughput while also increasing individual request latency.
- A memory-bound kernel can show high activity while compute units wait for data.
- Many small kernels can create high dispatch activity but poor useful throughput.
- Synchronization, CPU preprocessing, data transfer, and queueing add runtime outside the main GPU kernel.
- Allocated memory can remain high because of caching even when little computation is running.
- Thermal or power limits can reduce clock speed during a long experiment.

Consequently, the report should pair utilization with workload throughput, latency percentiles, memory, achieved FLOPS/bandwidth, and profiler evidence.

## Kernel classification

- Large square matrix multiplication is usually compute-bound because each loaded value participates in many multiply-add operations.
- Eager element-wise addition and activation are usually memory-bound because they perform little computation per byte read or written.
- Random embedding lookup is usually memory-bound and sensitive to cache locality because it gathers rows from non-contiguous locations.
- Small versions of any of these workloads may be launch-bound rather than compute- or memory-bound.

## Limitations and next step

The Mac uses unified memory shared by CPU and GPU, unlike a typical NVIDIA server with system RAM, PCIe, and dedicated VRAM/HBM. MPS profiler data also should not be interpreted as CUDA kernel metrics. The next validation step is to run the same CLI on a Kaggle or Colab T4/L4, record `nvidia-smi`, and compare CUDA FP32/FP16 results without assuming the two platforms have identical memory behavior.

