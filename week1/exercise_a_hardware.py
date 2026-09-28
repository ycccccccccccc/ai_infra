import gc
import time
import statistics
import torch

""" Testing MPS performance on MacOS(M5)
size= 128, device=cpu, time=    0.005 ms, performance=   774.36 GFLOPS
size= 128, device=mps, time=    0.327 ms, performance=    12.84 GFLOPS
GPU speedup: 0.02x

size= 256, device=cpu, time=    0.026 ms, performance=  1266.21 GFLOPS
size= 256, device=mps, time=    0.569 ms, performance=    58.95 GFLOPS
GPU speedup: 0.05x

size= 512, device=cpu, time=    0.185 ms, performance=  1447.58 GFLOPS
size= 512, device=mps, time=    0.848 ms, performance=   316.63 GFLOPS
GPU speedup: 0.22x

size=1024, device=cpu, time=    1.117 ms, performance=  1922.62 GFLOPS
size=1024, device=mps, time=    1.582 ms, performance=  1357.13 GFLOPS
GPU speedup: 0.71x

size=2048, device=cpu, time=    8.128 ms, performance=  2113.57 GFLOPS
size=2048, device=mps, time=    4.895 ms, performance=  3509.59 GFLOPS
GPU speedup: 1.66x

size=4096, device=cpu, time=   67.864 ms, performance=  2025.20 GFLOPS
size=4096, device=mps, time=   39.187 ms, performance=  3507.25 GFLOPS
GPU speedup: 1.73x
"""

SIZES = [128, 256, 512, 1024, 2048, 4096]
DTYPE = torch.float32


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark(size, device, repeats=10):
    a = torch.randn(size, size, device=device, dtype=DTYPE)
    b = torch.randn(size, size, device=device, dtype=DTYPE)

    # MPS 第一次執行可能包含編譯成本，因此先 warm up。
    for _ in range(5):
        _ = a @ b
    synchronize(device)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        c = a @ b
        synchronize(device)
        times.append(time.perf_counter() - start)

    median_sec = statistics.median(times)
    gflops = 2 * size**3 / median_sec / 1e9

    del a, b, c
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    return median_sec * 1000, gflops


devices = [torch.device("cpu")]

if torch.backends.mps.is_available():
    devices.append(torch.device("mps"))

results = {}

for size in SIZES:
    results[size] = {}

    for device in devices:
        latency_ms, gflops = benchmark(size, device)
        results[size][device.type] = latency_ms

        print(
            f"size={size:4d}, "
            f"device={device.type:3s}, "
            f"time={latency_ms:9.3f} ms, "
            f"performance={gflops:9.2f} GFLOPS"
        )

    if "mps" in results[size]:
        speedup = results[size]["cpu"] / results[size]["mps"]
        print(f"GPU speedup: {speedup:.2f}x\n")