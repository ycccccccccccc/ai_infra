# Week 1 Hands-on Labs

這組實驗預設先在 Apple Silicon Mac 上使用 PyTorch MPS，之後同一份程式也可以放到 Kaggle 或 Google Colab 的 NVIDIA GPU 上執行。

## 檔案

- `exercise_a_hardware.md`：A100、H100、H200 規格整理與分析題。
- `exercise_cpu_mps_matmul.py`：CPU 與 GPU matrix multiplication benchmark。
- `exercise_b_transformer.py`：小型 Transformer 的 batch、sequence length、dtype benchmark。
- `exercise_c_kernels.py`：matmul、element-wise、embedding lookup 比較。
- `exercise_d_vllm.md`：使用免費 NVIDIA GPU 觀察 LLM prefill 與 decode。
- `ASSIGNMENTS.md`：四個 implementation tasks、指令與完成條件。
- `GPU_PERFORMANCE_FINDINGS.md`：根據實驗結果整理的短報告。
- `pyproject.toml`、`src/aip/`：`aip benchmark gpu` CLI package。
- `tests/`：CLI 的單元測試。
- `results/`：各程式產生的 CSV/JSON 結果。三個 Python script 會根據自身檔案位置寫入這裡，不受執行命令時所在資料夾影響。

## 1. 建立環境

在這個資料夾執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

確認 MPS：

```bash
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

## 2. 建議執行順序

```bash
python exercise_cpu_mps_matmul.py
python exercise_b_transformer.py
python exercise_c_kernels.py
```

安裝 week1 CLI（從 `week1` 資料夾執行）：

```bash
python -m pip install -e .
aip benchmark gpu --help
aip benchmark gpu --compare-cpu --sizes 128 512 1024 --batches 1 8 32
```

CLI 會報告 device、precision、batch、matrix size、runtime、throughput 與 memory，並寫入 `week1/results/aip_gpu.csv`。CLI 會根據已安裝 package 的 week1 位置解析此路徑，不受目前工作目錄影響。

程式預設將結果寫入 `week1/results/`。如果最大設定執行太久，可以先縮小實驗：

```bash
python exercise_b_transformer.py --batches 1 8 --sequences 128 512 --iterations 5
python exercise_c_kernels.py --matmul-size 2048 --element-count 8388608 --lookup-count 32768
```

## 3. 搬到 Kaggle / Colab

上傳三個 `.py` 檔，在有 NVIDIA GPU 的 notebook 中執行：

```bash
!nvidia-smi
!python exercise_cpu_mps_matmul.py --devices cpu cuda
!python exercise_b_transformer.py --device cuda --dtypes fp32 fp16
!python exercise_c_kernels.py --devices cpu cuda --dtype fp16
```

每次報告都應記錄：

- 裝置名稱
- PyTorch 與 CUDA/macOS 版本
- dtype
- warm-up 與 iterations
- latency 中位數
- throughput
- 記憶體用量
- OOM 或不支援的設定

## 4. 公平比較原則

1. GPU 計時前後必須同步。
2. 排除第一次編譯與初始化，因此先 warm-up。
3. 比較時一次只改一個變因。
4. 小 workload 增加 inner loops，避免計時誤差。
5. Mac MPS 是 unified memory；NVIDIA CUDA 通常是 system RAM 加獨立 VRAM/HBM，不能把兩者視為完全相同的硬體。
6. PyTorch Profiler 本身有 overhead；benchmark 計時與 profiler 分開執行。
