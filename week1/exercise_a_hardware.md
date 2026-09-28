# Exercise A：GPU 硬體規格表

規格快照日期：2026-09-12。以下以 SXM 版本比較；PCIe 版本可能具有不同頻寬、功耗與 interconnect。

| GPU | Architecture / SM | Tensor Core / supported dtype | HBM | Peak bandwidth | Peak Tensor compute | Interconnect |
|---|---|---|---:|---:|---:|---|
| NVIDIA A100 80GB SXM | Ampere，108 SM | 第 3 代；TF32、BF16、FP16、FP64、INT8、INT4 | 80GB HBM2e | 2.039 TB/s | BF16/FP16：312 dense／624 sparse TFLOPS；TF32：156 dense／312 sparse TFLOPS | NVLink 600 GB/s；PCIe Gen4 64 GB/s |
| NVIDIA H100 SXM | Hopper，132 SM | 第 4 代；FP8、BF16、FP16、TF32、FP64、INT8；Transformer Engine | 80GB HBM3 | 3.35 TB/s | FP8：1,979 dense／3,958 sparse TFLOPS；BF16/FP16：約 989.5 dense／1,979 sparse TFLOPS | NVLink 900 GB/s；PCIe Gen5 128 GB/s |
| NVIDIA H200 SXM | Hopper，132 SM | 第 4 代；dtype 與 H100 大致相同 | 141GB HBM3e | 4.8 TB/s | FP8：1,979 dense／3,958 sparse TFLOPS；BF16/FP16：約 989.5 dense／1,979 sparse TFLOPS | NVLink 900 GB/s；PCIe Gen5 128 GB/s |

## 結論

### 哪一張較適合大模型訓練？

三者之中優先選 H200。它的張量運算峰值與 H100 接近，但 141GB HBM 能容納較大的模型、activation、gradient 與 optimizer state，4.8 TB/s bandwidth 也能降低部分 memory bottleneck。若模型已能輕鬆放入 80GB，H100 與 H200 的實際差距仍須 benchmark 才能確定。

### 哪一張較適合 memory-bound inference？

H200。它的 HBM capacity 與 bandwidth 都最高，適合權重讀取與 KV cache 壓力較大的 inference，也能容納更多 concurrent sequences。

### 規格本身仍不足以確定什麼？

- 實際 kernel 能達到多少理論 FLOPS 或 bandwidth。
- 模型是否真正走 Tensor Core、FP8 或 structured sparsity 路徑。
- batch、sequence length、matrix shape、layout 與 cache locality 的影響。
- TTFT、ITL、tokens/s 與 tail latency。
- 多卡 NVSwitch 拓撲及跨節點網路的瓶頸。
- 軟體版本、compiler、kernel fusion、quantization 與 inference engine 的影響。
- 功耗、降頻、成本、可用性與每瓦／每美元效能。

## 重要註記

官方表格常把啟用 structured sparsity 後的數字列為 peak throughput。一般 dense 模型不能直接使用兩倍的 sparse 數字。

## 官方來源

- A100：https://www.nvidia.com/en-us/data-center/a100/
- A100 architecture：https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- H100：https://www.nvidia.com/en-us/data-center/h100/
- H200：https://www.nvidia.com/en-us/data-center/h200/

## 作業延伸

選擇你實際可取得的 Kaggle／Colab GPU，執行 `nvidia-smi`，把它加入上表。確認以下資訊：

1. GPU 型號與 form factor。
2. 是否有 Tensor Core。
3. 是否支援 FP16、BF16 或 FP8。
4. 顯存類型與容量。
5. 該 notebook 是否真的提供多 GPU interconnect。

