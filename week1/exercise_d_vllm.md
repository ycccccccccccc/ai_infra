# Exercise D：LLM Prefill 與 Decode

這一題建議在 Google Colab 或 Kaggle 的 NVIDIA T4（或更新 GPU）上執行。Mac 上可以使用 vLLM-Metal 延伸練習，但其 backend 與 CUDA 不同，不宜直接混合比較。

## 1. 確認 GPU

在 notebook cell 執行：

```bash
!nvidia-smi
```

若拿到 P100，新版 vLLM 可能不支援；Exercise B/C 仍可執行，但這一題建議改用 T4、L4 或更新 GPU。

## 2. 安裝並啟動 server

```bash
!pip install -q vllm
```

在 terminal 或背景 process 啟動：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --enable-chunked-prefill
```

如果 OOM，先把 `--max-model-len` 降到 4096，再把 `--gpu-memory-utilization` 調到 0.85。

## 3. Baseline

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 100 \
  --request-rate inf \
  --max-concurrency 8 \
  --ignore-eos \
  --num-warmups 5 \
  --save-result \
  --save-detailed \
  --result-dir week1/results/vllm
```

上面的相對路徑假設命令從 repository 根目錄執行；結果會集中在 `week1/results/vllm/`。

記錄 mean/median/p99 TTFT、ITL 或 TPOT、total token throughput。

## 4. One-factor-at-a-time 實驗矩陣

每組只改一個變因：

| 實驗 | 改變 | 固定 |
|---|---|---|
| Input length | 128、512、2048、4096 | output=128、concurrency=8 |
| Output length | 16、128、512、1024 | input=512、concurrency=8 |
| Concurrency | 1、8、32、128 | input=512、output=128 |
| Batch policy | max batched tokens=512、4096、8192；max seqs=8、32、128 | input/output workload 相同 |

`max-num-batched-tokens` 和 `max-num-seqs` 是 server 參數，因此每次變更後要重啟 server。

## 5. HBM 與 KV cache

另一個 terminal 持續觀察：

```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory \
  --format=csv -l 1
```

查詢 vLLM metrics：

```bash
curl -s http://localhost:8000/metrics | \
  grep -E 'kv_cache_usage|num_requests_(running|waiting)|num_preemptions'
```

vLLM 通常啟動時就預留大量 GPU memory，所以 `nvidia-smi memory.used` 可能變化不大。應同時觀察 KV cache usage、waiting requests 與 preemption。

## 6. 預期結論

- Input length 增加主要提高 prefill 計算量，因此 TTFT 通常明顯增加；初始 KV cache 也會變大。
- Output length 增加主要提高 decode 時間與 KV cache 壓力，總延遲增加，但 TTFT 不一定大幅改變。
- Concurrency 增加通常先提升 aggregate tokens/s，之後 queueing、TTFT、ITL 與 KV cache usage 開始上升。
- 較大的 batch budget 通常有利 throughput，但可能惡化個別 request 的延遲。
- Chunked prefill 能限制長 prompt 一次占用的 scheduler budget，常能改善 decode request 被長 prefill 阻塞的情況。

## 7. 結果表

| input | output | concurrency | max batched tokens | max seqs | median TTFT | p99 TTFT | median ITL/TPOT | tokens/s | peak HBM | peak KV cache | preemptions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

重複 benchmark 時要重啟 server 或更換 random seed，避免 prefix cache 讓後續實驗得到不公平的加速。
