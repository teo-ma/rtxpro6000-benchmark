# 测试记录：Qwen 3 32B（10K 输入 → 0.8K 输出）

## 测试设定

- 模型：Qwen/Qwen3-32B-FP8（FP8 checkpoint）
- 负载：输入 10000 tokens；输出 800 tokens
- 精度：FP8、FP4
- 并发：1 / 10 / 40 / 200
- GPU：RTX Pro 6000 BSE（96GB 显存）
- 推理引擎：vLLM 0.12.0

## 环境信息（每次测试前记录一次）

- 测试时间（时区）：2025-12-18 01:19:29 CST
- VM / 节点：
- GPU Driver：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, 98304 MiB, 580.105.08
- MIG：Enabled/Disabled；Profile：
- 关键参数：tensor_parallel_size1；gpu_memory_utilization=；其它：

## 结果汇总（填写）

### FP8

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 实际 Input tokens | 实际 Output tokens | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.026 | 123192.312 | 21.213 | 81.174 | 37747.629 | 47.142 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时，严格输出=800）；Prefix cache hit rate：GPU=49.92%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-32B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8003`；strict：`python3 /mnt/data/work/tools/vllm_strict_c1_check.py --model /mnt/data/models/Qwen_Qwen3-32B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-32B-FP8 --api-base http://127.0.0.1:8003 --metrics-url http://127.0.0.1:8003/metrics --server-log <outdir>/vllm_server.log --in-tokens 10000 --out-tokens 800 --requests 2 --stream-metrics --out-json <outdir>/strict_result.json` |
| 10 | 0.246 | 97879.829 | 19.686 | 102.166 | 40690.183 | 50.799 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；30 req / c=10；严格输出=800）；Prefix cache hit rate：GPU=96.62%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-32B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8003`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-32B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-32B-FP8 --api-base http://127.0.0.1:8003 --metrics-url http://127.0.0.1:8003/metrics --in-tokens 10000 --out-tokens 800 --concurrency 10 --warmup 1 --total-requests 30 --out-json <outdir>/result_strict_c10.json` |
| 40 | 0.861 | 60793.479 | 17.263 | 164.491 | 46449.495 | 57.929 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；120 req / c=40；严格输出=800；未做外部 request-rate 限流）；Prefix cache hit rate：GPU=99.01%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-32B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8003`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-32B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-32B-FP8 --api-base http://127.0.0.1:8003 --metrics-url http://127.0.0.1:8003/metrics --in-tokens 10000 --out-tokens 800 --concurrency 40 --warmup 1 --total-requests 120 --out-json <outdir>/result_strict_c40.json` |




### FP4

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.336 | 103552.020 | 26.987 | 101.209 | 29745.514 | 37.055 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；|
| 40 | 0.587 | 74638.102 | 23.362 | 141.805 | 34566.358 | 43.031 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；max_inflight=24 |

说明：以上 FP4/NVFP4 的结果来自 vLLM bench（result.json），并且当时设置了 `max_inflight=24`（属于压测侧“在途请求数”外部限流；对并发=10 不生效，对并发=40 会生效），因此并发=40 这行不是“无外部限流”的 true concurrency 数据。

重测（并发=40，无外部限流）：

- 在 VM 上运行：`bash tools/vm_run_qwen3_32b_nvfp4_vllm_c40_no_throttle.sh`
- 该脚本会强制 `BENCH_MAX_INFLIGHT=0`（不注入 `--max_inflight`），并打印 `OUTDIR=...`
- 产物：`<OUTDIR>/result.json`（以及 `<OUTDIR>/run.log` 作为证据）
- 复核要点：新 `result.json` 的 `engine_args` 中不应出现 `max_inflight=24`（或应为 0 / None）

