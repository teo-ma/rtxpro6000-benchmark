# 测试记录：Qwen 3 14B（10K 输入 → 0.8K 输出）

## 测试设定

- 模型：FP8=Qwen/Qwen3-14B-FP8；FP4(NVFP4)=RedHatAI/Qwen3-14B-NVFP4
- 负载：输入 10000 tokens；输出 800 tokens
- 精度：FP8、FP4（NVFP4）
- 并发：10 / 40 / 200
- GPU：RTX Pro 6000 BSE（96GB 显存）
- 推理引擎：vLLM 0.12.0

## 环境信息（每次测试前记录一次）

- 测试时间（时区）：FP8=2025-12-18 10:39:40 CST；FP4(NVFP4)=2025-12-18 11:28:39 CST
- VM / 节点：
- GPU Driver：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, 98304 MiB, 580.105.08
- MIG：Enabled/Disabled；Profile：
- 关键参数：tensor_parallel_size=1；gpu_memory_utilization=0.90（vLLM）；其它：

## 结果汇总（填写）

### FP8

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 实际 Input tokens | 实际 Output tokens | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.054 | 197352.079 | 43.501 | 50.671 | 18417.992 | 22.988 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时，严格输出=800）；Prefix cache hit rate：GPU=49.92%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-14B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8005`；strict：`python3 /mnt/data/work/tools/vllm_strict_c1_check.py --model /mnt/data/models/Qwen_Qwen3-14B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-14B-FP8 --api-base http://127.0.0.1:8005 --metrics-url http://127.0.0.1:8005/metrics --server-log <outdir>/vllm_server.log --in-tokens 10000 --out-tokens 800 --requests 2 --stream-metrics --out-json <outdir>/strict_result.json` |
| 10 | 0.486 | 194913.743 | 38.979 | 51.305 | 20549.589 | 25.655 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；30 req / c=10；严格输出=800）；Prefix cache hit rate：GPU=96.62%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-14B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8006`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-14B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-14B-FP8 --api-base http://127.0.0.1:8006 --metrics-url http://127.0.0.1:8006/metrics --in-tokens 10000 --out-tokens 800 --concurrency 10 --warmup 1 --total-requests 30 --out-json <outdir>/result_strict_c10.json` |
| 40 | 1.625 | 107634.070 | 32.602 | 92.907 | 24600.357 | 30.673 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；120 req / c=40；严格输出=800；未做外部 request-rate 限流）；Prefix cache hit rate：GPU=99.01%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-14B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8007`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-14B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-14B-FP8 --api-base http://127.0.0.1:8007 --metrics-url http://127.0.0.1:8007/metrics --in-tokens 10000 --out-tokens 800 --concurrency 40 --warmup 1 --total-requests 120 --out-json <outdir>/result_strict_c40.json` |




### FP4

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.637 | 148747.311 | 51.164 | 68.637 | 15704.483 | 19.545 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |
| 40 | 1.938 | 68569.446 | 39.050 | 147.213 | 20633.955 | 25.608 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |
| 200 | 4.337 | 13741.922 | 34.719 | 9146.091 | 32251.407 | 28.882 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |


