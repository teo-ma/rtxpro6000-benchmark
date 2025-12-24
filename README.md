# Azure NC RTX Pro 6000 FP8 / NVFP4推理测试汇总

本文汇总了 **FP8** 以及 **FP4（NVFP4）** 推理测试结果，并给出测试方法、指标含义、以及在 Azure 上使用 RTX Pro 6000 GPU VM 的简要说明。

## 1) 测试概览

- 测试目标：在 RTX Pro 6000（96GB）上评估不同模型在 **长上下文 + 长输出**场景下的 FP8/NVFP4 推理吞吐与延迟。
- 负载设置：
  - Qwen 3 14B：10K 输入 → 0.8K 输出
  - Qwen 3 32B：10K 输入 → 0.8K 输出
  - Qwen 2.5 72B：20K 输入 → 1K 输出
- 并发：10 / 40 / 200
- GPU Driver：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, 98304 MiB, 580.105.08


## 2) 指标说明（读表指南）

- **QPS**：每秒完成的请求数（requests/s）。
- **Prompt TPS**：提示词（prefill）阶段的吞吐（tokens/s）。长上下文场景里它决定了 TTFT 的下限。
- **Decode TPS**：生成（decode）阶段的吞吐（tokens/s）。长输出场景里它决定生成速度。
- **TTFT**（ms）：Time To First Token，首 token 生成延迟。
- **E2E**（ms）：端到端延迟（从请求开始到完成）。
- **TPOT**（ms）：每输出 token 的平均耗时（越低越好）。

## 3) FP8 结果汇总

### 3.1 Qwen 3 14B（10K → 0.8K，FP8）

- 模型：Qwen/Qwen3-14B-FP8
- 引擎：vLLM 0.12.0

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 实际 Input tokens | 实际 Output tokens | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.054 | 197352.079 | 43.501 | 50.671 | 18417.992 | 22.988 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时，严格输出=800）；Prefix cache hit rate：GPU=49.92%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-14B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8005`；strict：`python3 /mnt/data/work/tools/vllm_strict_c1_check.py --model /mnt/data/models/Qwen_Qwen3-14B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-14B-FP8 --api-base http://127.0.0.1:8005 --metrics-url http://127.0.0.1:8005/metrics --server-log <outdir>/vllm_server.log --in-tokens 10000 --out-tokens 800 --requests 2 --stream-metrics --out-json <outdir>/strict_result.json` |
| 10 | 0.486 | 194913.743 | 38.979 | 51.305 | 20549.589 | 25.655 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；30 req / c=10；严格输出=800）；Prefix cache hit rate：GPU=96.62%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-14B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8006`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-14B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-14B-FP8 --api-base http://127.0.0.1:8006 --metrics-url http://127.0.0.1:8006/metrics --in-tokens 10000 --out-tokens 800 --concurrency 10 --warmup 1 --total-requests 30 --out-json <outdir>/result_strict_c10.json` |
| 40 | 1.625 | 107634.070 | 32.602 | 92.907 | 24600.357 | 30.673 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；120 req / c=40；严格输出=800；未做外部 request-rate 限流）；Prefix cache hit rate：GPU=99.01%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-14B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8007`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-14B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-14B-FP8 --api-base http://127.0.0.1:8007 --metrics-url http://127.0.0.1:8007/metrics --in-tokens 10000 --out-tokens 800 --concurrency 40 --warmup 1 --total-requests 120 --out-json <outdir>/result_strict_c40.json` |

### 3.2 Qwen 3 32B（10K → 0.8K，FP8）

- 模型：Qwen/Qwen3-32B-FP8（FP8 checkpoint）
- 引擎：vLLM 0.12.0

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 实际 Input tokens | 实际 Output tokens | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.026 | 123192.312 | 21.213 | 81.174 | 37747.629 | 47.142 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时，严格输出=800）；Prefix cache hit rate：GPU=49.92%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-32B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8003`；strict：`python3 /mnt/data/work/tools/vllm_strict_c1_check.py --model /mnt/data/models/Qwen_Qwen3-32B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-32B-FP8 --api-base http://127.0.0.1:8003 --metrics-url http://127.0.0.1:8003/metrics --server-log <outdir>/vllm_server.log --in-tokens 10000 --out-tokens 800 --requests 2 --stream-metrics --out-json <outdir>/strict_result.json` |
| 10 | 0.246 | 97879.829 | 19.686 | 102.166 | 40690.183 | 50.799 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；30 req / c=10；严格输出=800）；Prefix cache hit rate：GPU=96.62%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-32B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8003`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-32B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-32B-FP8 --api-base http://127.0.0.1:8003 --metrics-url http://127.0.0.1:8003/metrics --in-tokens 10000 --out-tokens 800 --concurrency 10 --warmup 1 --total-requests 30 --out-json <outdir>/result_strict_c10.json` |
| 40 | 0.861 | 60793.479 | 17.263 | 164.491 | 46449.495 | 57.929 | 10000 | 800 | vLLM 0.12.0（OpenAI `/v1/completions`，streaming 计时 + ThreadPoolExecutor 真并发；120 req / c=40；严格输出=800；未做外部 request-rate 限流）；Prefix cache hit rate：GPU=99.01%；server：`vllm serve /mnt/data/models/Qwen_Qwen3-32B-FP8 --enforce-eager --enable-prefix-caching --max-model-len 10928 --gpu-memory-utilization 0.90 --port 8003`；bench：`python3 /mnt/data/work/tools/vllm_strict_concurrency_bench.py --model /mnt/data/models/Qwen_Qwen3-32B-FP8 --tokenizer /mnt/data/models/Qwen_Qwen3-32B-FP8 --api-base http://127.0.0.1:8003 --metrics-url http://127.0.0.1:8003/metrics --in-tokens 10000 --out-tokens 800 --concurrency 40 --warmup 1 --total-requests 120 --out-json <outdir>/result_strict_c40.json` |

### 3.3 Qwen 2.5 72B（20K → 1K，FP8）

- 模型：Qwen2.5-72B-Instruct-fp8
- 引擎：sglang 0.5.6.post2

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.014 | 3.281 | 12.638 | 83.362 | 71850.196 | 79.038 | sglang fp8; kv-cache-dtype=fp8_e5m2 |

## 3B) FP4（NVFP4）结果补充

### 3B.1 Qwen 3 14B（10K → 0.8K，FP4 / NVFP4）

- 模型：Qwen3-14B-NVFP4
- 引擎：vLLM 0.12.0

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.637 | 148747.311 | 51.164 | 68.637 | 15704.483 | 19.545 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |
| 40 | 1.938 | 68569.446 | 39.050 | 147.213 | 20633.955 | 25.608 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |
| 200 | 4.337 | 13741.922 | 34.719 | 9146.091 | 32251.407 | 28.882 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |

### 3B.2 Qwen 3 32B（10K → 0.8K，FP4 / NVFP4）

- 模型：Qwen3-32B-NVFP4
- 引擎：vLLM 0.12.0

> 说明：下表当前数据来自 vLLM bench 的 `result.json`，当时设置了 `max_inflight=24`（压测侧外部“在途请求数”限流）。按需求要获得“并发=40 且不做外部限流”的结果，请在 VM 上运行 `tools/vm_run_qwen3_32b_nvfp4_vllm_c40_no_throttle.sh`（强制 `BENCH_MAX_INFLIGHT=0`），并用新 `result.json` 回填后再同步更新本表。

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.336 | 103552.020 | 26.987 | 101.209 | 29745.514 | 37.055 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；|
| 40 | 0.587 | 74638.102 | 23.362 | 141.805 | 34566.358 | 43.031 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；max_inflight=24 |

## 3C) NVFP（NVFP4）特性简介

- **更低比特的“浮点”量化**：NVFP4 是面向 NVIDIA GPU 的 4-bit 浮点量化方案，常用于把模型权重压缩到 4-bit，同时保留一定动态范围。
- **带来显存与带宽收益**：相较 FP16/BF16，4-bit 权重能显著降低显存占用与读取带宽压力，长上下文场景通常更受益于带宽与缓存效率。
- **精度/易用性取决于模型包**：我们使用的 `RedHatAI/Qwen3-*-NVFP4` 属于“已量化并打包”的 checkpoint，推理端通过 vLLM 的 `compressed-tensors` 路径直接加载，无需本地再做量化。
- **需要框架与 kernel 支持**：NVFP4 并非通用格式，通常要求推理框架（如 vLLM）与底层算子/内核已适配。
- **运行时参数仍很关键**：高并发 + 长上下文下容易触发 KV cache/调度瓶颈，可能需要像 `max_inflight` 这类限流参数来保持稳定（见 32B 的备注）。

## 4) 结果解读（面向客户的要点）

- **长上下文场景 TTFT 往往随并发显著上升**：10K/20K prompt 会带来较大的 prefill 计算与 KV cache 访问开销；当并发提高时，排队与批处理导致 TTFT 更明显拉长。
- **Decode 吞吐通常比 Prefill 更“敏感”**：长输出（0.8K/1K）下，decode TPS 更直接体现生成速度；大模型（72B）在长上下文时往往会出现更低的 decode TPS 与更高的 E2E。
- **FP8 vs FP4（NVFP4）的取舍**：FP4（NVFP4）通常能显著降低权重显存占用与读带宽压力，从而在部分场景带来更高吞吐/更低延迟；但精度与可用算子路径依赖具体模型包与框架实现，建议以目标任务的质量与稳定性为准做最终选择。
- **不同引擎/配置会影响吞吐与稳定性**：例如 vLLM 的 `enforce_eager=1`（禁用 torch.compile/cudagraph）通常更保守、更利于稳定复现，但可能牺牲部分性能上限；sglang 与 vLLM 在调度、融合算子、KV 管理路径上也不同。

## 5) Azure 上 RTX Pro 6000 GPU VM 简介（概述）

- Azure 提供 **GPU 加速虚拟机（GPU VM）**，适合运行推理/训练、可视化、以及需要高显存的工作负载。
- 本次测试使用的 Azure GPU VM 规格：**Standard_NC128ds_xl_RTXPRO6000BSE_v6**
  - 128 vCPU
  - 512GB 内存
  - 1 × NVIDIA RTX PRO 6000 BSE GPU（96GB 显存）
  - 官方规格文档：https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-rtxpro6000-bse-v6-series?tabs=sizebasicgp%2Csizeacceleratorsco
- 本次测试环境观测到的 GPU Driver：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, 98304 MiB, 580.105.08

> 备注：实际可用区域、配额、驱动版本与是否启用 MIG 等能力，可能随订阅/区域/镜像与配置不同而变化；本文中的 VM 规格与链接为本次测试所用型号的官方参考信息。

## 6) 数据来源（原始记录）

- Qwen 3 14B（FP8/FP4）：[qwen3_14b_10k_0p8k.md](qwen3_14b_10k_0p8k.md)
- Qwen 3 32B（FP8/FP4）：[qwen3_32b_10k_0p8k.md](qwen3_32b_10k_0p8k.md)
- Qwen 2.5 72B（FP8/FP4）：[qwen25_72b_20k_1k.md](qwen25_72b_20k_1k.md)
