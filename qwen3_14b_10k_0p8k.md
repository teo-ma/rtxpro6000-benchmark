# 测试记录：Qwen 3 14B（10K 输入 → 0.8K 输出）

## 测试设定

- 模型：FP8=Qwen/Qwen3-14B-FP8；FP4(NVFP4)=RedHatAI/Qwen3-14B-NVFP4
- 负载：输入 10000 tokens；输出 800 tokens
- 精度：FP8、FP4（NVFP4）
- 并发：10 / 40 / 200
- GPU：RTX Pro 6000 BSE（96GB 显存）
- 推理引擎：FP8=sglang 0.5.6.post2；FP4(NVFP4)=vLLM 0.12.0

## 环境信息（每次测试前记录一次）

- 测试时间（时区）：FP8=2025-12-18 10:39:40 CST；FP4(NVFP4)=2025-12-18 11:28:39 CST
- VM / 节点：
- GPU Driver：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, 98304 MiB, 580.105.08
- MIG：Enabled/Disabled；Profile：
- 关键参数：tensor_parallel_size=；gpu_memory_utilization=0.90（vLLM）；其它：

## 结果汇总（填写）

### FP8

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.532 | 2005.716 | 193.291 | 1624.730 | 11627.491 | 29.143 | sglang 0.5.6.post2; attention_backend=triton; sampling_backend=pytorch; fp8_gemm_backend=cutlass; SGLANG_ENABLE_JIT_DEEPGEMM=0 |
| 40 | 1.201 | 5321.382 | 464.224 | 5709.358 | 23091.384 | 85.780 | sglang 0.5.6.post2; attention_backend=triton; sampling_backend=pytorch; fp8_gemm_backend=cutlass; SGLANG_ENABLE_JIT_DEEPGEMM=0 |
| 200 | 1.634 | 7486.921 | 700.456 | 34816.736 | 81231.592 | 118.182 | sglang 0.5.6.post2; attention_backend=triton; sampling_backend=pytorch; fp8_gemm_backend=cutlass; SGLANG_ENABLE_JIT_DEEPGEMM=0 |

### FP4

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.637 | 148747.311 | 51.164 | 68.637 | 15704.483 | 19.545 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |
| 40 | 1.938 | 68569.446 | 39.050 | 147.213 | 20633.955 | 25.608 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |
| 200 | 4.337 | 13741.922 | 34.719 | 9146.091 | 32251.407 | 28.882 | vLLM 0.12.0 + NVFP4 (RedHatAI/Qwen3-14B-NVFP4) |

## 分析与结论（填写）

- 结论摘要：
- FP8 vs FP4：
- 并发扩展性（10→40→200）：
- 是否出现 OOM/排队/抖动：

## 运行产物（可选）

- 日志路径：
- JSON 输出：
- 复现实验命令：
