# 测试记录：Qwen 3 32B（10K 输入 → 0.8K 输出）

## 测试设定

- 模型：Qwen/Qwen3-32B-FP8（FP8 checkpoint）
- 负载：输入 10000 tokens；输出 800 tokens
- 精度：FP8、FP4
- 并发：10 / 40 / 200
- GPU：RTX Pro 6000 BSE（96GB 显存）
- 推理引擎：vLLM 0.12.0

## 环境信息（每次测试前记录一次）

- 测试时间（时区）：2025-12-18 01:19:29 CST
- VM / 节点：
- GPU Driver：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, 98304 MiB, 580.105.08
- MIG：Enabled/Disabled；Profile：
- 关键参数：tensor_parallel_size=；gpu_memory_utilization=；其它：

## 结果汇总（填写）

### FP8

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.242 | 80347.309 | 19.444 | 133.260 | 41277.883 | 51.431 | HF: Qwen/Qwen3-32B-FP8；checkpoint含float8；enforce_eager=1(禁用torch.compile/cudagraph) |
| 40 | 0.859 | 45856.968 | 17.275 | 241.641 | 46550.419 | 57.886 | HF: Qwen/Qwen3-32B-FP8；checkpoint含float8；enforce_eager=1(禁用torch.compile/cudagraph) |
| 200 | 1.927 | 8522.003 | 15.415 | 20130.718 | 72115.927 | 64.982 | HF: Qwen/Qwen3-32B-FP8；checkpoint含float8；enforce_eager=1(禁用torch.compile/cudagraph) |

### FP4

| 并发 | QPS | Prompt TPS (token/s) | Decode TPS (token/s) | TTFT (ms) | E2E (ms) | TPOT (ms) | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.336 | 103552.020 | 26.987 | 101.209 | 29745.514 | 37.055 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；max_inflight=24 |
| 40 | 0.587 | 74638.102 | 23.362 | 141.805 | 34566.358 | 43.031 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；max_inflight=24 |
| 200 | 0.624 | 73498.586 | 22.233 | 138.771 | 36217.538 | 45.098 | HF: RedHatAI/Qwen3-32B-NVFP4；vLLM compressed-tensors (NVFP4)；gpu_mem_util=0.90；max_model_len=10928；max_inflight=24 |

## 分析与结论（填写）

- 结论摘要：
- FP8 vs FP4：
- 并发扩展性（10→40→200）：
- 是否出现 OOM/排队/抖动：

## 运行产物（可选）

- 日志路径：
- JSON 输出：
- 复现实验命令：
