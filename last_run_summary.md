# 上一次成功运行结果摘要（FP8 推理）

- 运行时间：2025-12-17
- 运行目录：`/mnt/data/work/bench_runs/20251217_131136`
- 输入/输出长度：10000 tokens → 800 tokens

## 环境信息

- 模型名称：Qwen2.5-14B-Instruct（本地路径：`/mnt/data/models/Qwen2.5-14B-Instruct`）
- GPU 型号：RTX Pro 6000 BSE（nvidia-smi 显示：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q），96GB 显存
- 推理引擎名称及版本：vLLM 0.12.0（quantization=fp8）

## 指标汇总

> 说明：本次为单请求基准（1 query）。`Queries Per Second` 以 `QPS = 1 / E2E` 计算。

| 指标 | 数值 | 说明 |
|---|---:|---|
| Queries Per Second | 0.0535 | 由 E2E=18.7046s 反算（单请求） |
| Prompt 阶段 TPS（token/s） | 15908.807 | 由脚本统计（prefill） |
| Decode 阶段 TPS（token/s） | 44.257 | 由脚本统计（decode） |
| First Token Latency (ms) | 628.583 | TTFT |
| End‑to‑end Latency (ms) | 18704.625 | E2E |
| Time Per Output Token（ms） | 22.595 | TPOT |
| GPU 利用率（%） | 18.23 | 来自 `gpm_mig.csv` 的 SM Activity 均值（max=85） |
| Tensor Core 利用率（%） | 6.68 | 来自 `gpm_mig.csv` 的 MMA Activity 均值（max=39） |
| 显存占用（GB） | 96.22 | 来自 `compute_apps.csv` 进程峰值（VLLM::EngineCore），约等于 89.61 GiB |
| KV Cache 占用（GB） | 77.14 | 来自 vLLM 日志“Available KV cache memory: 71.84 GiB”（容量/可用预算） |
| 显存带宽利用率（%） | N/A | `gpm_mig.csv` 的 DRAM Activity 大多为 “-”，有效样本仅 2 条且为 0（不具代表性） |

## 原始产物（便于追溯）

- `bench.json`：TTFT/E2E/Prompt TPS/Decode TPS/TPOT 等
- `bench.log`：包含 KV cache 相关日志行
- `compute_apps.csv`：按进程采样的显存占用
- `gpm_mig.csv`：MIG 级别 GPM 采样（SM/MMA/DRAM Activity）
