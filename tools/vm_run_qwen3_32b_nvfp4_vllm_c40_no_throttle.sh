#!/usr/bin/env bash
set -euo pipefail

# Re-run Qwen3-32B NVFP4 (compressed-tensors) with requested concurrency=40
# and NO external in-flight throttling.
#
# Output:
# - Prints UNIT=... and OUTDIR=...
# - Result JSON: <OUTDIR>/result.json
# - Run log:     <OUTDIR>/run.log

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_path="${MODEL_PATH:-/mnt/data/models/RedHatAI_Qwen3-32B-NVFP4}"
in_tokens="${IN_TOKENS:-10000}"
out_tokens="${OUT_TOKENS:-800}"
gpu_mem_util="${GPU_MEM_UTIL:-0.90}"
cpu_offload_gb="${CPU_OFFLOAD_GB:-0}"
concurrency="${CONCURRENCY:-40}"

# Label used to name OUTDIR on the VM
label="${LABEL:-qwen3_32b_nvfp4_vllm_10k_0p8k_c40_no_throttle}"

# NVFP4 is pre-quantized checkpoint; bench uses quant=none.
quant="${QUANT:-none}"

# 0 = do not add '--enforce_eager' to bench_model_quant.py
# (This is unrelated to max_inflight throttling; override if you need parity.)
enforce_eager="${ENFORCE_EAGER:-0}"

# Key knob requested by the task:
# BENCH_MAX_INFLIGHT=0 means 'unlimited' (no external throttling).
export BENCH_MAX_INFLIGHT=0

# Optional quality-of-life knobs
export BENCH_FORCE_KILL="${BENCH_FORCE_KILL:-0}"
export BENCH_RUNTIME_MAX_SECS="${BENCH_RUNTIME_MAX_SECS:-infinity}"

exec "${root_dir}/vm_run_fp8_bench.sh" \
  "${model_path}" \
  "${in_tokens}" \
  "${out_tokens}" \
  "${label}" \
  "${quant}" \
  "${gpu_mem_util}" \
  "${cpu_offload_gb}" \
  "${concurrency}" \
  "${enforce_eager}"
