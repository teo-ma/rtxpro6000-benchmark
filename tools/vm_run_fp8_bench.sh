#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <model_path> <in_tokens> <out_tokens> <label> [quant] [gpu_mem_util] [cpu_offload_gb] [concurrencies] [enforce_eager:0|1]" >&2
  echo "Env: BENCH_MAX_INFLIGHT=<int>  # Limit in-flight requests (0=unlimited). Useful when requested concurrency exceeds KV-cache capacity." >&2
  exit 2
fi

model_path="$1"
in_tokens="$2"
out_tokens="$3"
label="$4"

# Backward-compatible parsing:
# - arg5: quant (fp8|bitsandbytes|none) OR gpu_mem_util (float) OR omitted
# - next: gpu_mem_util (float) OR cpu_offload_gb OR concurrencies
# - next: cpu_offload_gb OR concurrencies
# - final: concurrencies
quant="fp8"
gpu_mem_util="0.90"
cpu_offload_gb="0"
concurrencies="10,40,200"
enforce_eager="0"

if [[ ${5:-} != "" ]]; then
  case "$5" in
    fp8|bitsandbytes|none)
      quant="$5"
      ;;
    *)
      gpu_mem_util="$5"
      ;;
  esac
fi

if [[ ${6:-} != "" ]]; then
  case "$6" in
    *","*)
      concurrencies="$6"
      ;;
    *)
      # If arg5 was a quant selector, then arg6 is gpu_mem_util;
      # otherwise keep backward-compat (arg6 = cpu_offload_gb).
      if [[ "$5" == "fp8" || "$5" == "bitsandbytes" || "$5" == "none" ]]; then
        gpu_mem_util="$6"
      else
        cpu_offload_gb="$6"
      fi
      ;;
  esac
fi

if [[ ${7:-} != "" ]]; then
  case "$7" in
    *","*)
      concurrencies="$7"
      ;;
    *)
      cpu_offload_gb="$7"
      ;;
  esac
fi

if [[ ${8:-} != "" ]]; then
  concurrencies="$8"
fi

if [[ ${9:-} != "" ]]; then
  enforce_eager="$9"
fi

base="/mnt/data/work/bench_suite_results"
ts="$(date +%Y%m%d_%H%M%S)"
outdir="$base/${ts}_${label}"
unit="bench_${ts}_${label}"

# Some distros/defaults apply a short RuntimeMaxSec to transient units.
# Use 'infinity' by default to disable the runtime limit. Override via env.
# Examples:
#   BENCH_RUNTIME_MAX_SECS=7200   # 2 hours
#   BENCH_RUNTIME_MAX_SECS=infinity
runtime_max_secs="${BENCH_RUNTIME_MAX_SECS:-infinity}"

mkdir -p "$outdir"

# Avoid accidentally killing another in-flight benchmark. By default we wait
# for GPU to become idle. Set BENCH_FORCE_KILL=1 to forcefully stop any
# existing vLLM/benchmark processes.
if [[ "${BENCH_FORCE_KILL:-0}" == "1" ]]; then
  sudo pkill -f 'VLLM::Engine[C]ore' || true
  sudo pkill -f '[b]ench_model_quant.py' || true
  sleep 2
fi

# Wait for VRAM to be released. This avoids flaky starts where a previous
# EngineCore process is still holding memory (or another benchmark is running).
wait_secs="${BENCH_WAIT_SECS:-180}"
deadline=$(( $(date +%s) + wait_secs ))
while true; do
  # If another bench systemd unit is running, wait.
  if systemctl list-units --type=service --state=running 2>/dev/null | grep -qE '^bench_'; then
    if [[ $(date +%s) -ge $deadline ]]; then
      echo "ERROR: another bench unit is still running; set BENCH_FORCE_KILL=1 to override" >&2
      systemctl list-units --type=service --state=running | grep -E '^bench_' >&2 || true
      exit 3
    fi
    sleep 5
    continue
  fi

  # If vLLM is still running or GPU is occupied by compute processes, wait.
  if pgrep -f 'VLLM::EngineCore' >/dev/null 2>&1 || \
     nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    if [[ $(date +%s) -ge $deadline ]]; then
      echo "ERROR: GPU is still busy; set BENCH_FORCE_KILL=1 to override" >&2
      nvidia-smi >&2 || true
      exit 4
    fi
    sleep 5
    continue
  fi

  break
done

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' | tr -d '\r' || echo 0)"
if [[ "${free_mib:-0}" -lt 4096 ]]; then
  echo "ERROR: GPU free memory is too low (${free_mib} MiB). Another process may still be running." >&2
  nvidia-smi >&2 || true
  exit 1
fi

sudo systemd-run \
  --unit="$unit" \
  --collect \
  --property=WorkingDirectory=/mnt/data/work \
  --property=TimeoutStartSec=infinity \
  --property=RuntimeMaxSec="$runtime_max_secs" \
  /bin/bash -lc \
  "set -euo pipefail; \
   export TOKENIZERS_PARALLELISM=false; \
   export PYTHONUNBUFFERED=1; \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
    export TMPDIR=/mnt/data/tmp; \
    export XDG_CACHE_HOME=/mnt/data/cache; \
    export TRITON_CACHE_DIR=/mnt/data/cache/triton; \
    export TORCHINDUCTOR_CACHE_DIR=/mnt/data/cache/torchinductor; \
    mkdir -p /mnt/data/tmp /mnt/data/cache /mnt/data/cache/triton /mnt/data/cache/torchinductor; \
   echo START: \$(date) | tee -a '$outdir/run.log'; \
   echo MODEL: '$model_path' | tee -a '$outdir/run.log'; \
  echo QUANT: '$quant' | tee -a '$outdir/run.log'; \
   echo GPU_MEM_UTIL: '$gpu_mem_util' | tee -a '$outdir/run.log'; \
   echo CPU_OFFLOAD_GB: '$cpu_offload_gb' | tee -a '$outdir/run.log'; \
   nvidia-smi | tee -a '$outdir/run.log' || true; \
   /mnt/data/venvs/qwen/bin/python /mnt/data/work/bench_model_quant.py \
     --model '$model_path' \
     --quant '$quant' \
     --gpu_mem_util '$gpu_mem_util' \
     --cpu_offload_gb '$cpu_offload_gb' \
     --in_tokens '$in_tokens' \
     --out_tokens '$out_tokens' \
     --concurrencies '$concurrencies' \
     --outdir '$outdir' \
     $([[ "${BENCH_MAX_INFLIGHT:-0}" != "0" ]] && echo --max_inflight "${BENCH_MAX_INFLIGHT}") \
     $([[ "$enforce_eager" == "1" ]] && echo --enforce_eager) \
     2>&1 | tee -a '$outdir/run.log'; \
   echo END: \$(date) | tee -a '$outdir/run.log'" >/dev/null


echo "UNIT=$unit"
echo "OUTDIR=$outdir"
