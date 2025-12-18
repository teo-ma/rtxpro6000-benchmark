#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <model_path_or_hf_id> <in_tokens> <out_tokens> <label> [concurrencies] [mem_fraction_static] [context_length] [port]" >&2
  exit 2
fi

model_ref="$1"
in_tokens="$2"
out_tokens="$3"
label="$4"

# Prefer stable per-model folder naming for HF downloads.
safe_name="${model_ref//\//_}"

concurrencies="${5:-10,40,200}"
mem_fraction_static="${6:-0.90}"
context_length="${7:-$((in_tokens + out_tokens + 128))}"
port="${8:-30000}"

# Optional sglang-specific knobs (read once here so they can be safely injected
# into the systemd-run command string).
kv_cache_dtype="${BENCH_SGLANG_KV_CACHE_DTYPE:-auto}"
quant_param_path="${BENCH_SGLANG_QUANT_PARAM_PATH:-}"

base="/mnt/data/work/bench_suite_results"
ts="$(date +%Y%m%d_%H%M%S)"
outdir="$base/${ts}_${label}"
unit="bench_${ts}_${label}"

# Large models can take minutes to load/compile. Make transient unit timeouts
# explicit to avoid unexpected stops.
timeout_start_secs="${BENCH_TIMEOUT_START_SECS:-infinity}"
runtime_max_secs="${BENCH_RUNTIME_MAX_SECS:-infinity}"

mkdir -p "$outdir"

wait_secs="${BENCH_WAIT_SECS:-180}"
deadline=$(( $(date +%s) + wait_secs ))
while true; do
  if systemctl list-units --type=service --state=running 2>/dev/null | grep -qE '^bench_'; then
    if [[ $(date +%s) -ge $deadline ]]; then
      echo "ERROR: another bench unit is still running; set BENCH_FORCE_KILL=1 to override" >&2
      systemctl list-units --type=service --state=running | grep -E '^bench_' >&2 || true
      exit 3
    fi
    sleep 5
    continue
  fi

  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
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

if [[ "${BENCH_FORCE_KILL:-0}" == "1" ]]; then
  # Avoid pkill -f matching itself by using a bracket pattern.
  sudo pkill -f '[s]glang.launch_server' || true
  sudo pkill -f '[s]glang.bench_serving' || true
  sleep 2
fi

sudo systemd-run \
  --unit="$unit" \
  --collect \
  --property=WorkingDirectory=/mnt/data/work \
  --property=TimeoutStartSec="$timeout_start_secs" \
  --property=RuntimeMaxSec="$runtime_max_secs" \
  /bin/bash -lc \
  "set -euo pipefail; \
  model_ref='$model_ref'; \
  safe_name='$safe_name'; \
   export TOKENIZERS_PARALLELISM=false; \
   export PYTHONUNBUFFERED=1; \
   export HF_HUB_ENABLE_HF_TRANSFER=1; \
   export TMPDIR=/mnt/data/tmp; \
   export XDG_CACHE_HOME=/mnt/data/cache; \
   mkdir -p /mnt/data/tmp /mnt/data/cache; \
  export CUDA_HOME=/usr/local/cuda-13.0; \
  export CUDA_PATH=/usr/local/cuda-13.0; \
  export SGLANG_ENABLE_JIT_DEEPGEMM=0; \
   echo START: \$(date) | tee -a '$outdir/run.log'; \
   echo MODEL_REF: '$model_ref' | tee -a '$outdir/run.log'; \
   echo IN_TOKENS: '$in_tokens' OUT_TOKENS: '$out_tokens' | tee -a '$outdir/run.log'; \
   echo CONCURRENCIES: '$concurrencies' | tee -a '$outdir/run.log'; \
   echo CONTEXT_LENGTH: '$context_length' MEM_FRACTION_STATIC: '$mem_fraction_static' | tee -a '$outdir/run.log'; \
  echo KV_CACHE_DTYPE: '$kv_cache_dtype' QUANT_PARAM_PATH: '$quant_param_path' | tee -a '$outdir/run.log'; \
   nvidia-smi | tee -a '$outdir/run.log' || true; \

   # Resolve model path: if the arg looks like a local path and exists, use it;
   # otherwise download from HF into /mnt/data/models/<org>_<repo>.
   model_path=\"\$model_ref\"; \
   if [[ ! -d \"\$model_path\" ]]; then \
     candidate=\"/mnt/data/models/\$safe_name\"; \
     if [[ -f \"\$candidate/config.json\" ]]; then \
       model_path=\"\$candidate\"; \
     elif [[ \"\$model_ref\" == \"/mnt/data/models\" && -f /mnt/data/models/config.json ]]; then \
       # Backward-compat: only if the user explicitly passed /mnt/data/models.
       model_path=/mnt/data/models; \
     else \
       model_path=\"\$candidate\"; \
       echo DOWNLOADING: '$model_ref' '->' \"\$model_path\" | tee -a '$outdir/run.log'; \
       /mnt/data/venvs/qwen/bin/hf download '$model_ref' --local-dir \"\$model_path\" 2>&1 | tee -a '$outdir/run.log'; \
     fi; \
   fi; \
   echo MODEL_PATH: \"\$model_path\" | tee -a '$outdir/run.log'; \

   # Launch sglang server
   server_log='$outdir/server.log'; \
  quant_param_args=(); \
  if [[ -n '$quant_param_path' ]]; then quant_param_args+=(--quantization-param-path '$quant_param_path'); fi; \
   /mnt/data/venvs/qwen/bin/python -m sglang.launch_server \
      --model-path \"\$model_path\" \
     --host 0.0.0.0 \
     --port '$port' \
     --tensor-parallel-size 1 \
     --context-length '$context_length' \
     --mem-fraction-static '$mem_fraction_static' \
     --attention-backend triton \
     --sampling-backend pytorch \
     --fp8-gemm-backend cutlass \
     --quantization fp8 \
     --kv-cache-dtype '$kv_cache_dtype' \
    \${quant_param_args[@]} \
     --trust-remote-code \
     2>&1 | tee -a \"\$server_log\" & \
   server_pid=\$!; \
   echo SERVER_PID: \$server_pid | tee -a '$outdir/run.log'; \

   # Wait until /v1/models is ready
   deadline=\$(( \$(date +%s) + 600 )); \
   while true; do \
     if curl -fsS http://127.0.0.1:'$port'/v1/models >/dev/null 2>&1; then break; fi; \
     if [[ \$(date +%s) -ge \$deadline ]]; then \
       echo 'ERROR: server did not become ready in time' | tee -a '$outdir/run.log'; \
       tail -n 200 \"\$server_log\" | tee -a '$outdir/run.log' || true; \
       kill \$server_pid || true; \
       exit 5; \
     fi; \
     sleep 2; \
   done; \
   echo SERVER_READY: \$(date) | tee -a '$outdir/run.log'; \

   out_jsonl='$outdir/bench_sglang.jsonl'; \
  rm -f \"\$out_jsonl\"; \

   for c in \$(echo '$concurrencies' | tr ',' ' '); do \
     c=\$(echo \"\$c\" | tr -d ' '); \
     echo BENCH_CONCURRENCY: \"\$c\" | tee -a '$outdir/run.log'; \
     /mnt/data/venvs/qwen/bin/python -m sglang.bench_serving \
       --backend sglang \
       --host 127.0.0.1 \
       --port '$port' \
       --dataset-name random \
       --num-prompts \"\$c\" \
       --max-concurrency \"\$c\" \
       --random-input-len '$in_tokens' \
       --random-output-len '$out_tokens' \
       --warmup-requests 0 \
       --disable-tqdm \
       --output-file \"\$out_jsonl\" \
       --tag \"${label}_c\${c}\" \
       2>&1 | tee -a '$outdir/run.log'; \
   done; \

   echo STOPPING_SERVER: \$(date) | tee -a '$outdir/run.log'; \
   kill \$server_pid || true; \
   wait \$server_pid || true; \

   echo END: \$(date) | tee -a '$outdir/run.log'" >/dev/null

echo "UNIT=$unit"
echo "OUTDIR=$outdir"
echo "JSONL=$outdir/bench_sglang.jsonl"
