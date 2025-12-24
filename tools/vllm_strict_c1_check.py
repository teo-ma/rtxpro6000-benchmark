#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class PrefixCacheHitRate:
    gpu_percent: Optional[float]
    cpu_percent: Optional[float]


def _extract_metric_float(metrics_text: str, metric_name: str) -> Optional[float]:
    # Prometheus exposition format: <name>{...} <value>
    # We'll take the last sample for that metric.
    last_val: Optional[float] = None
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not (line.startswith(metric_name) and (line == metric_name or line[len(metric_name)] in "{ \t")):
            continue
        # Strip labels if present
        try:
            value_str = line.split()[-1]
            last_val = float(value_str)
        except Exception:
            continue
    return last_val


def get_prefix_cache_hit_rate(metrics_url: str, timeout_s: float = 3.0) -> PrefixCacheHitRate:
    try:
        r = requests.get(metrics_url, timeout=timeout_s)
        r.raise_for_status()
        text = r.text
    except Exception:
        return PrefixCacheHitRate(gpu_percent=None, cpu_percent=None)

    # vLLM commonly exposes these metrics names (may vary by version/build).
    candidates = [
        ("vllm:gpu_prefix_cache_hit_rate", "vllm:cpu_prefix_cache_hit_rate"),
        ("vllm_gpu_prefix_cache_hit_rate", "vllm_cpu_prefix_cache_hit_rate"),
        ("vllm:prefix_cache_hit_rate_gpu", "vllm:prefix_cache_hit_rate_cpu"),
        ("vllm_prefix_cache_hit_rate_gpu", "vllm_prefix_cache_hit_rate_cpu"),
    ]

    for gpu_name, cpu_name in candidates:
        gpu = _extract_metric_float(text, gpu_name)
        cpu = _extract_metric_float(text, cpu_name)
        if gpu is not None or cpu is not None:
            # Many exporters expose 0~1, some expose percent already. Normalize to percent if <= 1.
            def norm(x: Optional[float]) -> Optional[float]:
                if x is None:
                    return None
                return x * 100.0 if 0.0 <= x <= 1.0 else x

            return PrefixCacheHitRate(gpu_percent=norm(gpu), cpu_percent=norm(cpu))

    # Newer vLLM versions may expose only counters, not a direct hit-rate gauge.
    # Compute hit% = hits_total / queries_total.
    counter_candidates = [
        ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
        ("vllm_prefix_cache_hits_total", "vllm_prefix_cache_queries_total"),
    ]
    for hits_name, queries_name in counter_candidates:
        hits = _extract_metric_float(text, hits_name)
        queries = _extract_metric_float(text, queries_name)
        if hits is None or queries is None:
            continue
        if queries <= 0:
            continue
        return PrefixCacheHitRate(gpu_percent=(hits / queries) * 100.0, cpu_percent=None)

    return PrefixCacheHitRate(gpu_percent=None, cpu_percent=None)


def parse_prefix_cache_hit_rate_from_log_text(log_text: str) -> PrefixCacheHitRate:
    # Example log line (seen in vLLM metrics logger):
    # "Prefix cache hit rate: GPU: 0.00%, CPU: 0.00%"
    matches = re.findall(r"Prefix cache hit rate:\s*GPU:\s*([0-9.]+)%\s*,\s*CPU:\s*([0-9.]+)%", log_text)
    if matches:
        gpu_s, cpu_s = matches[-1]
        try:
            return PrefixCacheHitRate(gpu_percent=float(gpu_s), cpu_percent=float(cpu_s))
        except Exception:
            return PrefixCacheHitRate(gpu_percent=None, cpu_percent=None)

    # Another common variant (observed in vLLM loggers):
    # "Prefix cache hit rate: 0.0%"
    matches2 = re.findall(r"Prefix cache hit rate:\s*([0-9.]+)%", log_text)
    if matches2:
        try:
            return PrefixCacheHitRate(gpu_percent=float(matches2[-1]), cpu_percent=None)
        except Exception:
            return PrefixCacheHitRate(gpu_percent=None, cpu_percent=None)

    return PrefixCacheHitRate(gpu_percent=None, cpu_percent=None)


def build_prompt_exact_tokens(tokenizer, target_tokens: int) -> str:
    # Build a prompt with *exactly* target_tokens tokens by:
    # 1) growing text until tokenized length >= target
    # 2) trimming token ids to target
    # 3) decoding back to text
    seed = (
        "请根据以下材料生成一段中性、客观的说明文字。\n"
        "材料：\n"
        "- 这是一段用于基准测试的长提示词。\n"
        "- 目标是精确控制 token 长度。\n\n"
        "正文：\n"
    )

    text = seed
    chunk = "这是一段用于填充上下文的内容。"

    # Grow fast
    while True:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            break
        # Exponential-ish growth without exploding memory
        text += (chunk + "\n") * 64

    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def call_vllm_completions(api_base: str, model: str, prompt: str, out_tokens: int, timeout_s: float) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/v1/completions"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_tokens,
        # Try to prevent early stop
        "min_tokens": out_tokens,
        "temperature": 0.0,
        "stream": False,
        # vLLM accepts extra parameters; keep both spellings to maximize compatibility
        "ignore_eos": True,
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


@dataclass(frozen=True)
class StreamingTiming:
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    qps: float
    prompt_tps: float
    decode_tps: float


def call_vllm_completions_stream_timing(
    api_base: str,
    model: str,
    prompt: str,
    out_tokens: int,
    timeout_s: float,
    tokenizer,
) -> tuple[str, StreamingTiming]:
    """Send a *streaming* /v1/completions request and compute TTFT/E2E/TPOT.

    We build these metrics from wall-clock timestamps:
    - TTFT: time from request start to first non-empty token chunk
    - E2E: time from request start to stream completion
    - TPOT: average time per output token (excluding the first token)

    Strict length is enforced by counting completion tokens with the provided tokenizer.
    """

    url = api_base.rstrip("/") + "/v1/completions"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_tokens,
        "min_tokens": out_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }

    t0 = time.time()
    t_first: Optional[float] = None
    completion_parts: list[str] = []
    last_full_text: str = ""

    with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except Exception:
                continue

            choices = evt.get("choices") or []
            if not choices:
                continue
            c0 = choices[0] or {}

            # Handle both streaming formats:
            # - /v1/completions: choices[0].text (often incremental, sometimes cumulative)
            # - /v1/chat/completions: choices[0].delta.content
            text_piece: Optional[str] = None
            delta = c0.get("delta") or {}
            if isinstance(delta, dict):
                if isinstance(delta.get("text"), str):
                    text_piece = delta.get("text")
                elif isinstance(delta.get("content"), str):
                    text_piece = delta.get("content")

            if text_piece is None and isinstance(c0.get("text"), str):
                text_piece = c0.get("text")

            if not text_piece:
                continue

            if t_first is None:
                t_first = time.time()

            # If the server sends cumulative text, only append the new suffix.
            if last_full_text and text_piece.startswith(last_full_text):
                completion_parts.append(text_piece[len(last_full_text) :])
                last_full_text = text_piece
            else:
                completion_parts.append(text_piece)
                last_full_text += text_piece

    t_end = time.time()
    completion_text = "".join(completion_parts)

    if t_first is None:
        # No text deltas received: treat TTFT as full duration.
        t_first = t_end

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    actual_in = len(prompt_ids)
    actual_out = len(completion_ids)
    if actual_out < out_tokens:
        raise RuntimeError(
            f"streamed completion too short: {actual_out} < {out_tokens}. "
            "Try enabling ignore_eos/min_tokens support in your vLLM server, or adjust the request payload."
        )
    if actual_out != out_tokens:
        raise RuntimeError(
            f"streamed completion tokens not exact: {actual_out} != {out_tokens}. "
            "This script requires exact length for recording; check server support for min_tokens/ignore_eos."
        )

    ttft_s = max(t_first - t0, 1e-9)
    e2e_s = max(t_end - t0, 1e-9)
    decode_s = max(t_end - t_first, 1e-9)

    # Align with common definitions used in serving benchmarks.
    qps = 1.0 / e2e_s
    prompt_tps = actual_in / ttft_s
    if out_tokens <= 1:
        decode_tps = float("inf")
        tpot_ms = 0.0
    else:
        decode_tps = (out_tokens - 1) / decode_s
        tpot_ms = (decode_s / (out_tokens - 1)) * 1000.0

    timing = StreamingTiming(
        ttft_ms=ttft_s * 1000.0,
        e2e_ms=e2e_s * 1000.0,
        tpot_ms=tpot_ms,
        qps=qps,
        prompt_tps=prompt_tps,
        decode_tps=decode_tps,
    )
    return completion_text, timing


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Strict vLLM c=1 check: ensure completion tokens reach exactly out_tokens, "
            "record actual token counts, and capture prefix-cache hit rate from /metrics (best-effort)."
        )
    )
    ap.add_argument("--model", required=True, help="served model name or model path accepted by vLLM OpenAI API")
    ap.add_argument("--tokenizer", required=True, help="path for HF tokenizer (usually same as model path)")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000", help="vLLM OpenAI server base URL")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics", help="Prometheus metrics URL")
    ap.add_argument("--in-tokens", type=int, default=20000)
    ap.add_argument("--out-tokens", type=int, default=1000)
    ap.add_argument("--requests", type=int, default=2, help="how many sequential requests to send (cache warmup + measurement)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out-json", required=True, help="write a JSON summary here")
    ap.add_argument("--server-log", help="optional vLLM server log path; used to parse prefix-cache hit rate if /metrics is unavailable")
    ap.add_argument(
        "--stream-metrics",
        action="store_true",
        help="use streaming for the last request to compute TTFT/E2E/TPOT and derive QPS/Prompt&Decode TPS",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("ERROR: transformers not available. Install on VM: pip install transformers", file=sys.stderr)
        raise

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=True)
    prompt = build_prompt_exact_tokens(tok, args.in_tokens)

    # Verify prompt token length
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    actual_in = len(prompt_ids)
    if actual_in != args.in_tokens:
        print(f"WARNING: actual prompt tokens={actual_in} != target={args.in_tokens}", file=sys.stderr)

    completion_text_last: Optional[str] = None
    usage_last: Optional[dict[str, Any]] = None
    actual_out_last: Optional[int] = None
    timing_last: Optional[StreamingTiming] = None

    t0 = time.time()
    for i in range(args.requests):
        is_last = i == args.requests - 1
        if is_last and args.stream_metrics:
            completion_text_last, timing_last = call_vllm_completions_stream_timing(
                args.api_base, args.model, prompt, args.out_tokens, args.timeout, tok
            )
            usage_last = None
            completion_ids = tok.encode(completion_text_last, add_special_tokens=False)
            actual_out_last = len(completion_ids)
        else:
            resp = call_vllm_completions(args.api_base, args.model, prompt, args.out_tokens, args.timeout)
            choice0 = (resp.get("choices") or [{}])[0]
            completion_text_last = choice0.get("text") or ""
            usage_last = resp.get("usage")

            # Prefer server usage if present, else tokenize locally
            if isinstance(usage_last, dict) and isinstance(usage_last.get("completion_tokens"), int):
                actual_out_last = int(usage_last["completion_tokens"])
            else:
                completion_ids = tok.encode(completion_text_last, add_special_tokens=False)
                actual_out_last = len(completion_ids)

            # On the last request we will enforce strict length.
            if is_last:
                if actual_out_last < args.out_tokens:
                    raise RuntimeError(
                        f"completion too short: {actual_out_last} < {args.out_tokens}. "
                        "Try enabling ignore_eos/min_tokens support in your vLLM server, or adjust the request payload."
                    )
                if actual_out_last != args.out_tokens:
                    raise RuntimeError(
                        f"completion tokens not exact: {actual_out_last} != {args.out_tokens}. "
                        "This script requires exact length for recording; check server support for min_tokens/ignore_eos."
                    )

    elapsed_s = time.time() - t0

    hit = get_prefix_cache_hit_rate(args.metrics_url)
    if (hit.gpu_percent is None and hit.cpu_percent is None) and args.server_log:
        try:
            log_text = Path(args.server_log).read_text(encoding="utf-8", errors="ignore")
            hit2 = parse_prefix_cache_hit_rate_from_log_text(log_text)
            if hit2.gpu_percent is not None or hit2.cpu_percent is not None:
                hit = hit2
        except Exception:
            pass

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "api_base": args.api_base,
        "metrics_url": args.metrics_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "in_tokens_target": args.in_tokens,
        "out_tokens_target": args.out_tokens,
        "requests": args.requests,
        "elapsed_s": elapsed_s,
        "actual_input_tokens": actual_in,
        "actual_output_tokens": actual_out_last,
        "usage": usage_last,
        "timing": None
        if timing_last is None
        else {
            "qps": timing_last.qps,
            "prompt_tps": timing_last.prompt_tps,
            "decode_tps": timing_last.decode_tps,
            "ttft_ms": timing_last.ttft_ms,
            "e2e_ms": timing_last.e2e_ms,
            "tpot_ms": timing_last.tpot_ms,
        },
        "prefix_cache_hit_rate_gpu_percent": hit.gpu_percent,
        "prefix_cache_hit_rate_cpu_percent": hit.cpu_percent,
        "notes": "last request is used for strict token equality check",
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
