#!/usr/bin/env python3
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class PrefixCacheHitRate:
    gpu_percent: Optional[float]
    cpu_percent: Optional[float]


def _extract_metric_float(metrics_text: str, metric_name: str) -> Optional[float]:
    last_val: Optional[float] = None
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not (line.startswith(metric_name) and (line == metric_name or line[len(metric_name)] in "{ \t")):
            continue
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
            def norm(x: Optional[float]) -> Optional[float]:
                if x is None:
                    return None
                return x * 100.0 if 0.0 <= x <= 1.0 else x

            return PrefixCacheHitRate(gpu_percent=norm(gpu), cpu_percent=norm(cpu))

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


@dataclass(frozen=True)
class RequestTiming:
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float


def build_prompt_exact_tokens(tokenizer, target_tokens: int) -> str:
    seed = (
        "请根据以下材料生成一段中性、客观的说明文字。\n"
        "材料：\n"
        "- 这是一段用于基准测试的长提示词。\n"
        "- 目标是精确控制 token 长度。\n\n"
        "正文：\n"
    )

    text = seed
    chunk = "这是一段用于填充上下文的内容。"

    while True:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            break
        text += (chunk + "\n") * 64

    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _parse_stream_text_chunks(resp: requests.Response) -> tuple[str, float]:
    t0 = time.time()
    t_first: Optional[float] = None

    parts: list[str] = []
    last_full_text = ""

    for raw_line in resp.iter_lines(decode_unicode=True):
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

        if last_full_text and text_piece.startswith(last_full_text):
            parts.append(text_piece[len(last_full_text) :])
            last_full_text = text_piece
        else:
            parts.append(text_piece)
            last_full_text += text_piece

    if t_first is None:
        t_first = time.time()

    return "".join(parts), t_first - t0


def run_one_stream_request(
    *,
    api_base: str,
    model: str,
    prompt: str,
    out_tokens: int,
    timeout_s: float,
    request_id: int,
) -> tuple[int, str, RequestTiming]:
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
    with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        completion_text, ttft_s = _parse_stream_text_chunks(r)

    t_end = time.time()
    e2e_s = max(t_end - t0, 1e-9)
    decode_s = max(e2e_s - ttft_s, 1e-9)

    if out_tokens <= 1:
        tpot_ms = 0.0
    else:
        tpot_ms = (decode_s / (out_tokens - 1)) * 1000.0

    timing = RequestTiming(ttft_ms=ttft_s * 1000.0, e2e_ms=e2e_s * 1000.0, tpot_ms=tpot_ms)
    return request_id, completion_text, timing


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Strict concurrency benchmark for vLLM OpenAI /v1/completions: "
            "exact input token length, exact output token length, streaming-based timing, "
            "and best-effort prefix-cache hit rate capture."
        )
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--in-tokens", type=int, default=10000)
    ap.add_argument("--out-tokens", type=int, default=800)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--total-requests", type=int, default=30, help="number of measured requests (excluding warmup)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception:
        print("ERROR: transformers not available. Install: pip install transformers", file=sys.stderr)
        raise

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=True)
    prompt = build_prompt_exact_tokens(tok, args.in_tokens)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    actual_in = len(prompt_ids)
    if actual_in != args.in_tokens:
        print(f"WARNING: actual prompt tokens={actual_in} != target={args.in_tokens}", file=sys.stderr)

    # Warmup sequentially (helps populate prefix cache)
    for i in range(args.warmup):
        _rid, warm_text, _tim = run_one_stream_request(
            api_base=args.api_base,
            model=args.model,
            prompt=prompt,
            out_tokens=args.out_tokens,
            timeout_s=args.timeout,
            request_id=-(i + 1),
        )
        warm_out = len(tok.encode(warm_text, add_special_tokens=False))
        if warm_out != args.out_tokens:
            raise RuntimeError(f"warmup completion tokens not exact: {warm_out} != {args.out_tokens}")

    # Measured requests with true concurrency
    t_batch0 = time.time()
    results: list[tuple[int, str, RequestTiming]] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(
                run_one_stream_request,
                api_base=args.api_base,
                model=args.model,
                prompt=prompt,
                out_tokens=args.out_tokens,
                timeout_s=args.timeout,
                request_id=i,
            )
            for i in range(args.total_requests)
        ]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(repr(e))

    t_batch1 = time.time()
    wall_s = max(t_batch1 - t_batch0, 1e-9)

    if errors:
        raise RuntimeError(f"{len(errors)} requests failed; first error: {errors[0]}")

    # Strict token equality check for all measured requests
    actual_out_list: list[int] = []
    ttft_ms_list: list[float] = []
    e2e_ms_list: list[float] = []
    tpot_ms_list: list[float] = []

    for _rid, text, timing in results:
        out_n = len(tok.encode(text, add_special_tokens=False))
        actual_out_list.append(out_n)
        if out_n != args.out_tokens:
            raise RuntimeError(f"completion tokens not exact: {out_n} != {args.out_tokens}")
        ttft_ms_list.append(timing.ttft_ms)
        e2e_ms_list.append(timing.e2e_ms)
        tpot_ms_list.append(timing.tpot_ms)

    # Aggregate metrics
    qps = len(results) / wall_s

    mean_ttft_s = _mean([x / 1000.0 for x in ttft_ms_list])
    mean_decode_s = _mean([(e - t) / 1000.0 for e, t in zip(e2e_ms_list, ttft_ms_list)])

    prompt_tps = actual_in / max(mean_ttft_s, 1e-9)
    if args.out_tokens <= 1:
        decode_tps = float("inf")
    else:
        decode_tps = (args.out_tokens - 1) / max(mean_decode_s, 1e-9)

    hit = get_prefix_cache_hit_rate(args.metrics_url)

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "api_base": args.api_base,
        "metrics_url": args.metrics_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "in_tokens_target": args.in_tokens,
        "out_tokens_target": args.out_tokens,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "total_requests": args.total_requests,
        "wall_time_s": wall_s,
        "actual_input_tokens": actual_in,
        "actual_output_tokens": args.out_tokens,
        "per_request": {
            "actual_output_tokens": actual_out_list,
            "ttft_ms": ttft_ms_list,
            "e2e_ms": e2e_ms_list,
            "tpot_ms": tpot_ms_list,
        },
        "summary": {
            "qps": qps,
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
            "ttft_ms": _mean(ttft_ms_list),
            "e2e_ms": _mean(e2e_ms_list),
            "tpot_ms": _mean(tpot_ms_list),
        },
        "prefix_cache_hit_rate_gpu_percent": hit.gpu_percent,
        "prefix_cache_hit_rate_cpu_percent": hit.cpu_percent,
        "notes": "warmup is sequential; measured requests are sent with ThreadPoolExecutor to achieve real concurrency",
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
