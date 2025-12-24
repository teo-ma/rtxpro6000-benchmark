#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class BenchResult:
    model: str
    api_base: str
    in_tokens_target: int
    out_tokens_target: int
    warmups: int
    requests: int
    prompt_tokens_per_request: int
    completion_tokens_per_request: int
    completed: int
    failed: int
    total_elapsed_s: float
    qps: float
    prompt_tps: float
    decode_tps: float
    mean_ttft_ms: Optional[float]
    mean_e2e_ms: Optional[float]
    mean_tpot_ms: Optional[float]


def _now() -> float:
    return time.perf_counter()


def _parse_first_delta_token_time_sse(resp: requests.Response) -> tuple[Optional[float], Optional[float], int]:
    """Parse OpenAI-style SSE stream.

    Returns:
      (ttft_s, total_s, token_events)

    Notes:
    - For /v1/completions streaming, payload lines are usually: `data: {...}`.
    - We treat the first token as the first non-empty delta that carries `text`.
    """
    start = _now()
    ttft_s: Optional[float] = None
    token_events = 0

    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        # OpenAI completions stream shape: {"choices":[{"text":"...", ...}]}
        text = ""
        try:
            text = obj.get("choices", [{}])[0].get("text", "")
        except Exception:
            text = ""

        # Some servers may emit empty text events; we only count non-empty.
        if text:
            token_events += 1
            if ttft_s is None:
                ttft_s = _now() - start

    total_s = _now() - start
    return ttft_s, total_s, token_events


def _get_model_id(api_base: str) -> str:
    r = requests.get(f"{api_base}/v1/models", timeout=10)
    r.raise_for_status()
    data = r.json()
    # Prefer the first model id.
    try:
        return data["data"][0]["id"]
    except Exception:
        return ""


def _build_prompt_exact_tokens(
    tokenizer_path: str,
    in_tokens: int,
) -> str:
    """Build a prompt that tokenizes to exactly `in_tokens` for the given tokenizer.

    We reuse the same approach as strict checker: generate a repetitive tokenizable string.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise SystemExit(
            "transformers is required. Install with: pip install -U transformers"
        ) from e

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # A stable seed text that usually tokenizes consistently.
    # We use lots of ' a' to create many short tokens.
    base = "a"
    # Start with an oversized prompt, then trim via binary search on token count.
    lo, hi = 1, max(4, in_tokens * 4)

    def tok_len(n: int) -> int:
        s = (base + " ") * n
        return len(tok(s, add_special_tokens=False).input_ids)

    while tok_len(hi) < in_tokens:
        hi *= 2
        if hi > 10_000_000:
            raise SystemExit("Failed to reach target token length; tokenizer behaves unexpectedly")

    # Find smallest n with tok_len(n) >= in_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if tok_len(mid) >= in_tokens:
            hi = mid
        else:
            lo = mid + 1

    s = (base + " ") * lo
    ids = tok(s, add_special_tokens=False).input_ids
    if len(ids) < in_tokens:
        raise SystemExit("Internal error: prompt too short")

    # Trim exactly to in_tokens and decode back.
    ids = ids[:in_tokens]
    prompt = tok.decode(ids)

    # Sanity check.
    final_len = len(tok(prompt, add_special_tokens=False).input_ids)
    if final_len != in_tokens:
        # Sometimes decode/encode can drift (rare). Fall back to strict trimming with extra guard.
        # Append a deterministic suffix and retry a few times.
        for _ in range(5):
            prompt = tok.decode(ids)
            final_len = len(tok(prompt, add_special_tokens=False).input_ids)
            if final_len == in_tokens:
                break
        if final_len != in_tokens:
            raise SystemExit(f"Failed to build exact prompt tokens: got {final_len}, expected {in_tokens}")

    return prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming c=1 benchmark (no vllm bench serve).")
    ap.add_argument("--api-base", required=True, help="e.g. http://127.0.0.1:8005")
    ap.add_argument("--model", default=None, help="served model id; if omitted, auto-detect from /v1/models")
    ap.add_argument("--tokenizer", required=True, help="tokenizer path (local folder) for prompt construction")
    ap.add_argument("--in-tokens", type=int, required=True)
    ap.add_argument("--out-tokens", type=int, required=True)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--ignore-eos", action="store_true")
    args = ap.parse_args()

    api_base = args.api_base.rstrip("/")
    model = args.model or _get_model_id(api_base)
    if not model:
        raise SystemExit("Failed to determine model id from /v1/models; pass --model explicitly")

    prompt = _build_prompt_exact_tokens(args.tokenizer, args.in_tokens)

    # Fixed-length generation. For best throughput and determinism:
    # - stream=true to measure TTFT precisely
    # - ignore_eos to reduce early stop risk
    # - temperature=0 (greedy)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": args.out_tokens,
        "min_tokens": args.out_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": bool(args.ignore_eos),
    }

    ttfts = []
    e2es = []

    total_start = _now()
    completed = 0
    failed = 0

    def do_one() -> None:
        nonlocal completed, failed
        try:
            with requests.post(
                f"{api_base}/v1/completions",
                json=payload,
                stream=True,
                timeout=args.timeout,
            ) as resp:
                resp.raise_for_status()
                ttft_s, total_s, _tok_events = _parse_first_delta_token_time_sse(resp)
                if ttft_s is None:
                    # If server doesn't stream token deltas, we can still record e2e.
                    # Keep ttft None for this request.
                    pass
                else:
                    ttfts.append(ttft_s)
                e2es.append(total_s)
                completed += 1
        except Exception:
            failed += 1

    for _ in range(max(0, args.warmups)):
        do_one()

    # Reset stats after warmups.
    ttfts = []
    e2es = []
    completed = 0
    failed = 0

    for _ in range(max(1, args.requests)):
        do_one()

    total_elapsed = _now() - total_start

    prompt_tokens_total = args.in_tokens * completed
    completion_tokens_total = args.out_tokens * completed

    qps = (completed / total_elapsed) if total_elapsed > 0 else 0.0
    prompt_tps = (prompt_tokens_total / total_elapsed) if total_elapsed > 0 else 0.0
    decode_tps = (completion_tokens_total / total_elapsed) if total_elapsed > 0 else 0.0

    mean_ttft_ms = (sum(ttfts) / len(ttfts) * 1000.0) if ttfts else None
    mean_e2e_ms = (sum(e2es) / len(e2es) * 1000.0) if e2es else None

    mean_tpot_ms: Optional[float] = None
    if mean_e2e_ms is not None and mean_ttft_ms is not None:
        denom = max(1, args.out_tokens - 1)
        mean_tpot_ms = max(0.0, (mean_e2e_ms - mean_ttft_ms) / denom)

    res = BenchResult(
        model=model,
        api_base=api_base,
        in_tokens_target=args.in_tokens,
        out_tokens_target=args.out_tokens,
        warmups=args.warmups,
        requests=args.requests,
        prompt_tokens_per_request=args.in_tokens,
        completion_tokens_per_request=args.out_tokens,
        completed=completed,
        failed=failed,
        total_elapsed_s=total_elapsed,
        qps=qps,
        prompt_tps=prompt_tps,
        decode_tps=decode_tps,
        mean_ttft_ms=mean_ttft_ms,
        mean_e2e_ms=mean_e2e_ms,
        mean_tpot_ms=mean_tpot_ms,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res.__dict__, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
