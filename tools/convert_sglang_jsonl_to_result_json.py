#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    concurrency: int
    qps: float
    prompt_tps: float
    decode_tps: float
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float


def _first_existing(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        items.append(json.loads(ln))
    return items


def to_rows(items: list[dict[str, Any]]) -> dict[int, Row]:
    rows: dict[int, Row] = {}
    for it in items:
        # Prefer requested concurrency (max_concurrency) over measured average concurrency.
        conc = int(_first_existing(it, ["max_concurrency", "concurrency"]) or 0)
        if conc <= 0:
            continue

        qps = float(it["request_throughput"])
        prompt_tps = float(it["input_throughput"])
        decode_tps = float(it["output_throughput"])
        ttft_ms = float(it["mean_ttft_ms"])
        e2e_ms = float(it["mean_e2e_latency_ms"])
        tpot_ms = float(it["mean_tpot_ms"])

        rows[conc] = Row(
            concurrency=conc,
            qps=qps,
            prompt_tps=prompt_tps,
            decode_tps=decode_tps,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=tpot_ms,
        )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert sglang.bench_serving JSONL output to the result.json schema used by tools/fill_md_from_vllm_result.py"
        )
    )
    ap.add_argument("--jsonl", required=True, help="Path to bench_sglang.jsonl")
    ap.add_argument("--out", required=True, help="Path to write result.json")
    ap.add_argument("--model", required=True, help="Model identifier (e.g., Qwen/Qwen3-14B-FP8)")
    ap.add_argument("--engine_version", required=True, help="sglang version string")
    ap.add_argument("--gpu", required=True, help="GPU driver string (from nvidia-smi)")
    ap.add_argument("--timestamp", required=True, help="Timestamp string")
    ap.add_argument("--in_tokens", type=int, required=True)
    ap.add_argument("--out_tokens", type=int, required=True)
    args = ap.parse_args()

    items = parse_jsonl(Path(args.jsonl))
    rows = to_rows(items)

    batches: dict[str, Any] = {}
    for conc in sorted(rows.keys()):
        r = rows[conc]
        # Fill p50/p95/max with avg (bench_serving is aggregated already; keep schema stable)
        def stat(v: float) -> dict[str, float]:
            return {"avg": v, "max": v, "p50": v, "p95": v}

        batches[str(conc)] = {
            "concurrency": conc,
            "qps": r.qps,
            "prompt_tps": stat(r.prompt_tps),
            "decode_tps": stat(r.decode_tps),
            "ttft_ms": stat(r.ttft_ms),
            "e2e_ms": stat(r.e2e_ms),
            "tpot_ms": stat(r.tpot_ms),
            "wall_time_s": None,
        }

    out = {
        "batches": batches,
        "engine": {"name": "sglang", "version": args.engine_version},
        "engine_args": {
            "context_length": args.in_tokens + args.out_tokens + 128,
        },
        "gpu": args.gpu,
        "in_tokens": args.in_tokens,
        "out_tokens": args.out_tokens,
        "model": args.model,
        "quantization": "fp8",
        "timestamp": args.timestamp,
        "source": "sglang.bench_serving",
    }

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
