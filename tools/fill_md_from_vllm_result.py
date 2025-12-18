#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class BatchMetrics:
    qps: float
    prompt_tps: Optional[float]
    decode_tps: Optional[float]
    ttft_ms: Optional[float]
    e2e_ms: float
    tpot_ms: float


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_batches(result_json_path: Path) -> tuple[dict[int, BatchMetrics], dict[str, Any]]:
    data = json.loads(result_json_path.read_text())
    batches_raw = data.get("batches", {})
    batches: dict[int, BatchMetrics] = {}

    for conc_str, b in batches_raw.items():
        try:
            conc = int(conc_str)
        except ValueError:
            continue

        batches[conc] = BatchMetrics(
            qps=float(b["qps"]),
            prompt_tps=_get(b, "prompt_tps.avg"),
            decode_tps=_get(b, "decode_tps.avg"),
            ttft_ms=_get(b, "ttft_ms.avg"),
            e2e_ms=float(_get(b, "e2e_ms.avg") or b["e2e_ms"]["avg"]),
            tpot_ms=float(_get(b, "tpot_ms.avg") or b["tpot_ms"]["avg"]),
        )

    return batches, data


def format_row(conc: int, m: BatchMetrics, note: str = "") -> str:
    def f(x: Optional[float]) -> str:
        return "" if x is None else f"{x:.3f}"

    note_cell = note.strip()
    return (
        f"| {conc} | {m.qps:.3f} | {f(m.prompt_tps)} | {f(m.decode_tps)} | {f(m.ttft_ms)} | {m.e2e_ms:.3f} | {m.tpot_ms:.3f} | {note_cell} |"
    )


def replace_table_rows(md: str, section_title: str, batches: dict[int, BatchMetrics], note: str = "") -> str:
    lines = md.splitlines()

    # Find section heading
    try:
        section_idx = next(i for i, ln in enumerate(lines) if ln.strip() == f"### {section_title}")
    except StopIteration:
        return md

    # Find table start after section
    table_header_idx = None
    for i in range(section_idx + 1, len(lines)):
        if lines[i].lstrip().startswith("| ") and "并发" in lines[i]:
            table_header_idx = i
            break
        if lines[i].startswith("### "):
            break

    if table_header_idx is None:
        return md

    # Find first data row line (starts with | 10 | etc.)
    data_start = None
    for i in range(table_header_idx + 1, len(lines)):
        if re.match(r"^\|\s*\d+\s*\|", lines[i].strip()):
            data_start = i
            break
    if data_start is None:
        return md

    # Replace data rows until blank line or next heading
    i = data_start
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            break
        if ln.startswith("### "):
            break
        m = re.match(r"^\|\s*(\d+)\s*\|", ln)
        if not m:
            i += 1
            continue
        conc = int(m.group(1))
        if conc in batches:
            lines[i] = format_row(conc, batches[conc], note=note)
        i += 1

    return "\n".join(lines) + ("\n" if md.endswith("\n") else "")


def patch_environment_fields(md: str, meta: dict[str, Any], model_hint: Optional[str]) -> str:
    # Best-effort: replace placeholder fields only if they exist.
    model_value = model_hint or meta.get("model")
    engine_name = _get(meta, "engine.name")
    engine_version = _get(meta, "engine.version")
    gpu = meta.get("gpu")
    ts = meta.get("timestamp")

    def sub(pattern: str, repl: str, text: str) -> str:
        return re.sub(pattern, repl, text, flags=re.MULTILINE)

    out = md
    if model_value:
        out = sub(r"^-\s*模型：.*$", f"- 模型：{model_value}", out)
    if engine_name and engine_version:
        out = sub(r"^-\s*推理引擎：.*$", f"- 推理引擎：{engine_name} {engine_version}", out)
    if ts:
        out = sub(r"^-\s*测试时间（时区）：.*$", f"- 测试时间（时区）：{ts}", out)
    if gpu:
        out = sub(r"^-\s*GPU Driver：.*$", f"- GPU Driver：{gpu}", out)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill benchmark markdown tables from vLLM result.json outputs.")
    ap.add_argument("--template", required=True, help="Path to the markdown template file")
    ap.add_argument("--fp8", help="Path to FP8 result.json")
    ap.add_argument("--fp4", help="Path to FP4 (bitsandbytes) result.json")
    ap.add_argument("--fp8_note", default="", help="Optional text for the 备注 column in FP8 rows")
    ap.add_argument("--fp4_note", default="bitsandbytes 4-bit", help="Optional text for the 备注 column in FP4 rows")
    ap.add_argument("--model_hint", help="Optional model string to put into the template")
    ap.add_argument(
        "--no_env_patch",
        action="store_true",
        help="Do not patch environment/model/engine fields; only fill tables.",
    )
    ap.add_argument("--inplace", action="store_true", help="Write changes back to the template file")
    args = ap.parse_args()

    template_path = Path(args.template)
    md = template_path.read_text()

    meta_any: Optional[dict[str, Any]] = None

    if args.fp8:
        fp8_batches, fp8_meta = load_batches(Path(args.fp8))
        md = replace_table_rows(md, "FP8", fp8_batches, note=args.fp8_note)
        meta_any = fp8_meta

    if args.fp4:
        fp4_batches, fp4_meta = load_batches(Path(args.fp4))
        md = replace_table_rows(md, "FP4", fp4_batches, note=args.fp4_note)
        meta_any = meta_any or fp4_meta

    if meta_any is not None and not args.no_env_patch:
        md = patch_environment_fields(md, meta_any, args.model_hint)

    if args.inplace:
        template_path.write_text(md)
    else:
        print(md)


if __name__ == "__main__":
    main()
