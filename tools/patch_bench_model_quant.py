import re
from pathlib import Path


def patch_file(path: str) -> None:
    p = Path(path)
    src = p.read_text(encoding="utf-8")

    # 1) argparse option
    if "--max_inflight" not in src:
        marker = '    ap.add_argument("--swap_space", type=float, default=4.0)\n'
        if marker not in src:
            raise RuntimeError("Could not find swap_space argparse marker")
        src = src.replace(
            marker,
            marker
            + '    ap.add_argument("--max_inflight", type=int, default=0, '
            + 'help="limit concurrent in-flight requests (0=unlimited)")\n',
            1,
        )

    # 2) run_batch signature
    # Keep this very robust: replace the entire signature block with a canonical
    # one. This also repairs any previous bad substitutions (e.g. stray '\\1').
    run_batch_sig_any = re.compile(
        r"async def run_batch\(\n(?:\s+.*\n)+?\) -> dict:\n",
        re.MULTILINE,
    )
    canonical_sig = (
        "async def run_batch(\n"
        "    engine: AsyncLLMEngine,\n"
        "    prompt_ids: list[int],\n"
        "    sp: SamplingParams,\n"
        "    run_id: str,\n"
        "    concurrency: int,\n"
        "    max_inflight: int = 0,\n"
        ") -> dict:\n"
    )
    src = run_batch_sig_any.sub(canonical_sig, src, count=1)

    # 3) tasks block
    tasks_block = re.compile(
        r"\n\s*tasks = \[\]\n"
        r"\s*for i in range\(concurrency\):\n"
        r"\s*req_id = f\"bench-\{run_id\}-c\{concurrency\}-\{i\}\"\n"
        r"\s*tasks\.append\(asyncio\.create_task\(run_one\(engine, prompt_ids, sp, req_id\)\)\)\n"
        r"\s*results = await asyncio\.gather\(\*tasks\)\n",
        re.MULTILINE,
    )
    if tasks_block.search(src):
        replacement = "\n"
        replacement += "    sem = asyncio.Semaphore(min(max_inflight, concurrency)) if max_inflight and max_inflight > 0 else None\n\n"
        replacement += "    async def _run_one_limited(i: int):\n"
        replacement += "        req_id = f\"bench-{run_id}-c{concurrency}-{i}\"\n"
        replacement += "        if sem is None:\n"
        replacement += "            return await run_one(engine, prompt_ids, sp, req_id)\n"
        replacement += "        await sem.acquire()\n"
        replacement += "        try:\n"
        replacement += "            return await run_one(engine, prompt_ids, sp, req_id)\n"
        replacement += "        finally:\n"
        replacement += "            sem.release()\n\n"
        replacement += "    tasks = [asyncio.create_task(_run_one_limited(i)) for i in range(concurrency)]\n"
        replacement += "    results = await asyncio.gather(*tasks)\n"
        src = tasks_block.sub(replacement, src, count=1)

    # 4) call site
    # Support both single-line and wrapped calls.
    callsite = re.compile(
        r"await run_batch\(engine,\s*prompt_ids,\s*sp,\s*run_id,\s*c\s*\)"
    )
    src = callsite.sub(
        "await run_batch(engine, prompt_ids, sp, run_id, c, max_inflight=args.max_inflight)",
        src,
        count=1,
    )

    # 5) metadata
    if '"swap_space": args.swap_space,' in src and '"max_inflight": args.max_inflight,' not in src:
        src = src.replace(
            '"swap_space": args.swap_space,',
            '"swap_space": args.swap_space,\n                "max_inflight": args.max_inflight,',
            1,
        )

    p.write_text(src, encoding="utf-8")


if __name__ == "__main__":
    patch_file("/mnt/data/work/bench_model_quant.py")
    print("patched /mnt/data/work/bench_model_quant.py")
