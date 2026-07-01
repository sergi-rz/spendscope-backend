#!/usr/bin/env python3
"""Does parallelizing large multi-chunk /parse over OpenAI actually win wall-clock?

nan.builders (gemma) serializes concurrent requests, so firing chunks in parallel there gains
nothing (and was unstable). OpenAI has a fixed ~8-9s latency floor per call but accepts real
concurrency. Hypothesis: for a large statement split into N chunks, gpt-4o-mini fired K-at-a-time
finishes in ~ceil(N/K) waves, beating gemma's sequential N×(fast-but-serial).

This measures the four scenarios on the SAME chunks and reports wall-clock + cost + failures, so
we can decide whether to route large imports to OpenAI-in-parallel. Uses local .env keys; nothing
is persisted.

    .venv/bin/python scripts/bench_parse_concurrency.py --file samples/extracto-1000-lineas.csv --max-chunks 12 --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

import httpx

BR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BR))

from app.config import settings  # noqa: E402
from app.services.preprocess import prepare  # noqa: E402
from app.routers.parse import _chunk_statement  # noqa: E402
from app.services.prompts import PARSE_SYSTEM  # noqa: E402
from bench_llm import PRICING, call_once  # noqa: E402

SAMPLES = BR.parent / "spendscope-app" / "samples"


async def parse_all(client, model, chunks, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def one(chunk):
        async with sem:
            return await call_once(
                client, model, PARSE_SYSTEM,
                f"Parse this statement into transactions:\n\n{chunk}",
            )

    start = time.perf_counter()
    results = await asyncio.gather(*[one(c) for c in chunks])
    return time.perf_counter() - start, results


def cost_of(model, results):
    price = PRICING.get(model)
    if not price:
        return 0.0
    it = sum(r.in_tokens for r in results)
    ot = sum(r.out_tokens for r in results)
    return it / 1e6 * price["in"] + ot / 1e6 * price["out"]


async def main_async(file: Path, max_chunks: int, conc: int):
    raw = file.read_bytes()
    if file.suffix.lower() in (".csv", ".txt"):
        _, payload = prepare("text", raw.decode("utf-8", "replace"), file.name)
    else:
        _, payload = prepare("image", base64.b64encode(raw).decode(), file.name)
    chunks = _chunk_statement(payload)
    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"\nFile: {file.name}  |  {len(chunks)} chunks")
    print("-" * 72)
    scenarios = conc if conc else [
        ("gemma4  seq",   "gemma4",      1),
        ("gemma4  x6",    "gemma4",      6),
        ("gpt4omini seq",  "gpt-4o-mini", 1),
        ("gpt4omini x6", "gpt-4o-mini", 6),
    ]
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        for label, model, c in scenarios:
            print(f"  → running {label} ...", flush=True)
            wall, results = await parse_all(client, model, chunks, c)
            ok = sum(1 for r in results if r.ok)
            fails = len(results) - ok
            rows = sum(len((r.data.get("transactions") or [])) for r in results if r.ok)
            errs = "; ".join(sorted({r.error[:40] for r in results if not r.ok}))
            print(f"  {label:<14} wall={wall:6.1f}s  ok={ok}/{len(results)}  rows≈{rows:<4} "
                  f"cost=${cost_of(model, results):.4f}" + (f"  ERR: {errs}" if fails else ""),
                  flush=True)
    print("-" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="samples/extracto-1000-lineas.csv")
    ap.add_argument("--max-chunks", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=6)
    # Sweep one model at these concurrencies instead of the default 4-way gemma/gpt comparison.
    ap.add_argument("--sweep", help="comma list of concurrencies, e.g. 2,3,4")
    ap.add_argument("--sweep-model", default="gemma4")
    args = ap.parse_args()
    f = Path(args.file)
    if not f.is_absolute():
        f = BR.parent / args.file
    if not f.exists():
        f = SAMPLES / Path(args.file).name
    conc = None
    if args.sweep:
        m = args.sweep_model
        conc = [(f"{m} x{n}", m, int(n)) for n in args.sweep.split(",")]
    asyncio.run(main_async(f, args.max_chunks, conc or args.concurrency))


if __name__ == "__main__":
    main()
