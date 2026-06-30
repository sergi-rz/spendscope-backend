#!/usr/bin/env python3
"""Speed/cost benchmark: gemma4 (nan.builders) vs gpt-4o-mini (OpenAI).

Measures the two things the user actually waits on — statement parsing (/parse) and
categorization (/categorize, single + batch) — calling each model DIRECTLY (no fallback
chain) with the exact production prompts and preprocessing. Reports latency percentiles,
token usage, and a projected cost so we can decide whether OpenAI-as-primary is worth it
(globally, by file size, for a user's first imports, or budget-gated).

Run from the backend repo with its venv (needs httpx + the app's .env):

    .venv/bin/python scripts/bench_llm.py --suite all --reps 3
    .venv/bin/python scripts/bench_llm.py --suite parse --files samples/extracto-1000-lineas.csv --max-chunks 4
    .venv/bin/python scripts/bench_llm.py --models gemma4,gpt-4o-mini --suite categorize

Nothing is persisted. Inputs are the local sample files only.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.services.preprocess import prepare  # noqa: E402
from app.routers.parse import _chunk_statement  # noqa: E402
from app.services.prompts import (  # noqa: E402
    PARSE_SYSTEM,
    CATEGORIZE_SYSTEM,
    CATEGORIZE_BATCH_SYSTEM,
    categorize_user_prompt,
    categorize_batch_user_prompt,
)
from app.services.llm import _build_messages  # noqa: E402

# Where the sample statements live (app repo, sibling of the backend).
SAMPLES_DIR = BACKEND_ROOT.parent / "spendscope-app" / "samples"
CATEGORIES_JSON = (
    BACKEND_ROOT.parent / "spendscope-app" / "SpendScope" / "Resources" / "DefaultCategories.json"
)

# Public OpenAI pricing (USD per 1M tokens). gemma4 runs on nan.builders (EU, flat/self-host) —
# we don't have a per-token rate, so its € column is left blank and only tokens are reported.
PRICING = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}

# Map a model name to the credential set it belongs to (primary = nan.builders, fallback = OpenAI).
def _creds_for(model: str) -> tuple[str, str]:
    primary_models = {m.strip() for m in settings.primary_model_text.split(",")} | {
        m.strip() for m in settings.primary_model_vision.split(",")
    }
    if model in primary_models or not model.startswith("gpt"):
        return settings.primary_base_url, settings.primary_api_key
    return settings.fallback_base_url, settings.fallback_api_key


@dataclass
class CallResult:
    ok: bool
    latency_s: float
    in_tokens: int = 0
    out_tokens: int = 0
    error: str = ""
    data: dict = field(default_factory=dict)


async def call_once(
    client: httpx.AsyncClient,
    model: str,
    system: str,
    user_text: str,
    image_data_uri: str | None = None,
) -> CallResult:
    """One direct chat-completion call, mirroring llm._call_provider but timed + token-aware."""
    base_url, api_key = _creds_for(model)
    body = {
        "model": model,
        "messages": _build_messages(system, user_text, image_data_uri),
        "temperature": 0,
        "max_tokens": settings.llm_max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"

    start = time.perf_counter()
    try:
        resp = await client.post(url, json=body, headers=headers)
        latency = time.perf_counter() - start
        if resp.status_code >= 400:
            return CallResult(False, latency, error=f"HTTP {resp.status_code}: {resp.text[:120]}")
        payload = resp.json()
        usage = payload.get("usage") or {}
        content = payload["choices"][0]["message"]["content"]
        try:
            data = json.loads(content.strip().strip("`").removeprefix("json").strip())
        except Exception:
            data = {}
        return CallResult(
            True, latency,
            in_tokens=int(usage.get("prompt_tokens", 0)),
            out_tokens=int(usage.get("completion_tokens", 0)),
            data=data,
        )
    except Exception as exc:  # noqa: BLE001 — bench tool, report any failure
        return CallResult(False, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((p / 100) * (len(s) - 1))))
    return s[k]


def fmt_stats(label: str, latencies: list[float], in_tok: int, out_tok: int, model: str) -> str:
    if not latencies:
        return f"  {label:<16} {model:<14} — no successful calls"
    p50, p95 = pct(latencies, 50), pct(latencies, 95)
    mean = statistics.mean(latencies)
    cost = ""
    price = PRICING.get(model)
    if price and (in_tok or out_tok):
        usd = in_tok / 1e6 * price["in"] + out_tok / 1e6 * price["out"]
        cost = f"  ${usd:.5f}"
    return (
        f"  {label:<16} {model:<14} "
        f"p50={p50:6.2f}s  p95={p95:6.2f}s  mean={mean:6.2f}s  "
        f"n={len(latencies):>2}  tok(in/out)={in_tok}/{out_tok}{cost}"
    )


def build_category_labels() -> list[str]:
    data = json.loads(CATEGORIES_JSON.read_text())
    labels: list[str] = []
    for cat in data["categories"]:
        parent = cat["label_en"]
        subs = cat.get("subcategories") or []
        if not subs:
            labels.append(parent)
        for sub in subs:
            labels.append(f"{parent} / {sub['label_en']}")
    return labels


# Representative transactions (ES bank wording, mix of clear + ambiguous + income).
SAMPLE_CONCEPTS = [
    ("MERCADONA", -67.82), ("ZARA ESPANA", -49.95), ("REPSOL E.S. 0421", -55.00),
    ("Netflix.com", -12.99), ("FARMACIA SANT JORDI", -8.40), ("BIZUM DE JUAN PEREZ", 20.00),
    ("NOMINA ACME SL", 1850.00), ("IBERDROLA CLIENTES SAU", -64.30), ("AMAZON EU SARL", -23.49),
    ("BAR PEPE", -3.20), ("RENFE VIAJEROS", -45.00), ("DECATHLON", -89.90),
    ("GLOVO BARCELONA", -18.50), ("RECIBO ALQUILER VIVIENDA", -750.00), ("APPLE.COM/BILL", -2.99),
]


async def bench_parse(models: list[str], files: list[Path], reps: int, max_chunks: int | None) -> None:
    print("\n" + "=" * 78)
    print("SUITE: /parse  (statement standardization — latency is per full file)")
    print("=" * 78)
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        for path in files:
            if not path.exists():
                print(f"\n[skip] {path} not found")
                continue
            raw = path.read_bytes()
            if path.suffix.lower() in (".csv", ".txt"):
                modality, payload = prepare("text", raw.decode("utf-8", "replace"), path.name)
            else:
                b64 = base64.b64encode(raw).decode()
                modality, payload = prepare("file", b64, path.name)
            if modality != "text":
                print(f"\n[skip] {path.name}: vision modality not in this suite")
                continue
            chunks = _chunk_statement(payload)
            if max_chunks:
                chunks = chunks[:max_chunks]
            print(f"\n• {path.name}  ({len(raw)} bytes, {len(chunks)} chunk(s), {reps} rep(s))")
            for model in models:
                file_latencies: list[float] = []  # one entry per rep = total time for the file
                in_tok = out_tok = rows = fails = 0
                for _ in range(reps):
                    rep_total = 0.0
                    for chunk in chunks:
                        user = f"Parse this statement into transactions:\n\n{chunk}"
                        r = await call_once(client, model, PARSE_SYSTEM, user)
                        rep_total += r.latency_s
                        in_tok += r.in_tokens
                        out_tok += r.out_tokens
                        if r.ok:
                            rows += len(r.data.get("transactions", []) or [])
                        else:
                            fails += 1
                    file_latencies.append(rep_total)
                line = fmt_stats(path.name[:16], file_latencies, in_tok, out_tok, model)
                print(line + (f"  rows≈{rows // max(1, reps)}" if rows else "") +
                      (f"  FAILS={fails}" if fails else ""))


async def bench_categorize(models: list[str], reps: int) -> None:
    labels = build_category_labels()
    print("\n" + "=" * 78)
    print(f"SUITE: /categorize  ({len(SAMPLE_CONCEPTS)} concepts, {len(labels)} categories)")
    print("=" * 78)
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        # --- Single (/categorize): the per-transaction path, one call each ---
        print("\n[single] one LLM call per transaction (worst case, no cache)")
        for model in models:
            latencies: list[float] = []
            in_tok = out_tok = fails = 0
            for _ in range(reps):
                for concept, amount in SAMPLE_CONCEPTS:
                    prompt = categorize_user_prompt(concept, amount, None, None, labels, [], "es")
                    r = await call_once(client, model, CATEGORIZE_SYSTEM, prompt)
                    latencies.append(r.latency_s)
                    in_tok += r.in_tokens
                    out_tok += r.out_tokens
                    fails += 0 if r.ok else 1
            print(fmt_stats("per-concept", latencies, in_tok, out_tok, model) +
                  (f"  FAILS={fails}" if fails else ""))

        # --- Batch (/categorize/batch): what the app actually uses on import ---
        print("\n[batch] all concepts in ONE call (the real import path)")

        class _Item:
            def __init__(self, concept, amount):
                self.concept, self.amount = concept, amount
                self.transaction_type = self.notes = self.source_category = None

        items = [_Item(c, a) for c, a in SAMPLE_CONCEPTS]
        for model in models:
            latencies: list[float] = []
            in_tok = out_tok = fails = 0
            for _ in range(reps):
                prompt = categorize_batch_user_prompt(items, labels, [], "es")
                r = await call_once(client, model, CATEGORIZE_BATCH_SYSTEM, prompt)
                latencies.append(r.latency_s)
                in_tok += r.in_tokens
                out_tok += r.out_tokens
                fails += 0 if r.ok else 1
            n_ret = len(r.data.get("results", []) or []) if r.ok else 0
            print(fmt_stats(f"{len(items)}-in-one", latencies, in_tok, out_tok, model) +
                  (f"  returned={n_ret}" if n_ret else "") + (f"  FAILS={fails}" if fails else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="gemma4,gpt-4o-mini")
    ap.add_argument("--suite", choices=["parse", "categorize", "all"], default="all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--files", nargs="*", help="parse sample paths (default: a small/large mix)")
    ap.add_argument("--max-chunks", type=int, default=None, help="cap chunks/file (large statements)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        _, key = _creds_for(model)
        if not key:
            print(f"⚠️  no API key for {model} — check .env (PRIMARY_API_KEY / FALLBACK_API_KEY)")
            sys.exit(1)

    if args.files:
        files = [Path(f) if Path(f).is_absolute() else (BACKEND_ROOT.parent / f) for f in args.files]
        files = [
            f if f.exists() else (SAMPLES_DIR / Path(f).name) for f in files
        ]
    else:
        files = [
            SAMPLES_DIR / "extracto-ejemplo.csv",
            SAMPLES_DIR / "Gastos paula sergio - Extracto 2026.csv",
            SAMPLES_DIR / "extracto-1000-lineas.csv",
        ]

    print(f"Models: {models}  |  reps: {args.reps}  |  suite: {args.suite}")
    print(f"Primary base: {settings.primary_base_url}  |  Fallback base: {settings.fallback_base_url}")

    async def run() -> None:
        if args.suite in ("parse", "all"):
            cap = args.max_chunks if args.max_chunks is not None else (4 if args.suite == "all" else None)
            await bench_parse(models, files, args.reps, cap)
        if args.suite in ("categorize", "all"):
            await bench_categorize(models, args.reps)

    asyncio.run(run())
    print("\nDone. (Latency is wall-clock incl. network; gemma4 = EU/nan.builders, gpt = OpenAI.)")


if __name__ == "__main__":
    main()
