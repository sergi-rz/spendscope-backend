"""LLM eval harness: send each sample through the REAL deployed backend (/extract -> /parse ->
/categorize/batch), exactly as the app does, and record how the LLM standardized and categorized
every movement. Produces a machine report (results.json) and a human report (report.md).

It does NOT test the app UI — only the LLM's parsing + categorization quality. /enrich (tickets,
Amazon screenshots) is premium-gated in production and is intentionally NOT exercised here.

Usage:
  .venv/bin/python scripts/llm_eval.py                 # synthetic + light real files
  .venv/bin/python scripts/llm_eval.py --include-heavy  # also the big/xlsx files (slow)
  .venv/bin/python scripts/llm_eval.py --only es-generico.csv intl-revolut.csv
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.routers.parse import _chunk_statement  # noqa: E402  (identical chunking to the server)

BASE = "https://spendscope.hostsrg.com/api/v1"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124 Safari/537.36"}
SAMPLES = Path(__file__).resolve().parent.parent.parent / "spendscope-app" / "samples" / "eval"
APP = Path(__file__).resolve().parent.parent.parent / "spendscope-app"
USER_ID = "eval-run-2026-07"           # single anon user so rate limits are per-this-run
PACE = 1.2                              # seconds between same-endpoint calls (limit is 60/min)
CATEGORIZE_BATCH = 30
TEXT_EXT = {".csv", ".txt"}


def category_labels() -> list[str]:
    d = json.loads((APP / "SpendScope/Resources/DefaultCategories.json").read_text())
    labels = []
    for c in d["categories"]:
        labels.append(c["label_es"])
        for s in c.get("subcategories", []):
            labels.append(f'{c["label_es"]} / {s["label_es"]}')
    return labels


def post(client: httpx.Client, path: str, payload: dict) -> dict:
    for attempt in range(4):
        r = client.post(f"{BASE}{path}", json=payload, timeout=150)
        if r.status_code == 429:  # rate limited — back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def do_parse(client, entry: dict, path: Path) -> tuple[list[dict], dict]:
    """Return (transactions, meta). Mirrors the app: binary -> /extract, then chunked /parse."""
    meta = {"parse_calls": 0, "modality": "text", "errors": []}
    ext = path.suffix.lower()

    if ext in TEXT_EXT:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        b64 = base64.b64encode(path.read_bytes()).decode()
        ex = post(client, "/extract", {"user_id": USER_ID, "input_type": "image",
                                       "content": b64, "filename": path.name})
        time.sleep(PACE)
        if ex.get("kind") == "image":
            meta["modality"] = "vision"
            res = post(client, "/parse", {"user_id": USER_ID, "input_type": "image",
                                          "content": b64, "filename": path.name})
            meta["parse_calls"] = 1
            time.sleep(PACE)
            return res.get("transactions", []), meta
        text = ex.get("text") or ""

    txns: list[dict] = []
    for chunk in _chunk_statement(text):
        try:
            res = post(client, "/parse", {"user_id": USER_ID, "input_type": "text",
                                          "content": chunk, "filename": path.name})
            txns.extend(res.get("transactions", []))
        except httpx.HTTPStatusError as e:
            meta["errors"].append(f"parse chunk: {e}")
        meta["parse_calls"] += 1
        time.sleep(PACE)
    return txns, meta


def do_categorize(client, txns: list[dict], labels: list[str]) -> list[dict]:
    """Categorize in chunks of 30, forwarding already-suggested names (the #dedup change)."""
    out = list(txns)
    suggested: dict[str, str] = {}   # lowercased name -> display name
    for start in range(0, len(out), CATEGORIZE_BATCH):
        group = out[start:start + CATEGORIZE_BATCH]
        # Wire contract is snake_case (the backend has no camelCase aliases) — matches what the app
        # sends via its convert-to-snake-case encoder.
        items = [{"concept": t["concept"], "amount": t["amount"],
                  "transaction_type": t.get("transaction_type"), "notes": t.get("notes"),
                  "source_category": t.get("source_category")} for t in group]
        res = post(client, "/categorize/batch", {
            "user_id": USER_ID, "items": items, "categories": labels,
            "rejected_suggestions": [], "already_suggested": list(suggested.values()),
            "language": "es",
        })
        by_idx = {r["index"]: r for r in res.get("results", [])}
        for i, t in enumerate(group):
            r = by_idx.get(i, {})
            t["assigned"] = r.get("category")
            t["confidence"] = r.get("confidence")
            sug = r.get("suggested_category")
            t["suggested_category"] = sug
            t["suggested_pattern"] = r.get("suggested_pattern")
            if sug and sug.get("name"):
                suggested.setdefault(sug["name"].lower(), sug["name"])
        time.sleep(PACE)
    return out


def match_token(concept: str, expected: dict) -> tuple[str | None, dict | None]:
    c = concept.lower()
    for tok, info in expected.items():
        if tok in c:
            return tok, info
    return None, None


def score_file(entry, txns, expected):
    """Return per-transaction verdicts and a summary for a scored synthetic file."""
    rows, found_tokens = [], set()
    for t in txns:
        tok, info = match_token(t["concept"], expected)
        assigned = t.get("assigned")
        verdict = "unknown"
        if info:
            found_tokens.add(tok)
            ok = {info["expected"], *info["also_ok"]}
            if assigned in ok:
                verdict = "correct"
            elif info["ambiguous"]:
                verdict = "ambiguous"
            elif assigned is None:
                verdict = "uncategorized"
            else:
                verdict = "wrong"
        rows.append({
            "concept": t["concept"], "amount": t["amount"], "date": t.get("date"),
            "assigned": assigned, "expected": info["expected"] if info else None,
            "also_ok": info["also_ok"] if info else [], "verdict": verdict,
            "confidence": t.get("confidence"),
            "suggested": (t.get("suggested_category") or {}).get("name"),
        })
    missing = [tok for tok in entry.get("tokens", []) if tok not in found_tokens]
    summary = {
        "expected_rows": entry.get("rows"), "parsed_rows": len(txns),
        "missing_tokens": missing,
        "correct": sum(1 for r in rows if r["verdict"] == "correct"),
        "wrong": sum(1 for r in rows if r["verdict"] == "wrong"),
        "uncategorized": sum(1 for r in rows if r["verdict"] == "uncategorized"),
        "ambiguous": sum(1 for r in rows if r["verdict"] == "ambiguous"),
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-heavy", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    labels = category_labels()
    manifest = json.loads((SAMPLES / "manifest.json").read_text())
    expected = json.loads((SAMPLES / "expected.json").read_text())

    results = []
    with httpx.Client(headers=UA) as client:
        for entry in manifest:
            name = entry["file"]
            if args.only and Path(name).name not in args.only:
                continue
            if entry.get("heavy") and not args.include_heavy:
                print(f"skip (heavy)  {name}")
                continue
            path = (SAMPLES / name).resolve()
            if not path.exists():
                print(f"MISSING       {name}")
                continue

            print(f"running       {name} ...", flush=True)
            t0 = time.perf_counter()
            try:
                txns, meta = do_parse(client, entry, path)
                txns = do_categorize(client, txns, labels) if txns else []
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR: {e}")
                results.append({"file": name, "error": str(e)})
                continue
            dt = time.perf_counter() - t0

            rec = {"file": Path(name).name, "source": entry.get("source"),
                   "modality": meta["modality"], "parse_calls": meta["parse_calls"],
                   "seconds": round(dt, 1), "parse_errors": meta["errors"],
                   "transactions": txns}
            if entry.get("scored"):
                rows, summary = score_file(entry, txns, expected)
                rec["scored_rows"], rec["summary"] = rows, summary
                s = summary
                print(f"  parsed {s['parsed_rows']}/{s['expected_rows']} rows | "
                      f"✓{s['correct']} ✗{s['wrong']} ~{s['ambiguous']} ∅{s['uncategorized']} "
                      f"| {dt:.0f}s")
            else:
                print(f"  parsed {len(txns)} rows (unscored) | {dt:.0f}s")
            results.append(rec)

    (SAMPLES / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    write_report(results)
    print(f"\nWrote results.json + report.md to {SAMPLES}")


def write_report(results):
    L = ["# LLM eval — parse + categorize", ""]
    scored = [r for r in results if "summary" in r]
    if scored:
        tc = sum(r["summary"]["correct"] for r in scored)
        tw = sum(r["summary"]["wrong"] for r in scored)
        ta = sum(r["summary"]["ambiguous"] for r in scored)
        tu = sum(r["summary"]["uncategorized"] for r in scored)
        tp = sum(r["summary"]["parsed_rows"] for r in scored)
        te = sum((r["summary"]["expected_rows"] or 0) for r in scored)
        L += [f"**Scored totals:** parsed {tp}/{te} rows · ✓{tc} correct · ✗{tw} wrong · "
              f"~{ta} ambiguous · ∅{tu} uncategorized", ""]

    for r in results:
        L.append(f"## {r['file']}")
        if "error" in r:
            L += [f"- ERROR: {r['error']}", ""]
            continue
        L.append(f"- modality={r['modality']} · parse_calls={r['parse_calls']} · {r['seconds']}s"
                 + (f" · parse_errors={r['parse_errors']}" if r.get("parse_errors") else ""))
        if "summary" in r:
            s = r["summary"]
            L.append(f"- parsed {s['parsed_rows']}/{s['expected_rows']} · ✓{s['correct']} ✗{s['wrong']} "
                     f"~{s['ambiguous']} ∅{s['uncategorized']}"
                     + (f" · MISSING: {', '.join(s['missing_tokens'])}" if s["missing_tokens"] else ""))
            L += ["", "| concept | amount | assigned | expected | verdict |",
                  "|---|---:|---|---|---|"]
            for row in r["scored_rows"]:
                mark = {"correct": "✓", "wrong": "✗ **", "ambiguous": "~", "uncategorized": "∅",
                        "unknown": "?"}[row["verdict"]]
                exp = row["expected"] or ""
                if row["also_ok"]:
                    exp += " (o " + ", ".join(row["also_ok"]) + ")"
                tail = "**" if row["verdict"] == "wrong" else ""
                L.append(f"| {row['concept'][:40]} | {row['amount']:.2f} | {row['assigned'] or '∅'} "
                         f"| {exp} | {mark}{tail} |")
            sug = [(row["concept"][:30], row["suggested"]) for row in r["scored_rows"] if row["suggested"]]
        else:
            L += ["", "| concept | amount | assigned | suggested |", "|---|---:|---|---|"]
            for t in r["transactions"]:
                L.append(f"| {t['concept'][:44]} | {t['amount']:.2f} | {t.get('assigned') or '∅'} "
                         f"| {(t.get('suggested_category') or {}).get('name') or ''} |")
            sug = [(t["concept"][:30], (t.get("suggested_category") or {}).get("name"))
                   for t in r["transactions"] if t.get("suggested_category")]
        if sug:
            L += ["", "**Categorías nuevas propuestas:** "
                  + "; ".join(f"“{n}” ({c})" for c, n in sug)]
        L.append("")

    (SAMPLES / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
