"""Enrich eval: send real receipts through the deployed /enrich and check the itemization +
per-item categorization. PREMIUM-GATED — only works while prod has REQUIRE_PREMIUM=false (temporary),
or against a staging instance. Reuses the wiring from llm_eval.py.

Answer keys (read from the real tickets in samples/):
  - Mercadona photo: TOTAL 44,16 · 15 product lines (they sum to 44,16 exactly)
  - Carrefour PDF:    SUBTOTAL 219,64 · TOTAL A PAGAR 211,58 · 73 articles, lots of non-product noise
    (3x2 promos with negative lines, discounts, VAT table, loyalty, payment) → stresses the filter.

Run (after the gate is open):  .venv/bin/python scripts/enrich_eval.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

from llm_eval import APP, BASE, UA, category_labels, post

SAMPLES_ROOT = APP / "samples"
OUT = SAMPLES_ROOT / "eval"
USER_ID = "eval-enrich-2026-07"

TICKETS = [
    {"file": "ticket-mercadona-2022.jpeg", "amount": -44.16, "subtotal": 44.16, "total": 44.16, "expected_items": 15},
    {"file": "ticket-carrefour.pdf", "amount": -211.58, "subtotal": 219.64, "total": 211.58, "expected_items": None},
]


def run_one(client, labels, spec):
    path = SAMPLES_ROOT / spec["file"]
    b64 = base64.b64encode(path.read_bytes()).decode()
    res = post(client, "/enrich", {
        "user_id": USER_ID, "input_type": "image", "content": b64,
        "transaction_amount": spec["amount"], "categories": labels,
    })
    items = res.get("items", [])
    item_sum = round(sum(i["amount"] for i in items), 2)
    return {
        "file": spec["file"], "spec": spec, "items": items, "item_sum": item_sum,
        "total_parsed": res.get("total_parsed"), "ticket_total": res.get("ticket_total"),
        "subtotal": res.get("subtotal"), "ticket_date": res.get("ticket_date"),
        "matches_transaction": res.get("matches_transaction"), "movements": res.get("movements"),
    }


def report(results):
    L = ["# Enrich eval — receipt itemization + per-item categorization", ""]
    for r in results:
        s = r["spec"]
        L += [f"## {r['file']}",
              f"- items extraídos: **{len(r['items'])}**"
              + (f" (esperados ~{s['expected_items']})" if s["expected_items"] else ""),
              f"- suma items: **{r['item_sum']}** · subtotal parseado: {r['subtotal']} "
              f"(real {s['subtotal']}) · total parseado: {r['ticket_total']} (real {s['total']})",
              f"- fecha: {r['ticket_date']} · coincide con importe: {r['matches_transaction']}",
              ""]
        # accuracy hints
        if r["subtotal"] is not None:
            L.append(f"- Δ subtotal parseado vs real: {round((r['subtotal'] or 0) - s['subtotal'], 2)}")
        L.append(f"- Δ suma items vs subtotal real: {round(r['item_sum'] - s['subtotal'], 2)}")
        L += ["", "| producto | importe | categoría sugerida |", "|---|---:|---|"]
        for it in r["items"]:
            L.append(f"| {it['description'][:44]} | {it['amount']:.2f} | {it.get('category_suggestion') or ''} |")
        L.append("")
    (OUT / "enrich-report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "enrich-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    labels = category_labels()
    results = []
    with httpx.Client(headers=UA) as client:
        for spec in TICKETS:
            print(f"enrich {spec['file']} ...", flush=True)
            try:
                r = run_one(client, labels, spec)
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                hint = "  (403 = premium gate still ON — set REQUIRE_PREMIUM=false and restart)" if code == 403 else ""
                print(f"  HTTP {code}{hint}")
                return
            results.append(r)
            print(f"  {len(r['items'])} items · suma {r['item_sum']} · total {r['ticket_total']} "
                  f"· match={r['matches_transaction']}")
    report(results)
    print(f"\nWrote enrich-report.md + enrich-results.json to {OUT}")


if __name__ == "__main__":
    main()
