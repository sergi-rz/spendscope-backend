"""Generate diverse synthetic bank-statement samples for the LLM eval harness (scripts/llm_eval.py).

Each merchant in the pool carries a UNIQUE lowercase `token` (always present in its concept string)
and the category it SHOULD land in (my answer key). The same merchants are rendered into several
bank layouts — different column names/order, separators, date formats, decimal marks, sign vs
split debit/credit columns, quoted fields, metadata headers, and an "export from another app" with
its own category column (#66). This tests BOTH statement standardization (/parse) and categorization
(/categorize) breadth, with minimal concept overlap between files so the concept cache doesn't mask
categorization quality.

Writes into ../spendscope-app/samples/eval/:
  - the sample files
  - manifest.json  (what to send, and in which language)
  - expected.json  (token -> expected category label + notes), consumed by the harness to score

Run:  .venv/bin/python scripts/make_eval_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "spendscope-app" / "samples" / "eval"


# token, concept (raw, bank-style, contains token), amount(EUR, signed), expected ES full label,
# also_ok (other defensible labels), ambiguous (don't count as wrong — just report for my review)
def T(token, concept, amount, expected, also_ok=None, ambiguous=False):
    return {
        "token": token, "concept": concept, "amount": amount,
        "expected": expected, "also_ok": also_ok or [], "ambiguous": ambiguous,
    }


# --- Spanish merchants ---------------------------------------------------------
POOL_ES = [
    T("mercadona", "COMPRA TARJ. 5402 MERCADONA OFRA", -42.30, "Alimentación / Supermercado"),
    T("carrefour", "CARREFOUR EXPRESS MADRID CENTRO", -18.75, "Alimentación / Supermercado"),
    T("bar pepe", "BAR PEPE COMIDAS", -12.40, "Restaurantes / Restaurante",
      also_ok=["Restaurantes / Cafeterías"], ambiguous=True),
    T("glovo", "GLOVOAPP271 BARCELONA", -23.10, "Restaurantes / Comida para llevar",
      also_ok=["Restaurantes / Restaurante"]),
    T("iberdrola", "RECIBO IBERDROLA CLIENTES SAU", -74.20, "Suministros / Luz"),
    T("isabel ii", "RECIBO CANAL DE ISABEL II GESTION", -31.05, "Suministros / Agua"),
    T("vodafone", "VODAFONE ESPANA SAU CUOTA", -49.99, "Suministros / Internet y móvil"),
    T("alquiler piso", "TRANSF ALQUILER PISO C/ MAYOR", -850.00, "Vivienda / Alquiler"),
    T("prestamo hipotec", "CUOTA PRESTAMO HIPOTECARIO 0012", -620.00, "Vivienda / Hipoteca"),
    T("repsol", "REPSOL E.S. AP-7 KM45", -60.00, "Vehículo / Combustible y carga"),
    T("easypark", "EASYPARK ESPANA SLU 0098", -3.20, "Vehículo / Parking y peajes"),
    T("renfe", "RENFE VIAJEROS AVE MADRID-VLC", -45.00, "Transporte / Transporte público",
      also_ok=["Vacaciones / Transporte"], ambiguous=True),
    T("bono metro", "EMT MADRID BONO METRO 10 VIAJES", -21.50, "Transporte / Transporte público"),
    T("farmacia", "FARMACIA GARCIA COLL", -9.80, "Salud / Farmacia"),
    T("clinica dental", "CLINICA DENTAL SONRIE MAS", -120.00, "Salud / Tratamientos"),
    T("basic fit", "BASIC FIT VALENCIA CAMPANAR", -19.99, "Salud / Gimnasio"),
    T("peluqueria", "PELUQUERIA LOOKS ESTILISTAS", -25.00, "Estética / Peluquería"),
    T("mapfre hogar", "RECIBO SEGURO MAPFRE HOGAR", -28.60, "Seguros / Hogar y vida"),
    T("mutua madrilena", "MUTUA MADRILENA SEGURO AUTO", -46.00, "Vehículo / Seguro",
      also_ok=["Seguros / Hogar y vida"]),
    T("comision mant", "COMISION MANTENIMIENTO CUENTA", -3.00, "Finanzas e impuestos / Comisiones bancarias"),
    T("reintegro cajero", "REINTEGRO CAJERO 4B OFICINA", -50.00, "Traspasos internos",
      also_ok=["Finanzas e impuestos / Comisiones bancarias"], ambiguous=True),
    T("bizum enviado", "BIZUM ENVIADO A MARIA G.", -20.00, "Transferencias"),
    T("cuenta ahorro", "TRASPASO A CUENTA AHORRO PROPIA", -200.00, "Traspasos internos",
      also_ok=["Ahorros"]),
    T("nomina acme", "NOMINA ACME TECHNOLOGIES SL", 2100.00, "Ingresos / Nómina"),
    T("zalando", "DEVOLUCION COMPRA ZALANDO", 39.95, "Devoluciones"),
    T("veterinario", "VETERINARIO MASCOTAS FELIZ", -60.00, "Mascota / Veterinario"),
    T("kiwoko", "TIENDA ANIMALES KIWOKO SANT", -22.00, "Mascota / Alimentación",
      also_ok=["Mascota"]),
    T("colegio", "COLEGIO SAN JOSE CUOTA MENSUAL", -180.00, "Niños / Colegio y material"),
    T("cruz roja", "DONACION CRUZ ROJA ESPANOLA", -10.00, "Donaciones"),
    T("decathlon", "DECATHLON VALENCIA MN4", -55.20, "Compras / Deporte"),
    T("zara", "ZARA ESPANA COMPRA TARJ 4587", -39.95, "Compras / Ropa y calzado"),
    T("ikea", "IKEA VALENCIA ALFAFAR", -128.40, "Hogar / Muebles y decoración"),
    T("agencia tributaria", "AEAT AGENCIA TRIBUTARIA IRPF", -240.00, "Finanzas e impuestos / Impuestos y tasas"),
]

# --- International merchants (for the US/UK/DE/Revolut layouts) -----------------
POOL_INTL = [
    T("tesco", "TESCO STORES 3421 LONDON", -54.10, "Alimentación / Supermercado"),
    T("aldi", "ALDI SUED FILIALE MUENCHEN", -33.20, "Alimentación / Supermercado"),
    T("starbucks", "STARBUCKS COFFEE BERLIN HBF", -5.60, "Restaurantes / Cafeterías"),
    T("mcdonald", "MCDONALDS 5567 LONDON", -9.30, "Restaurantes / Comida para llevar",
      also_ok=["Restaurantes / Restaurante"]),
    T("help.uber", "UBER *TRIP HELP.UBER.COM", -14.30, "Transporte / Taxi y VTC"),
    T("eats amsterdam", "UBER *EATS AMSTERDAM", -21.80, "Restaurantes / Comida para llevar"),
    T("shell", "SHELL SERVICE STATION A40", -62.00, "Vehículo / Combustible y carga"),
    T("netflix", "NETFLIX.COM AMSTERDAM", -13.99, "Suscripciones / Streaming"),
    T("spotify", "SPOTIFY P0A1B2C3 STOCKHOLM", -10.99, "Suscripciones / Streaming"),
    T("apple.com", "APPLE.COM/BILL ITUNES", -2.99, "Suscripciones / Apps",
      also_ok=["Suscripciones"]),
    T("amzn mktp", "AMZN MKTP DE*2B34XY", -34.90, "Compras",
      also_ok=["Compras / Ropa y calzado", "Hogar / Muebles y decoración"], ambiguous=True),
    T("ryanair", "RYANAIR ONLINE DUBLIN", -49.99, "Vacaciones / Transporte"),
    T("booking", "BOOKING.COM HOTEL AMSTERDAM", -180.00, "Vacaciones / Alojamiento"),
    T("h&m", "H&M HENNES MAURITZ 88 BERLIN", -29.95, "Compras / Ropa y calzado"),
    T("steamgames", "PAYPAL *STEAMGAMES VALVE", -19.99, "Suscripciones / Gaming",
      also_ok=["Ocio / Juegos y libros"]),
    T("dm drogerie", "DM DROGERIE MARKT MUENCHEN", -18.40, "Salud / Farmacia",
      also_ok=["Estética / Cosmética", "Hogar / Limpieza"], ambiguous=True),
    T("salary", "SALARY GLOBEX CORP PAYROLL", 2600.00, "Ingresos / Nómina"),
    T("lidl", "LIDL SAGT DANKE FILIALE 77", -27.65, "Alimentación / Supermercado"),
]


def es_amount(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def us_amount(x: float) -> str:
    return f"{x:,.2f}"


def iso(day: int) -> str:
    return f"2026-06-{day:02d}"


def dmy(day: int) -> str:
    return f"{day:02d}/06/2026"


def mdy(day: int) -> str:
    return f"06/{day:02d}/2026"


def dot_de(day: int) -> str:
    return f"{day:02d}.06.2026"


# --- layout formatters (each returns file text) --------------------------------

def fmt_generic_es(rows):
    out = ["Fecha,Concepto,Importe,Saldo"]
    bal = 3000.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        out.append(f'{dmy(2 + i)},{r["concept"]},{es_amount(r["amount"])},{es_amount(bal)}')
    return "\n".join(out) + "\n"


def fmt_santander_semicolon(rows):
    out = ["FECHA OPERACIÓN;FECHA VALOR;CONCEPTO;IMPORTE;SALDO"]
    bal = 3000.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        d = dmy(2 + i)
        out.append(f'{d};{d};{r["concept"]};{es_amount(r["amount"])};{es_amount(bal)}')
    return "\n".join(out) + "\n"


def fmt_caixabank_metadata(rows):
    out = [
        "Titular: JUAN EJEMPLO PEREZ",
        "Cuenta: ES91 2100 0418 4502 0005 1332",
        "Periodo: 01/06/2026 - 30/06/2026",
        "Divisa: EUR",
        "",
        "Fecha;Concepto;Importe;Saldo",
    ]
    bal = 3000.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        out.append(f'{dmy(2 + i)};{r["concept"]};{es_amount(r["amount"])};{es_amount(bal)}')
    return "\n".join(out) + "\n"


def fmt_bbva_quoted(rows):
    # Concepts wrapped in quotes; a couple carry commas to test quoted parsing.
    out = ['"Fecha";"Concepto";"Importe";"Divisa";"Disponible"']
    bal = 3000.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        concept = r["concept"] + (", RECIBO DOMICILIADO" if i % 3 == 0 else "")
        out.append(f'"{dmy(2 + i)}";"{concept}";"{es_amount(r["amount"])}";"EUR";"{es_amount(bal)}"')
    return "\n".join(out) + "\n"


def fmt_chase_us(rows):
    out = ["Details,Posting Date,Description,Amount,Type,Balance"]
    bal = 5000.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        details = "CREDIT" if r["amount"] > 0 else "DEBIT"
        typ = "ACH_CREDIT" if r["amount"] > 0 else "DEBIT_CARD"
        out.append(f'{details},{mdy(2 + i)},{r["concept"]},{us_amount(r["amount"])},{typ},{us_amount(bal)}')
    return "\n".join(out) + "\n"


def fmt_uk_split(rows):
    # No signed amount: separate Money In / Money Out columns — the model must infer direction.
    out = ["Date,Counter Party,Reference,Money In,Money Out,Balance"]
    bal = 4000.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        mi = us_amount(r["amount"]) if r["amount"] > 0 else ""
        mo = us_amount(-r["amount"]) if r["amount"] < 0 else ""
        out.append(f'{dmy(2 + i)},{r["concept"]},Ref{1000 + i},{mi},{mo},{us_amount(bal)}')
    return "\n".join(out) + "\n"


def fmt_n26_de(rows):
    out = ['"Booking Date","Value Date","Partner Name","Type","Payment Reference","Amount (EUR)"']
    for i, r in enumerate(rows):
        typ = "Income" if r["amount"] > 0 else "Presentment"
        out.append(f'"{iso(2 + i)}","{iso(2 + i)}","{r["concept"]}","{typ}","Ref {i}","{us_amount(r["amount"])}"')
    return "\n".join(out) + "\n"


def fmt_revolut(rows):
    out = ["Type,Product,Started Date,Completed Date,Description,Amount,Fee,Currency,State,Balance"]
    bal = 1200.0
    for i, r in enumerate(rows):
        bal += r["amount"]
        typ = "TOPUP" if r["amount"] > 0 else "CARD_PAYMENT"
        out.append(
            f'{typ},Current,{iso(2 + i)} 08:1{i%10}:00,{iso(2 + i)} 08:1{i%10}:05,'
            f'{r["concept"]},{us_amount(r["amount"])},0.00,EUR,COMPLETED,{us_amount(bal)}'
        )
    return "\n".join(out) + "\n"


def fmt_plaintext(rows):
    # A pasted blob, no header, inline dates and trailing euro sign.
    lines = ["Movimientos de la cuenta - últimos días", ""]
    for i, r in enumerate(rows):
        lines.append(f'{dmy(2 + i)}   {r["concept"]}   {es_amount(r["amount"])} €')
    return "\n".join(lines) + "\n"


def fmt_ynab_export(rows):
    # Export from another finance app WITH its own category column (#66 source_category). The app's
    # labels are deliberately different from ours, to test mapping rather than verbatim reuse.
    app_cat = {
        "mercadona": "Groceries", "carrefour": "Groceries", "glovo": "Restaurants",
        "iberdrola": "Bills", "vodafone": "Bills", "repsol": "Auto & Transport: Gas",
        "netflix": "Subscriptions", "farmacia": "Health", "nomina acme": "Income: Paycheck",
        "zara": "Shopping: Clothing", "decathlon": "Shopping", "colegio": "Kids",
    }
    out = ["Date,Payee,Category,Memo,Outflow,Inflow"]
    for i, r in enumerate(rows):
        cat = ""
        for tok, c in app_cat.items():
            if tok in r["concept"].lower():
                cat = c
                break
        outflow = us_amount(-r["amount"]) if r["amount"] < 0 else ""
        inflow = us_amount(r["amount"]) if r["amount"] > 0 else ""
        out.append(f'{dmy(2 + i)},{r["concept"]},{cat},,{outflow},{inflow}')
    return "\n".join(out) + "\n"


# file -> (formatter, slice of pool, language). Slices are mostly disjoint so categorization breadth
# is exercised across files without the concept cache short-circuiting later files.
FILES = [
    ("es-generico.csv",        fmt_generic_es,        POOL_ES[0:16],   "es"),
    ("es-santander-pyc.csv",   fmt_santander_semicolon, POOL_ES[16:32], "es"),
    ("es-caixabank-cabecera.csv", fmt_caixabank_metadata, POOL_ES[4:18], "es"),
    ("es-bbva-comillas.csv",   fmt_bbva_quoted,       POOL_ES[8:24],   "es"),
    ("es-texto-pegado.txt",    fmt_plaintext,         POOL_ES[0:12],   "es"),
    ("intl-chase-us.csv",      fmt_chase_us,          POOL_INTL[0:12], "es"),
    ("intl-uk-debito-credito.csv", fmt_uk_split,      POOL_INTL[6:18], "es"),
    ("intl-n26-de.csv",        fmt_n26_de,            POOL_INTL[0:9],  "es"),
    ("intl-revolut.csv",       fmt_revolut,           POOL_INTL[9:18], "es"),
    ("export-ynab-categorias.csv", fmt_ynab_export,   POOL_ES[0:14],   "es"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    expected = {}
    for r in POOL_ES + POOL_INTL:
        expected[r["token"]] = {
            "concept": r["concept"], "amount": r["amount"], "expected": r["expected"],
            "also_ok": r["also_ok"], "ambiguous": r["ambiguous"],
        }

    manifest = []
    for name, fmt, rows, lang in FILES:
        (OUT / name).write_text(fmt(rows), encoding="utf-8")
        manifest.append({
            "file": name, "language": lang, "source": "synthetic",
            "scored": True, "heavy": False,
            "rows": len(rows), "tokens": [r["token"] for r in rows],
            "note": f"{len(rows)} rows, layout={fmt.__name__.replace('fmt_', '')}",
        })

    # Pre-existing real files in samples/ — categorized but NOT auto-scored (unknown ground truth);
    # I review these by hand. Heavy ones are opt-in in the harness.
    existing = [
        ("../extracto-ejemplo.csv", "es", False, False),
        ("../extracto-banco-convertido.csv", "es", False, False),
        ("../Gastos paula sergio - Extracto 2026.csv", "es", False, False),
        ("../export-otra-app.csv", "es", False, False),
        ("../extracto-mini-carrefour.xlsx", "es", False, False),
        ("../2026Y-06M-16D-21_11_22-Últimos movimientos.xlsx", "es", False, True),
        ("../2026Y-06M-16D-21_13_01-Últimos movimientos.xlsx", "es", False, True),
        ("../extracto-1000-lineas.csv", "es", False, True),
    ]
    for rel, lang, scored, heavy in existing:
        if (OUT / rel).exists():
            manifest.append({
                "file": rel, "language": lang, "source": "existing",
                "scored": scored, "heavy": heavy, "note": "real file already in samples/",
            })

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(FILES)} synthetic files + manifest ({len(manifest)} entries) + expected.json to {OUT}")


if __name__ == "__main__":
    main()
