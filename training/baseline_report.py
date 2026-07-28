#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["F-F", "FL-F", "F-FI", "FL-FI"]
MODELS = ["2B", "8B", "32B"]


def load() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("training/*/*/baseline/*/results/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        ident = m.get("identity", {})
        mean = (m.get("by_section") or {}).get("mean") or {}
        raw = m.get("raw") or {}
        oper = m.get("operational") or {}
        env = m.get("environment") or {}
        rows.append(
            {
                "esperimento": m.get("experiment", "?"),
                "modello": ident.get("model_short", "?"),
                "dataset": ident.get("dataset_code", "?"),
                "target": m.get("target", "?"),
                "n": m.get("n_examples", 0),
                "rougeL": mean.get("rougeL"),
                "rouge1": mean.get("rouge1"),
                "bleu": mean.get("bleu"),
                "bleu_1": mean.get("bleu_1"),
                "raw_rougeL": raw.get("rougeL"),
                "sep_ratio": (m.get("format_compliance") or {}).get("ratio"),
                "vram_gb": env.get("vram_peak_allocated_gb") or env.get("vram_peak_gb"),
                "gen_s": (m.get("generation") or {}).get("seconds"),
                "lat_p50": oper.get("latency_p50_s"),
                "lat_p95": oper.get("latency_p95_s"),
                "mlflow": bool(m.get("mlflow_run_id")),
                "modello_caricato": ident.get("model", "?"),
                "cartella": str(path.parents[2].relative_to(ROOT)),
            }
        )
    return rows


def problemi(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        nome = r["esperimento"]
        if r["n"] != 590:
            out.append(f"{nome}: {r['n']} esempi invece di 590")
        if not r["rougeL"] or r["rougeL"] < 0.02:
            out.append(f"{nome}: rougeL ~0 — generazioni vuote o lingua sbagliata?")
        if r["target"] == "findings_impression" and (r["sep_ratio"] or 0) < 0.5:
            out.append(f"{nome}: solo {r['sep_ratio']:.0%} delle generazioni ha <SEP>")
        if not r["mlflow"]:
            out.append(f"{nome}: non tracciata su MLflow")
        atteso = r["cartella"].split("/")[1].replace("-", "").lower()
        if atteso not in r["modello_caricato"].replace("-", "").lower():
            out.append(f"{nome}: cartella {atteso} ma modello {r['modello_caricato']}")
    return out


def pivot(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for r in rows:
        table.setdefault(r["modello"], {})[r["dataset"]] = r[metric]
    return table


def fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(rows: list[dict]) -> str:
    out = [
        "# Baseline zero-shot",
        "",
        f"{len(rows)} baseline, {rows[0]['n'] if rows else 0} esempi di test ciascuna.",
        "Il modello base, senza fine-tuning, con gli stessi prompt e le stesse",
        "metriche degli esperimenti addestrati.",
        "",
    ]

    for metric, titolo in (("rougeL", "ROUGE-L"), ("bleu", "BLEU-4")):
        table = pivot(rows, metric)
        out += [
            f"## {titolo} per sezione",
            "",
            "| modello | " + " | ".join(DATASETS) + " |",
            "|---|" + "---|" * len(DATASETS),
        ]
        for model in MODELS:
            if model not in table:
                continue
            cells = [fmt(table[model].get(d)) for d in DATASETS]
            out.append(f"| {model} | " + " | ".join(cells) + " |")
        out.append("")

    out += [
        "## Dettaglio",
        "",
        "| esperimento | n | ROUGE-L | BLEU-4 | ROUGE-L integrale | `<SEP>` | VRAM | gen |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        rows,
        key=lambda r: (
            MODELS.index(r["modello"]) if r["modello"] in MODELS else 9,
            DATASETS.index(r["dataset"]) if r["dataset"] in DATASETS else 9,
        ),
    ):
        sep = "n/d" if r["target"] == "findings" else f"{(r['sep_ratio'] or 0):.1%}"
        out.append(
            f"| {r['esperimento']} | {r['n']} | {fmt(r['rougeL'])} | {fmt(r['bleu'])} "
            f"| {fmt(r['raw_rougeL'])} | {sep} "
            f"| {fmt(r['vram_gb'], 1)} GB | {fmt(r['gen_s'], 0)} s |"
        )
    out += [
        "",
        "`<SEP>` non si applica ai dataset con i soli reperti: una sezione sola",
        "non ha separatore.",
        "",
    ]

    guai = problemi(rows)
    out += ["## Controlli", ""]
    out += [f"- ⚠ {p}" for p in guai] if guai else ["Nessun problema rilevato."]
    return "\n".join(out) + "\n"


def bars(rows: list[dict], metric: str, titolo: str) -> str:
    table = pivot(rows, metric)
    models = [m for m in MODELS if m in table]
    valori = [
        v for m in models for v in (table[m].get(d) for d in DATASETS) if v is not None
    ]
    if not valori:
        return ""
    top = max(valori) * 1.15

    W, H, L, B, T = 720, 320, 56, 46, 28
    plot_w, plot_h = W - L - 16, H - B - T
    gruppo = plot_w / len(DATASETS)
    barra = gruppo / (len(models) + 1)
    colori = {"2B": "#4c8bf5", "8B": "#f5a04c", "32B": "#8b5cf6"}

    svg = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">',
        f'<text x="{L}" y="18" class="t">{titolo}</text>',
    ]

    for i in range(5):
        y = T + plot_h * i / 4
        val = top * (1 - i / 4)
        svg.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{W - 16}" y2="{y:.1f}" class="g"/>'
        )
        svg.append(
            f'<text x="{L - 8}" y="{y + 4:.1f}" class="a" '
            f'text-anchor="end">{val:.3f}</text>'
        )

    for gi, dataset in enumerate(DATASETS):
        x0 = L + gi * gruppo
        svg.append(
            f'<text x="{x0 + gruppo / 2:.1f}" y="{H - B + 20}" class="a" '
            f'text-anchor="middle">{dataset}</text>'
        )
        for mi, model in enumerate(models):
            v = table[model].get(dataset)
            if v is None:
                continue
            h = plot_h * v / top
            x = x0 + barra * (mi + 0.5)
            svg.append(
                f'<rect x="{x:.1f}" y="{T + plot_h - h:.1f}" width="{barra * 0.86:.1f}" '
                f'height="{h:.1f}" fill="{colori.get(model, "#888")}" rx="2"/>'
            )
            svg.append(
                f'<text x="{x + barra * 0.43:.1f}" y="{T + plot_h - h - 4:.1f}" '
                f'class="v" text-anchor="middle">{v:.3f}</text>'
            )

    for mi, model in enumerate(models):
        x = L + mi * 84
        svg.append(
            f'<rect x="{x}" y="{H - 14}" width="10" height="10" '
            f'fill="{colori.get(model, "#888")}" rx="2"/>'
        )
        svg.append(f'<text x="{x + 15}" y="{H - 5}" class="a">{model}</text>')

    return "\n".join(svg) + "\n</svg>"


def html(rows: list[dict]) -> str:
    charts = "\n".join(
        f'<div class="c">{bars(rows, m, t)}</div>'
        for m, t in (
            ("rougeL", "ROUGE-L per sezione"),
            ("bleu", "BLEU-4 per sezione"),
            ("rouge1", "ROUGE-1 per sezione"),
        )
        if bars(rows, m, t)
    )
    return f"""<!doctype html>
<meta charset="utf-8"><title>Baseline zero-shot</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:840px;color:#1a1a1a}}
 h1{{font-size:1.4rem}} .c{{margin:2rem 0;border:1px solid #e5e5e5;border-radius:8px;padding:1rem}}
 .t{{font:600 14px system-ui}} .a{{font:11px system-ui;fill:#666}}
 .v{{font:10px system-ui;fill:#333}} .g{{stroke:#eee;stroke-width:1}}
 p{{color:#555}}
 @media(prefers-color-scheme:dark){{body{{background:#111;color:#eee}}
   .c{{border-color:#333}} .a{{fill:#999}} .v{{fill:#ddd}} .g{{stroke:#2a2a2a}}}}
</style>
<h1>Baseline zero-shot</h1>
<p>{len(rows)} modelli base valutati sul test set senza fine-tuning.
Sono il riferimento contro cui si misura il guadagno degli esperimenti.</p>
{charts}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "results")
    args = parser.parse_args(argv)

    rows = load()
    if not rows:
        print("Nessuna baseline trovata: sono gia' state eseguite?", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline_report.md").write_text(markdown(rows), encoding="utf-8")
    (args.out / "baseline_report.html").write_text(html(rows), encoding="utf-8")
    with (args.out / "baseline_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(markdown(rows))
    print(f"scritti in {args.out.relative_to(ROOT)}/: " "baseline_report.{md,csv,html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
