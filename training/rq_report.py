#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASETS = ["F-F", "FL-F", "F-FI", "FL-FI"]
MODELLI = ["2B", "8B", "32B"]
MODALITA = ["lora", "qlora"]
COLORI = {"2B": "#4c8bf5", "8B": "#f5a04c", "32B": "#8b5cf6"}

VISTE = {"F-F": "1", "F-FI": "1", "FL-F": "2", "FL-FI": "2"}
TARGET = {"F-F": "R", "FL-F": "R", "F-FI": "R+I", "FL-FI": "R+I"}

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HA_GRAFICI = True
except ImportError:
    HA_GRAFICI = False


def carica(training_root: Path) -> list[dict]:
    righe = []
    for path in sorted(training_root.glob("*/*/*/*/results/results.json")):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # noqa: PERF203
            print(f"  ! {path}: illeggibile ({exc})", file=sys.stderr)
            continue
        ident = r.get("identity", {})
        best = r.get("best") or {}
        cfg = r.get("config", {})
        sezioni = best.get("sections") or {}
        righe.append(
            {
                "esperimento": r.get("experiment", path.parts[-3]),
                "modello": ident.get("model_short", "?"),
                "modalita": ident.get("mode", "?"),
                "dataset": ident.get("dataset_code", "?"),
                "viste": VISTE.get(ident.get("dataset_code", ""), "?"),
                "target": TARGET.get(ident.get("dataset_code", ""), "?"),
                "stato": r.get("status", "?"),
                "early_stopped": bool(r.get("early_stopped")),
                "epoche": r.get("epochs_completed"),
                "max_epoche": cfg.get("max_epochs"),
                "valutazioni": r.get("n_evaluations"),
                "best_epoca": best.get("epoch"),
                "best_valore": best.get("value"),
                "val_loss": best.get("val_loss"),
                "metriche": best.get("metrics") or {},
                "findings": (sezioni.get("findings") or {}),
                "impression": (sezioni.get("impression") or {}),
                "ore": (r.get("timing", {}).get("wall_clock_s") or 0) / 3600,
                "vram": r.get("environment", {}).get("vram_peak_allocated_gb"),
                "gpu": (r.get("environment", {}).get("gpus") or ["?"])[0],
                "curve": r.get("curves", {}),
            }
        )
    return righe


def baseline(training_root: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for path in training_root.glob("*/*/baseline/*/results/metrics.json"):
        m = json.loads(path.read_text(encoding="utf-8"))
        ident = m.get("identity", {})
        out[(ident.get("model_short"), ident.get("dataset_code"))] = {
            "media": (m.get("by_section") or {}).get("mean") or {},
            "findings": ((m.get("by_section") or {}).get("findings")) or {},
        }
    return out


def valore(riga: dict, metrica: str, sezione: str | None = None) -> float | None:
    fonte = riga["metriche"] if sezione is None else riga.get(sezione) or {}
    v = fonte.get(metrica)
    return float(v) if isinstance(v, (int, float)) else None


def tabella(intestazioni: list[str], righe: list[list]) -> str:
    if not righe:
        return "_nessun dato_\n"
    out = ["| " + " | ".join(intestazioni) + " |", "|" + "---|" * len(intestazioni)]
    out += [
        "| " + " | ".join("—" if c is None else str(c) for c in r) + " |" for r in righe
    ]
    return "\n".join(out) + "\n"


def num(v, cifre: int = 4, suffisso: str = "") -> str:
    return "—" if v is None else f"{v:.{cifre}f}{suffisso}"


def segno(v, cifre: int = 4) -> str:
    return "—" if v is None else f"{v:+.{cifre}f}"


def confronto_appaiato(
    righe: list[dict],
    variabile,
    chiave,
    etichetta: str,
    a: str,
    b: str,
    metrica: str,
    sezione: str | None = None,
) -> tuple[str, dict]:
    indice = {(chiave(r), variabile(r)): r for r in righe}
    coppie = sorted(
        (
            (r, indice[(k, b)])
            for (k, v), r in indice.items()
            if v == a and (k, b) in indice
        ),
        key=lambda p: p[0]["esperimento"],
    )

    dati, delta = [], []
    for ra, rb in coppie:
        va, vb = valore(ra, metrica, sezione), valore(rb, metrica, sezione)
        d = (vb - va) if (va is not None and vb is not None) else None
        if d is not None:
            delta.append(d)
        dati.append(
            [
                " · ".join(chiave(ra)),
                num(va),
                num(vb),
                segno(d),
                "—" if d is None else (b if d > 0 else a),
            ]
        )
    testo = tabella([etichetta, a, b, f"Δ ({b}−{a})", "migliore"], dati)
    grezzi = [(" · ".join(chiave(ra)), valore(ra, metrica, sezione),
               valore(rb, metrica, sezione)) for ra, rb in coppie]
    sintesi = {
        "n": len(delta),
        "media": statistics.fmean(delta) if delta else None,
        "mediana": statistics.median(delta) if delta else None,
        "vittorie_b": sum(1 for d in delta if d > 0),
        "coppie": grezzi,
    }
    return testo, sintesi


def salva(fig, out: Path, nome: str) -> str:
    fig.tight_layout()
    fig.savefig(out / nome, dpi=140)
    plt.close(fig)
    return f"\n![{nome}]({nome})\n"


def g_tempi(righe: list[dict], out: Path) -> str:
    dati = sorted([r for r in righe if r["ore"]], key=lambda r: r["ore"])
    if not dati:
        return ""
    fig, ax = plt.subplots(figsize=(9, max(4, len(dati) * 0.28)))
    ax.barh(
        [r["esperimento"] for r in dati],
        [r["ore"] for r in dati],
        color=[COLORI.get(r["modello"], "#888") for r in dati],
    )
    ax.set_xlabel("ore")
    ax.set_title("Durata di ogni fine-tuning")
    ax.grid(axis="x", alpha=0.3)
    return salva(fig, out, "rq1_tempi.png")


def g_early(righe: list[dict], out: Path) -> str:
    dati = [r for r in righe if r["best_epoca"] is not None and r["epoche"]]
    if not dati:
        return ""
    dati.sort(key=lambda r: (r["epoche"], r["best_epoca"]))
    fig, ax = plt.subplots(figsize=(10, max(4, len(dati) * 0.32)))
    y = range(len(dati))
    ax.barh(list(y), [r["epoche"] for r in dati], color="#dfe4ea",
            edgecolor="#c3c9d1", label="epoche eseguite")
    ax.barh(list(y), [r["best_epoca"] for r in dati],
            color=[COLORI.get(r["modello"], "#888") for r in dati],
            label="fino al best checkpoint")
    for i, r in enumerate(dati):
        ax.plot(r["best_epoca"], i, "|", color="#111", ms=13, mew=2)
        ax.text(r["epoche"] + 0.12, i, f"best all'epoca {r['best_epoca']:.0f}",
                va="center", fontsize=7.5, color="#555")
    ax.set_yticks(list(y)); ax.set_yticklabels([r["esperimento"] for r in dati], fontsize=8)
    ax.set_xlabel("epoche")
    fermati = sum(1 for r in dati if r["early_stopped"])
    ax.set_title(f"Quanto ha girato ogni esperimento e dove stava il best\n"
                 f"{fermati}/{len(dati)} fermati dall'early stopping · "
                 f"la parte grigia è addestramento dopo il best", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=.3)
    ax.set_xlim(0, max(r["epoche"] for r in dati) * 1.32)
    return salva(fig, out, "rq2_early_stopping.png")


def g_barre(righe: list[dict], metrica: str, out: Path, nome: str, titolo: str) -> str:
    dati = [r for r in righe if valore(r, metrica) is not None]
    if not dati:
        return ""
    serie = [(m, mo) for m in MODELLI for mo in MODALITA]
    fig, ax = plt.subplots(figsize=(10, 5))
    larghezza = 0.8 / len(serie)
    for i, (m, mo) in enumerate(serie):
        y = []
        for d in DATASETS:
            sel = [
                r
                for r in dati
                if r["modello"] == m and r["modalita"] == mo and r["dataset"] == d
            ]
            y.append(valore(sel[0], metrica) if sel else 0)
        x = [j + i * larghezza - 0.4 for j in range(len(DATASETS))]
        ax.bar(
            x,
            y,
            larghezza,
            label=f"{m} {mo}",
            color=COLORI.get(m),
            alpha=1.0 if mo == "lora" else 0.55,
        )
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel(metrica)
    ax.set_title(titolo)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    return salva(fig, out, nome)


def g_delta(sintesi: dict, a: str, b: str, titolo: str, out: Path, nome: str) -> str:
    dati = [(k, va, vb) for k, va, vb in sintesi.get("coppie", [])
            if va is not None and vb is not None]
    if not dati:
        return ""
    dati.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, max(4, len(dati) * 0.36)))
    for i, (k, va, vb) in enumerate(dati):
        colore = "#3f9b5c" if vb > va else "#c0504d"
        ax.plot([va, vb], [i, i], "-", color=colore, lw=2.2, zorder=1)
        ax.plot(va, i, "o", ms=8, color="#5b6b7f", zorder=2)
        ax.plot(vb, i, "o", ms=8, color=colore, zorder=2)
        ax.text(max(va, vb) + (max(v for _, x, y in dati for v in (x, y)) * 0.012),
                i, f"{vb - va:+.4f}", va="center", fontsize=8, color=colore)
    ax.set_yticks(range(len(dati)))
    ax.set_yticklabels([k for k, _, _ in dati], fontsize=8.5)
    media = sum(vb - va for _, va, vb in dati) / len(dati)
    vinte = sum(1 for _, va, vb in dati if vb > va)
    ax.plot([], [], "o", color="#5b6b7f", label=a)
    ax.plot([], [], "o", color="#3f9b5c", label=f"{b} (meglio)")
    ax.plot([], [], "o", color="#c0504d", label=f"{b} (peggio)")
    ax.set_xlabel("metrica sul best checkpoint")
    ax.set_title(f"{titolo}\n{b} vince in {vinte}/{len(dati)} coppie · "
                 f"Δ medio {media:+.4f} · ogni riga confronta due esperimenti "
                 f"identici tranne che per questa scelta", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=.3)
    lo = min(v for _, x, y in dati for v in (x, y))
    hi = max(v for _, x, y in dati for v in (x, y))
    margine = (hi - lo) * 0.22 or 0.01
    ax.set_xlim(lo - margine, hi + margine * 1.8)
    return salva(fig, out, nome)


def g_sezioni(ok: list[dict], metrica: str, out: Path) -> str:
    fi = [r for r in ok if r["target"] == "R+I" and valore(r, metrica, "findings") is not None]
    if not fi:
        return ""
    solo_f = {(r["modello"], r["modalita"], r["viste"]): valore(r, metrica)
              for r in ok if r["target"] == "R"}
    fi.sort(key=lambda r: valore(r, metrica, "findings"))
    y = range(len(fi))
    fig, ax = plt.subplots(figsize=(10, max(4, len(fi) * 0.4)))
    ax.barh([i - 0.22 for i in y], [valore(r, metrica, "findings") for r in fi],
            0.4, color="#4c8bf5", label="findings")
    ax.barh([i + 0.22 for i in y], [valore(r, metrica, "impression") or 0 for r in fi],
            0.4, color="#f5a04c", label="impression")
    for i, r in enumerate(fi):
        media = valore(r, metrica)
        if media is not None:
            ax.plot(media, i, "D", color="#111", ms=6, zorder=3)
        rif = solo_f.get((r["modello"], r["modalita"], r["viste"]))
        if rif is not None:
            ax.plot(rif, i - 0.22, "|", color="#c0504d", ms=16, mew=2.5, zorder=3)
    ax.plot([], [], "D", color="#111", label="media (la metrica riportata)")
    ax.plot([], [], "|", color="#c0504d", ms=12, mew=2.5,
            label="findings dell'esperimento senza impression")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["esperimento"] for r in fi], fontsize=8.5)
    ax.set_xlabel(metrica)
    ax.set_title("La media riportata sta fra le due sezioni\n"
                 "se l'impression e' molto piu' alta, la media non misura i reperti",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=.3)
    return salva(fig, out, "rq9_sezioni.png")


def g_curve(righe: list[dict], out: Path) -> str:
    dati = [r for r in righe if r["curve"].get("validation")]
    if not dati:
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        sel = [r for r in dati if r["dataset"] == dataset]
        for r in sel:
            v = r["curve"]["validation"]
            ax.plot(
                [p["epoch"] for p in v],
                [p["val_loss"] for p in v],
                color=COLORI.get(r["modello"], "#888"),
                ls="-" if r["modalita"] == "lora" else "--",
                lw=1.6,
                label=f"{r['modello']} {r['modalita']}",
            )
            if r["best_epoca"] is not None:
                ax.axvline(
                    r["best_epoca"],
                    color=COLORI.get(r["modello"], "#888"),
                    alpha=0.18,
                    lw=3,
                )
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoca")
        ax.set_ylabel("val loss")
        if sel:
            ax.legend(fontsize=7)
    fig.suptitle("Loss di validation — la banda verticale è il best checkpoint")
    return salva(fig, out, "rq7_val_loss.png")


def g_curve_train(righe: list[dict], out: Path) -> str:
    dati = [r for r in righe if r["curve"].get("train")]
    if not dati:
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for r in [x for x in dati if x["dataset"] == dataset]:
            t = r["curve"]["train"]
            ax.plot(
                [p["epoch"] for p in t],
                [p["loss"] for p in t],
                color=COLORI.get(r["modello"], "#888"),
                ls="-" if r["modalita"] == "lora" else "--",
                lw=1.2,
                alpha=0.85,
                label=f"{r['modello']} {r['modalita']}",
            )
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoca")
        ax.set_ylabel("train loss")
        ax.legend(fontsize=7)
    fig.suptitle("Loss di training")
    return salva(fig, out, "rq7_train_loss.png")


def g_classifica(righe: list[dict], metrica: str, base: dict, out: Path) -> str:
    dati = sorted(
        [r for r in righe if valore(r, metrica) is not None],
        key=lambda r: valore(r, metrica),
    )
    if not dati:
        return ""
    fig, ax = plt.subplots(figsize=(9, max(4, len(dati) * 0.3)))
    nomi = [r["esperimento"] for r in dati]
    ax.barh(
        nomi,
        [valore(r, metrica) for r in dati],
        color=[COLORI.get(r["modello"], "#888") for r in dati],
    )
    for i, r in enumerate(dati):
        b = base.get((r["modello"], r["dataset"]), {}).get("media", {}).get(metrica)
        if b is not None:
            ax.plot([b], [i], "|", color="#111", ms=14, mew=2)
    ax.set_xlabel(metrica)
    ax.set_title(f"{metrica} del best checkpoint — la tacca nera è la baseline")
    ax.grid(axis="x", alpha=0.3)
    return salva(fig, out, "rq8_classifica.png")


def report(righe: list[dict], base: dict, metrica: str, out: Path) -> str:
    ok = [
        r
        for r in righe
        if r["stato"] in ("completed", "early_stopped") and r["best_valore"] is not None
    ]
    falliti = [r for r in righe if r not in ok]
    G = HA_GRAFICI

    md = [
        f"# Campagna di fine-tuning — {len(righe)} esperimenti",
        "",
        f"Metrica di confronto: **{metrica}** sul best checkpoint di validation.",
        f"Esperimenti utilizzabili: **{len(ok)}**"
        + (f", falliti o incompleti: **{len(falliti)}**" if falliti else "."),
        "",
    ]
    if falliti:
        md += [
            tabella(
                ["esperimento", "stato", "epoche"],
                [[r["esperimento"], r["stato"], num(r["epoche"], 1)] for r in falliti],
            ),
            "",
        ]

    ore = sum(r["ore"] for r in righe)
    md += [
        "## 1. Quanto tempo impiegano i 24 fine-tuning",
        "",
        f"Totale **{ore:.1f} ore di GPU**. Con due schede in parallelo il tempo "
        f"reale trascorso è circa la metà, ~{ore/2:.1f} ore.",
        "",
    ]
    md += [
        tabella(
            ["modello", "esperimenti", "ore totali", "media", "min", "max"],
            [
                [
                    m,
                    len(s),
                    f"{sum(x['ore'] for x in s):.1f}",
                    f"{statistics.fmean([x['ore'] for x in s]):.2f}",
                    f"{min(x['ore'] for x in s):.2f}",
                    f"{max(x['ore'] for x in s):.2f}",
                ]
                for m in MODELLI
                if (s := [r for r in righe if r["modello"] == m and r["ore"]])
            ],
        ),
        "",
    ]
    md += [
        tabella(
            ["esperimento", "ore", "epoche", "valutazioni", "min/epoca", "VRAM"],
            [
                [
                    r["esperimento"],
                    f"{r['ore']:.2f}",
                    num(r["epoche"], 1),
                    r["valutazioni"],
                    f"{r['ore'] * 60 / r['epoche']:.1f}" if r["epoche"] else "—",
                    num(r["vram"], 1, " GB"),
                ]
                for r in sorted(righe, key=lambda r: -r["ore"])
            ],
        ),
        "",
    ]
    if G:
        md += [g_tempi(righe, out)]

    fermati = [r for r in ok if r["early_stopped"]]
    al_limite = [r for r in ok if not r["early_stopped"]]
    ancora = [
        r
        for r in ok
        if r["best_epoca"] is not None
        and r["epoche"]
        and r["epoche"] - r["best_epoca"] < 1.5
    ]
    md += [
        "## 2. Quanti vanno in early stopping",
        "",
        f"- fermati dall'early stopping: **{len(fermati)}/{len(ok)}**",
        f"- arrivati al limite di epoche: **{len(al_limite)}/{len(ok)}**",
        f"- con il best nell'ultima epoca o quasi: **{len(ancora)}** "
        f"— stavano ancora migliorando, il budget di epoche è stato il vincolo",
        "",
    ]
    md += [
        tabella(
            ["esperimento", "stato", "epoche", "best all'epoca", "margine"],
            [
                [
                    r["esperimento"],
                    r["stato"],
                    num(r["epoche"], 1),
                    num(r["best_epoca"], 1),
                    (
                        num(r["epoche"] - r["best_epoca"], 1)
                        if r["epoche"] and r["best_epoca"] is not None
                        else "—"
                    ),
                ]
                for r in sorted(ok, key=lambda r: r["esperimento"])
            ],
        ),
        "",
    ]
    if G:
        md += [g_early(ok, out)]

    t, s = confronto_appaiato(
        ok,
        lambda r: r["modalita"],
        lambda r: (r["modello"], r["dataset"]),
        "modello · dataset",
        "lora",
        "qlora",
        metrica,
    )
    md += [
        "## 3. LoRA o QLoRA",
        "",
        f"Confronto appaiato: stessa dimensione, stesso dataset, cambia solo la "
        f"modalità. **{s['n']} coppie**, QLoRA vince in **{s['vittorie_b']}**, "
        f"Δ medio **{segno(s['media'])}**, mediano **{segno(s['mediana'])}**.",
        "",
        t,
        "",
    ]

    if G:
        md += [g_delta(s, "lora", "qlora", "LoRA contro QLoRA",
                       out, "rq3_lora_qlora.png")]
    md += ["## 4. Quanto impatta la dimensione del modello", ""]
    md += [
        tabella(
            ["modello", "esperimenti", f"{metrica} medio", "mediano", "min", "max"],
            [
                [
                    m,
                    len(v),
                    num(statistics.fmean(v)),
                    num(statistics.median(v)),
                    num(min(v)),
                    num(max(v)),
                ]
                for m in MODELLI
                if (
                    v := [
                        valore(r, metrica)
                        for r in ok
                        if r["modello"] == m and valore(r, metrica) is not None
                    ]
                )
            ],
        ),
        "",
    ]
    md += [
        tabella(
            ["dataset"] + MODELLI,
            [
                [d]
                + [
                    (
                        num(statistics.fmean(v))
                        if (
                            v := [
                                valore(r, metrica)
                                for r in ok
                                if r["modello"] == m
                                and r["dataset"] == d
                                and valore(r, metrica) is not None
                            ]
                        )
                        else "—"
                    )
                    for m in MODELLI
                ]
                for d in DATASETS
            ],
        ),
        "",
    ]
    if G:
        md += [
            g_barre(
                ok,
                metrica,
                out,
                "rq4_dimensione.png",
                f"{metrica} per dataset e configurazione",
            )
        ]

    t5, s5 = confronto_appaiato(
        ok,
        lambda r: r["viste"],
        lambda r: (r["modello"], r["modalita"], r["target"]),
        "modello · modalità · target",
        "1",
        "2",
        metrica,
    )
    md += [
        "## 5. Quanto impatta l'immagine laterale",
        "",
        "Confronto appaiato a parità di modello, modalità e target: cambia solo "
        "il numero di proiezioni. Il target è identico, quindi la metrica "
        "complessiva è confrontabile.",
        "",
        f"**{s5['n']} coppie**, due immagini vincono in **{s5['vittorie_b']}**, "
        f"Δ medio **{segno(s5['media'])}**.",
        "",
        t5,
        "",
    ]

    t6, s6 = confronto_appaiato(
        ok,
        lambda r: r["target"],
        lambda r: (r["modello"], r["modalita"], r["viste"] + " img"),
        "modello · modalità · immagini",
        "R",
        "R+I",
        metrica,
        sezione="findings",
    )
    if G:
        md += [g_delta(s5, "1 immagine", "2 immagini",
                       "Effetto della proiezione laterale",
                       out, "rq5_laterale.png")]
    md += [
        "## 6. Quanto impatta l'aggiunta dell'impression",
        "",
        "**Attenzione**: fra un target di soli reperti e uno con anche "
        "l'impressione la metrica complessiva non è confrontabile — la prima "
        "misura una sezione, la seconda la media di due. Il confronto qui sotto "
        "usa perciò la sola sezione **findings**, presente in entrambi: dice se "
        "chiedere anche l'impressione peggiora o migliora i reperti.",
        "",
        f"**{s6['n']} coppie**, aggiungere l'impressione migliora i reperti in "
        f"**{s6['vittorie_b']}**, Δ medio **{segno(s6['media'])}**.",
        "",
        t6,
        "",
    ]
    md += [
        tabella(
            ["esperimento", "findings", "impression", "media"],
            [
                [
                    r["esperimento"],
                    num(valore(r, metrica, "findings")),
                    num(valore(r, metrica, "impression")),
                    num(valore(r, metrica)),
                ]
                for r in sorted(ok, key=lambda r: r["esperimento"])
                if r["target"] == "R+I"
            ],
        ),
        "",
    ]

    if G:
        md += [g_delta(s6, "solo reperti", "reperti+impressione",
                       "Effetto dell'impressione sui soli reperti",
                       out, "rq6_impressione.png")]
    md += [
        "## 7. Andamento delle loss",
        "",
        "Un pannello per dataset; tratto continuo LoRA, tratteggiato QLoRA; "
        "la banda verticale segna il best checkpoint.",
        "",
    ]
    md += [
        tabella(
            [
                "esperimento",
                "loss train iniziale",
                "finale",
                "val loss best",
                "valutazioni",
            ],
            [
                [
                    r["esperimento"],
                    (
                        num(r["curve"]["train"][0]["loss"], 3)
                        if r["curve"].get("train")
                        else "—"
                    ),
                    (
                        num(r["curve"]["train"][-1]["loss"], 3)
                        if r["curve"].get("train")
                        else "—"
                    ),
                    num(r["val_loss"], 3),
                    r["valutazioni"],
                ]
                for r in sorted(ok, key=lambda r: r["esperimento"])
            ],
        ),
        "",
    ]
    if G:
        md += [g_curve(ok, out), g_curve_train(ok, out)]

    md += [
        f"## 8. Confronto su {metrica}",
        "",
        "Δ baseline è il guadagno rispetto al modello base senza fine-tuning, "
        "che è il criterio di accettazione.",
        "",
    ]
    dati = []
    for r in sorted(ok, key=lambda r: -(valore(r, metrica) or 0)):
        b = base.get((r["modello"], r["dataset"]), {}).get("media", {}).get(metrica)
        v = valore(r, metrica)
        dati.append(
            [
                r["esperimento"],
                r["modello"],
                r["modalita"],
                r["dataset"],
                num(v),
                num(b),
                segno(v - b) if (b and v) else "—",
                f"{v / b:.1f}×" if (b and v) else "—",
            ]
        )
    md += [
        tabella(
            [
                "esperimento",
                "modello",
                "modalità",
                "dataset",
                metrica,
                "baseline",
                "Δ baseline",
                "rapporto",
            ],
            dati,
        ),
        "",
    ]
    if G:
        md += [g_classifica(ok, metrica, base, out)]


    fi = [r for r in ok if r["target"] == "R+I"]
    solo_f = {(r["modello"], r["modalita"], r["viste"]): valore(r, metrica)
              for r in ok if r["target"] == "R"}
    dati9, scarti, contro = [], [], []
    for r in sorted(fi, key=lambda r: r["esperimento"]):
        vf = valore(r, metrica, "findings")
        vi = valore(r, metrica, "impression")
        vm = valore(r, metrica)
        rif = solo_f.get((r["modello"], r["modalita"], r["viste"]))
        if vf is not None and vi is not None:
            scarti.append(vi - vf)
        if vf is not None and rif is not None:
            contro.append(vf - rif)
        dati9.append([r["esperimento"], num(vf), num(vi), num(vm), num(rif),
                      segno(vf - rif) if (vf is not None and rif is not None) else "—"])
    md += ["## 9. La media e' gonfiata dall'impression?", "",
           "Per i dataset con entrambe le sezioni la metrica riportata e' la **media** di "
           "findings e impression. L'impression e' corta e formulaica — in due referti su "
           "tre e' una negazione gia' presente nel training — quindi e' facile da "
           "azzeccare e tira su la media. Questa tabella la scompone, e confronta i "
           "findings con quelli dell'esperimento gemello addestrato **senza** impression.",
           ""]
    if scarti:
        md += [f"L'impression supera i findings di **{segno(statistics.fmean(scarti))}** in "
               f"media ({sum(1 for s in scarti if s > 0)}/{len(scarti)} esperimenti).", ""]
    if contro:
        md += [f"I findings degli esperimenti con impression differiscono da quelli senza "
               f"di **{segno(statistics.fmean(contro))}** in media: e' l'unico confronto "
               f"che dice se chiedere anche l'impression aiuta o danneggia i reperti.", ""]
    md += [tabella(["esperimento", "findings", "impression", "media riportata",
                    "findings del gemello senza impression", "Δ"], dati9), ""]
    if G:
        md += [g_sezioni(ok, metrica, out)]

    if not G:
        md += [
            "---",
            "",
            "_Grafici non generati: manca matplotlib._",
            "`uv add matplotlib` e rilancia.",
            "",
        ]
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--metric", default="rougeL")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "training" / "results" / "rq"
    )
    args = parser.parse_args(argv)

    righe = carica(args.training_root)
    if not righe:
        print("Nessun results.json trovato.", file=sys.stderr)
        return 1
    base = baseline(args.training_root)
    args.out.mkdir(parents=True, exist_ok=True)

    testo = report(righe, base, args.metric, args.out)
    (args.out / "REPORT.md").write_text(testo, encoding="utf-8")

    campi = [
        "esperimento",
        "modello",
        "modalita",
        "dataset",
        "viste",
        "target",
        "stato",
        "early_stopped",
        "epoche",
        "best_epoca",
        "best_valore",
        "val_loss",
        "ore",
        "vram",
        "gpu",
    ]
    with (args.out / "esperimenti.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campi, extrasaction="ignore")
        w.writeheader()
        w.writerows(righe)

    png = sorted(p.name for p in args.out.glob("*.png"))
    print(f"esperimenti letti : {len(righe)}")
    print(f"baseline trovate  : {len(base)}")
    print(f"report            : {(args.out / 'REPORT.md').relative_to(ROOT)}")
    print(
        f"grafici           : {', '.join(png) if png else 'nessuno (manca matplotlib)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
