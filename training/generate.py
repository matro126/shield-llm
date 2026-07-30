#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shield.training.config import (  # noqa: E402
    BASELINE_MODE,
    TRAINING_MODES,
    Identity,
    build_config,
    read_overrides,
)

BEGIN = "# ── OVERRIDES ───────────────────────────────────────────────────────────────"
END = "# ── fine OVERRIDES ──────────────────────────────────────────────────────────"

TEMPLATE = '''#!/usr/bin/env python3
"""Fine-tuning {mode} di {base_model} su {dataset_root}.

    lingua   {lang}          (dalla cartella training/{lang}/)
    modello  {model_dir}
    modalita {mode}          ({quant})
    dataset  {dataset_dir}   (views={views}, target={target})

GENERATO da training/generate.py — non modificare il corpo a mano: le modifiche
verrebbero perse alla prossima rigenerazione. Per cambiare un iperparametro:

  * di QUESTO esperimento  → il blocco OVERRIDES qui sotto (preservato);
  * di TUTTI               → training/defaults.toml.

Eseguibile da solo:

    python {script_relpath}

Risultati (loss di training, loss e metriche di validation, best adapter) in
{relpath}/results/. La valutazione sul test set e' un processo a parte:

    python training/evaluate_test.py --experiment {experiment}
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[{depth}]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main

{begin}
# Iperparametri specifici di questo esperimento. Vuoto = usa defaults.toml.
# Esempio:  OVERRIDES = {{"learning_rate": 2e-5, "per_device_train_batch_size": 4}}
OVERRIDES: dict = {overrides}
{end}

if __name__ == "__main__":
    raise SystemExit(main(__file__, OVERRIDES))
'''


BASELINE_TEMPLATE = '''#!/usr/bin/env python3
"""Baseline ZERO-SHOT di {base_model} su {dataset_root}.

Il modello BASE, senza fine-tuning, valutato sullo stesso test set con gli stessi
prompt e le stesse metriche degli esperimenti. E\' il riferimento senza cui le
metriche del modello addestrato non sono interpretabili.

Questa baseline e\' UNA per (modello, dataset) ed e\' condivisa dagli esperimenti:

    {shared_by}

Gira in bf16, cioe\' il modello come pubblicato: cosi\' i delta di lora e qlora sono
confrontabili fra loro. Per gli esperimenti qlora il delta include quindi anche
l\'effetto della quantizzazione a 4 bit, non solo il fine-tuning.

GENERATO da training/generate.py — non modificare il corpo a mano.

    python {baseline_relpath}

Risultati in {relpath}/results/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[{depth}]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main_baseline

{begin}
# Iperparametri specifici di questa baseline. Vuoto = usa defaults.toml.
# Esempio:  OVERRIDES = {{"baseline_max_samples": 50, "load_in_4bit": True}}
OVERRIDES: dict = {overrides}
{end}

if __name__ == "__main__":
    raise SystemExit(main_baseline(__file__, OVERRIDES))
'''


def discover(training_root: Path) -> list[Identity]:
    found = []
    for dataset_dir in sorted(training_root.glob("*/*/*/*")):
        if not dataset_dir.is_dir():
            continue
        identity = Identity.from_path(dataset_dir)
        if identity.mode in TRAINING_MODES:
            found.append(identity)
    return found


def baselines_for(experiments: list[Identity]) -> list[Identity]:
    seen: dict[str, Identity] = {}
    for experiment in experiments:
        baseline = experiment.baseline
        seen.setdefault(baseline.name, baseline)
    return [seen[name] for name in sorted(seen)]


def format_overrides(overrides: dict) -> str:
    if not overrides:
        return "{}"
    lines = ["{"]
    for key, value in overrides.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    return "\n".join(lines)


def render_baseline(identity: Identity, overrides: dict) -> str:
    shared = ", ".join(
        Identity(identity.lang, identity.model_dir, mode, identity.dataset_dir).name
        for mode in TRAINING_MODES
    )
    return BASELINE_TEMPLATE.format(
        base_model=identity.base_model,
        dataset_root=identity.dataset_root,
        shared_by=shared,
        relpath=identity.relpath,
        baseline_relpath=identity.script_relpath,
        depth=len(Path(identity.relpath).parts),
        overrides=format_overrides(overrides),
        begin=BEGIN,
        end=END,
    )


def render(identity: Identity, overrides: dict) -> str:
    return TEMPLATE.format(
        lang=identity.lang,
        model_dir=identity.model_dir,
        mode=identity.mode,
        dataset_dir=identity.dataset_dir,
        dataset_root=identity.dataset_root,
        base_model=identity.base_model,
        views=identity.views,
        target=identity.target,
        experiment=identity.name,
        relpath=identity.relpath,
        script_relpath=identity.script_relpath,
        quant="base in 4-bit NF4" if identity.mode == "qlora" else "base in bf16",
        depth=len(Path(identity.relpath).parts),
        overrides=format_overrides(overrides),
        begin=BEGIN,
        end=END,
    )


def parse_set(assignments: list[str]) -> dict:
    out: dict = {}
    for item in assignments:
        if "=" not in item:
            raise SystemExit(f"--set vuole chiave=valore, ricevuto: {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                value = raw
        out[key.strip()] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--only", action="append", default=[],
                        help="glob sul nome esperimento o sul percorso (ripetibile)")
    parser.add_argument("--set", dest="assignments", action="append", default=[],
                        help="chiave=valore da scrivere negli OVERRIDES selezionati")
    parser.add_argument("--unset", action="append", default=[],
                        help="chiave da rimuovere dagli OVERRIDES selezionati")
    parser.add_argument("--list", action="store_true", help="elenca e esci")
    parser.add_argument("--check", action="store_true",
                        help="esci con 1 se qualche script non e' aggiornato")
    parser.add_argument("--force", action="store_true",
                        help="ignora gli OVERRIDES esistenti e azzerali")
    args = parser.parse_args(argv)

    experiments = discover(args.training_root)
    if not experiments:
        print(f"Nessun esperimento sotto {args.training_root}", file=sys.stderr)
        return 1

    def selected(identity: Identity) -> bool:
        if not args.only:
            return True
        return any(
            fnmatch.fnmatch(identity.name, pattern)
            or fnmatch.fnmatch(identity.relpath, pattern)
            for pattern in args.only
        )


    chosen = [e for e in experiments if selected(e)]
    chosen_baselines = {b.name: b for b in baselines_for(chosen)}
    for baseline in baselines_for(experiments):
        if selected(baseline):
            chosen_baselines.setdefault(baseline.name, baseline)
    if not chosen and not chosen_baselines:
        print("Niente corrisponde a --only", file=sys.stderr)
        return 1

    if args.list:
        root = args.training_root.parent
        print(f"{len(experiments)} esperimenti, {len(chosen)} selezionati\n")
        for identity in experiments:
            mark = "→" if identity in chosen else " "
            script = root / identity.script_relpath
            over = read_overrides(script)
            print(f" {mark} {identity.name:<26}"
                  f"{'presente' if script.is_file() else 'DA GENERARE':<14}"
                  f"{('override: ' + ', '.join(over)) if over else ''}")
        baselines = list(chosen_baselines.values())
        print(f"\n{len(baselines)} baseline (una per modello × dataset):\n")
        for identity in baselines:
            script = root / identity.script_relpath
            over = read_overrides(script)
            print(f"   {identity.name:<26}"
                  f"{'presente' if script.is_file() else 'DA GENERARE':<14}"
                  f"{('override: ' + ', '.join(over)) if over else ''}")
        return 0

    assignments = parse_set(args.assignments)
    if assignments or args.unset:
        probe = experiments[0]
        try:
            build_config(probe, ROOT, assignments)
        except ValueError as exc:
            print(f"errore: {exc}", file=sys.stderr)
            return 1

    written, unchanged, stale = 0, 0, []
    targets: list[tuple[Identity, Any]] = [(i, render) for i in chosen]
    targets += [(b, render_baseline) for b in chosen_baselines.values()]

    def receives_assignments(identity: Identity) -> bool:
        if not args.only:
            return True
        return any(fnmatch.fnmatch(identity.name, pattern) for pattern in args.only)

    touched: list[str] = []
    for identity, renderer in targets:
        folder = args.training_root.parent / identity.relpath
        script = folder / identity.script_name
        overrides = {} if args.force else read_overrides(script)
        if receives_assignments(identity):
            if assignments or args.unset:
                touched.append(identity.name)
            overrides.update(assignments)
            for key in args.unset:
                overrides.pop(key, None)

        content = renderer(identity, overrides)
        current = script.read_text(encoding="utf-8") if script.is_file() else None
        if current == content:
            unchanged += 1
            continue
        if args.check:
            stale.append(script.name)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        script.write_text(content, encoding="utf-8")
        written += 1

    if args.check:
        if stale:
            print(f"{len(stale)} script non aggiornati: {', '.join(stale)}")
            print("rigenerali con: python training/generate.py")
            return 1
        print(f"tutti aggiornati ({unchanged} script).")
        return 0

    print(f"esperimenti: {len(experiments)}   selezionati: {len(chosen)}")
    print(f"scritti: {written}   gia' aggiornati: {unchanged}")
    if assignments:
        print(f"override applicati: {assignments}")
    if args.unset:
        print(f"override rimossi: {args.unset}")
    if touched:
        print(f"  su: {', '.join(touched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
