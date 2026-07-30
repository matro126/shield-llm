#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shield.training.config import TRAINING_MODES, Identity  # noqa: E402
from shield.training.dashboard import hms  # noqa: E402


def _is_done(results_json: Path) -> bool:
    if not results_json.is_file():
        return False
    try:
        status = json.loads(results_json.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return False
    return status in ("completed", "early_stopped")


def discover(training_root: Path) -> list[Identity]:
    return [
        identity
        for p in sorted(training_root.glob("*/*/*/*"))
        if p.is_dir()
        for identity in [Identity.from_path(p)]
        if identity.mode in TRAINING_MODES
    ]


def baselines_for(experiments: list[Identity]) -> list[Identity]:
    seen: dict[str, Identity] = {}
    for experiment in experiments:
        seen.setdefault(experiment.baseline.name, experiment.baseline)
    return [seen[name] for name in sorted(seen)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--only", action="append", default=[],
                        help="glob sul nome esperimento o sul percorso (ripetibile)")
    parser.add_argument("--baselines", action="store_true",
                        help="esegue le baseline zero-shot invece dei training")
    parser.add_argument("--skip-done", action="store_true",
                        help="salta gli esperimenti gia' completati (results.json)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="prosegue con i successivi invece di fermarsi")
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra l'elenco e i comandi, senza eseguire")
    parser.add_argument("--python", default=sys.executable,
                        help="interprete da usare per i sottoprocessi")
    args = parser.parse_args(argv)

    experiments = discover(args.training_root)
    if args.only:
        experiments = [
            e for e in experiments
            if any(
                fnmatch.fnmatch(e.name, p) or fnmatch.fnmatch(e.relpath, p)
                for p in args.only
            )
        ]
    if not experiments:
        print("Nessun esperimento selezionato.", file=sys.stderr)
        return 1

    targets = baselines_for(experiments) if args.baselines else experiments

    queue: list[tuple[Identity, Path]] = []
    skipped: list[str] = []
    for identity in targets:
        script = ROOT / identity.script_relpath
        folder = script.parent
        if not script.is_file():
            skipped.append(f"{identity.name} (script assente: python training/generate.py)")
            continue
        done = (
            (folder / "results" / "metrics.json").is_file()
            if args.baselines
            else _is_done(folder / "results" / "results.json")
        )
        if args.skip_done and done:
            skipped.append(f"{identity.name} (gia' completato)")
            continue
        queue.append((identity, script))

    print(f"esperimenti in coda: {len(queue)}")
    for identity, _ in queue:
        print(f"  → {identity.name}")
    for note in skipped:
        print(f"  – saltato: {note}")

    if args.dry_run:
        print("\ncomandi:")
        for _, script in queue:
            print(f"  {args.python} {script.relative_to(ROOT)}")
        return 0
    if not queue:
        return 0

    results: list[dict] = []
    t_campaign = time.time()
    for index, (identity, script) in enumerate(queue, start=1):
        results_dir = script.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / "run.log"

        print("\n" + "=" * 78)
        print(f"[{index}/{len(queue)}] {identity.name}")
        print(f"  log: {log_path.relative_to(ROOT)}")
        print("=" * 78, flush=True)

        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [args.python, str(script)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = process.wait()

        elapsed = time.time() - started
        best = None
        canonical = (
            results_dir / "metrics.json" if args.baselines else results_dir / "results.json"
        )
        if canonical.is_file():
            try:
                data = json.loads(canonical.read_text(encoding="utf-8"))
                best = (
                    {
                        "metric": "findings.rougeL",
                        "value": (data["by_section"]["findings"] or {}).get("rougeL"),
                    }
                    if args.baselines
                    else data.get("best")
                )
            except (json.JSONDecodeError, KeyError):
                best = None
        results.append(
            {
                "experiment": identity.name,
                "returncode": code,
                "elapsed_s": round(elapsed, 1),
                "best": best,
            }
        )
        print(f"\n[{index}/{len(queue)}] {identity.name}: "
              f"{'OK' if code == 0 else f'FALLITO (exit {code})'}  in {hms(elapsed)}")

        if code != 0 and not args.continue_on_error:
            print("\nmi fermo qui (--continue-on-error per proseguire).", file=sys.stderr)
            break

    report = {
        "total_s": round(time.time() - t_campaign, 1),
        "runs": results,
    }
    report_path = args.training_root / (
        "campaign_report_baselines.json" if args.baselines else "campaign_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"CAMPAGNA COMPLETATA in {hms(report['total_s'])}")
    print(f"  {'esperimento':<28}{'esito':<12}{'tempo':>12}   best")
    for row in results:
        best = row["best"]
        value = (f"{best['metric']}={best['value']:.4f}"
                 if best and isinstance(best.get("value"), (int, float)) else "—")
        print(f"  {row['experiment']:<28}"
              f"{('OK' if row['returncode'] == 0 else 'FALLITO'):<12}"
              f"{hms(row['elapsed_s']):>12}   {value}")
    print(f"\nriepilogo: {report_path.relative_to(ROOT)}")

    failures = sum(1 for r in results if r["returncode"] != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
