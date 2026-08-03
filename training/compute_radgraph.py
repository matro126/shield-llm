#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LEVELS = ("simple", "partial", "complete")
MODEL_TYPES = ("radgraph", "radgraph-xl", "echograph", "modern-radgraph-xl")
DEFAULT_MODEL_TYPE = "radgraph"
STEP_FILE = re.compile(r"step(\d+)\.json$")
OUTPUT_NAME = "radgraph.json"
RESULTS_DIR_PARAM = "config.training.results_dir"


def resolve_results(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for option in (candidate, candidate / "results"):
        if (option / "val_predictions").is_dir():
            return option
    raise SystemExit(f"Nessuna cartella val_predictions sotto {path}")


def discover(paths: list[Path]) -> list[Path]:
    if paths:
        return [resolve_results(p) for p in paths]
    found: list[Path] = []
    for predictions in sorted((ROOT / "training" / "en").rglob("val_predictions")):
        if predictions.is_dir():
            found.append(predictions.parent)
    if not found:
        raise SystemExit(f"Nessuna run inglese con generazioni sotto {ROOT / 'training' / 'en'}")
    return found


def run_config(results: Path) -> dict[str, Any]:
    path = results / "results.json"
    if not path.is_file():
        raise SystemExit(f"results.json assente in {results}")
    return json.loads(path.read_text(encoding="utf-8")).get("config", {})


def require_english(results: Path, config: dict[str, Any]) -> None:
    lang = config.get("lang")
    if lang != "en":
        raise SystemExit(
            f"{results}: lang={lang!r}. RadGraph e' addestrato solo su referti in inglese."
        )


def step_files(results: Path, selection: str) -> list[Path]:
    files = sorted(
        (p for p in (results / "val_predictions").glob("step*.json") if STEP_FILE.search(p.name)),
        key=lambda p: int(STEP_FILE.search(p.name).group(1)),
    )
    if not files:
        raise SystemExit(f"Nessuna generazione in {results / 'val_predictions'}")
    if selection == "all":
        return files
    if selection == "last":
        return files[-1:]
    best = results / "val_predictions_best.json"
    if not best.is_file():
        raise SystemExit(f"val_predictions_best.json assente in {results}")
    step = json.loads(best.read_text(encoding="utf-8")).get("step")
    chosen = [p for p in files if int(STEP_FILE.search(p.name).group(1)) == step]
    if not chosen:
        raise SystemExit(f"Lo step migliore {step} non ha un file in {results / 'val_predictions'}")
    return chosen


def section_text(sample: dict[str, Any], key: str, target: str, lowercase: bool) -> str:
    sections = sample.get(f"{key}_sections") or {}
    text = (sections.get(target) or sample.get(key) or "").strip()
    return text.lower() if lowercase else text


def package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("radgraph")
    except PackageNotFoundError:
        return "sconosciuta"


def build_scorer(model_type: str) -> Any:
    try:
        from radgraph import F1RadGraph
    except ImportError as error:
        raise SystemExit(
            "Il pacchetto radgraph non e' installato: uv sync (oppure uv add radgraph)"
        ) from error
    return F1RadGraph(reward_level="all", model_type=model_type)


def as_levels(mean_reward: Any) -> dict[str, float]:
    values = list(mean_reward) if isinstance(mean_reward, (list, tuple)) else [mean_reward]
    if len(values) != len(LEVELS):
        raise SystemExit(
            f"F1RadGraph(reward_level='all') ha restituito {len(values)} valori invece di "
            f"{len(LEVELS)}: la versione installata non corrisponde a quella attesa."
        )
    return {level: float(value) for level, value in zip(LEVELS, values)}


def score_step(
    scorer: Any, payload: dict[str, Any], target: str, lowercase: bool
) -> dict[str, Any]:
    samples = payload["samples"]
    hyps = [section_text(s, "prediction", target, lowercase) for s in samples]
    refs = [section_text(s, "reference", target, lowercase) for s in samples]
    mean_reward = scorer(hyps=hyps, refs=refs)[0]
    entry = {
        "step": payload["step"],
        "epoch": payload["epoch"],
        "n_examples": len(samples),
    }
    entry.update({f"radgraph_{k}": v for k, v in as_levels(mean_reward).items()})
    return entry


def mlflow_client(tracking_uri: str | None) -> tuple[Any, str]:
    from mlflow.tracking import MlflowClient

    from shield.tracking import DEFAULT_TRACKING_URI

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    return MlflowClient(tracking_uri=uri), uri


def find_run(client: Any, results_dir: str) -> list[tuple[str, str]]:
    from datetime import datetime

    ids = [e.experiment_id for e in client.search_experiments()]
    if not ids:
        return []
    runs = client.search_runs(
        experiment_ids=ids,
        filter_string=f'params.`{RESULTS_DIR_PARAM}` = "{results_dir}"',
        order_by=["attributes.start_time DESC"],
    )
    return [
        (r.info.run_id, datetime.fromtimestamp(r.info.start_time / 1000).isoformat(" ", "minutes"))
        for r in runs
    ]


def push(client: Any, run_id: str, entries: list[dict[str, Any]], target: str) -> int:
    written = 0
    for entry in entries:
        for level in LEVELS:
            client.log_metric(
                run_id,
                f"val.{target}.radgraph_{level}",
                entry[f"radgraph_{level}"],
                step=int(entry["step"]),
            )
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calcola RadGraph F1 a posteriori sulle generazioni delle run inglesi."
    )
    parser.add_argument("paths", type=Path, nargs="*", help="cartelle esperimento o results")
    parser.add_argument("--steps", choices=("all", "last", "best"), default="all")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=DEFAULT_MODEL_TYPE)
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="disattiva la normalizzazione a minuscolo di riferimenti e predizioni",
    )
    parser.add_argument("--mlflow", action="store_true", help="scrive le metriche su MLflow")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--run-id", default=None, help="run MLflow esplicita, con un solo path")
    parser.add_argument("--output", default=OUTPUT_NAME)
    args = parser.parse_args(argv)

    targets = discover(args.paths)
    if args.run_id and len(targets) != 1:
        raise SystemExit("--run-id richiede esattamente un path")

    configs = []
    for results in targets:
        config = run_config(results)
        require_english(results, config)
        configs.append(config)

    client = uri = None
    if args.mlflow:
        client, uri = mlflow_client(args.tracking_uri)
        print(f"mlflow      : {uri}")

    provenance = {
        "package": "radgraph",
        "package_version": package_version(),
        "model_type": args.model_type,
        "reward_level": "all",
        "lowercase": args.lowercase,
    }
    print(f"radgraph    : {provenance['package_version']}  model_type={args.model_type}")
    print(f"lowercase   : {args.lowercase}\n")

    scorer = build_scorer(args.model_type)
    failures = 0

    for results, config in zip(targets, configs):
        target = config.get("target", "findings")
        files = step_files(results, args.steps)
        print(f"esperimento : {config.get('experiment')}")
        print(f"risultati   : {results.relative_to(ROOT)}")
        print(f"sezione     : {target}   valutazioni: {len(files)}\n")
        print(f"{'ep':>3}{'step':>7}{'simple':>10}{'partial':>10}{'complete':>10}")

        entries = []
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry = score_step(scorer, payload, target, args.lowercase)
            entries.append(entry)
            print(
                f"{int(entry['epoch']):>3}{entry['step']:>7}"
                f"{entry['radgraph_simple']:>10.4f}"
                f"{entry['radgraph_partial']:>10.4f}"
                f"{entry['radgraph_complete']:>10.4f}"
            )

        destination = results / args.output
        destination.write_text(
            json.dumps(
                {
                    "experiment": config.get("experiment"),
                    "target": target,
                    "split": "val",
                    "provenance": provenance,
                    "entries": entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nscritto     : {destination.relative_to(ROOT)}")

        if args.mlflow:
            run_id = args.run_id
            if not run_id:
                candidates = find_run(client, config.get("results_dir", ""))
                if len(candidates) != 1:
                    print(
                        f"NON loggato : {len(candidates)} run MLflow con "
                        f"{RESULTS_DIR_PARAM}={config.get('results_dir')!r}"
                    )
                    for candidate, started in candidates:
                        print(f"              {candidate}  avviata {started}")
                    failures += 1
                    print()
                    continue
                run_id = candidates[0][0]
            written = push(client, run_id, entries, target)
            print(f"mlflow      : {written} metriche su {run_id}")
        print()

    if failures:
        print(
            f"{failures} run non loggate su MLflow: rilanciale una alla volta con --run-id"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
