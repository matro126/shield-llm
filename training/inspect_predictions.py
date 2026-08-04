#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HEALTHY = "No Finding"
IGNORED = {HEALTHY, "Other", "Unlabeled", "Support Devices"}
DEFAULT_PROBE = ROOT / "training" / "probe_set.json"
MAX_PROBE_SAMPLES = 30
STEP_FILE = re.compile(r"step(\d+)\.json$")


def resolve_results(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.is_dir() and candidate.name == "val_predictions":
        return candidate.parent
    for option in (candidate, candidate / "results"):
        if (option / "val_predictions").is_dir():
            return option
    raise SystemExit(f"Nessuna cartella val_predictions sotto {path}")


def mlflow_provenance(results: Path) -> dict[str, str | None]:
    path = results / "results.json"
    if not path.is_file():
        return {"run_id": None, "tracking_uri": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mlflow = payload.get("provenance", {}).get("mlflow") or {}
    return {
        "run_id": mlflow.get("run_id"),
        "tracking_uri": mlflow.get("tracking_uri"),
    }


def load_steps(results: Path) -> list[dict[str, Any]]:
    files = sorted(
        (p for p in (results / "val_predictions").glob("step*.json") if STEP_FILE.search(p.name)),
        key=lambda p: int(STEP_FILE.search(p.name).group(1)),
    )
    if not files:
        raise SystemExit(f"Nessuna generazione in {results / 'val_predictions'}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def best_step(results: Path) -> int | None:
    path = results / "val_predictions_best.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("step")


def categories(sample: dict[str, Any]) -> list[str]:
    return list(sample.get("factors", {}).get("diagnostic_category", []))


def is_healthy(sample: dict[str, Any]) -> bool:
    return categories(sample) == [HEALTHY]


def pathologies(sample: dict[str, Any]) -> list[str]:
    return [c for c in categories(sample) if c not in IGNORED]


def _diverse_sample(
    candidates: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0 or not candidates:
        return []
    remaining = list(candidates)
    frequencies = Counter(
        label for sample in candidates for label in pathologies(sample)
    )
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    tie_break = {sample["id"]: rng.random() for sample in remaining}
    while remaining and len(selected) < count:
        def score(sample: dict[str, Any]) -> tuple[float, float]:
            labels = pathologies(sample)
            diversity = sum(
                1.0 / ((selected_counts[label] + 1) * frequencies[label] ** 0.5)
                for label in labels
            )
            return diversity, tie_break[sample["id"]]

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        selected_counts.update(pathologies(chosen))
    return selected


def select_probe(
    samples: list[dict[str, Any]],
    n_healthy: int,
    n_sick: int,
    seed: int,
) -> dict[str, Any]:
    requested = n_healthy + n_sick
    if n_healthy < 0 or n_sick < 0:
        raise ValueError("Il numero di casi non può essere negativo.")
    if requested == 0:
        raise ValueError("Il probe set deve contenere almeno un caso.")
    if requested > MAX_PROBE_SAMPLES:
        raise ValueError(
            f"Il probe set può contenere al massimo {MAX_PROBE_SAMPLES} casi "
            f"(richiesti: {requested})."
        )

    rng = random.Random(seed)
    healthy = sorted((s for s in samples if is_healthy(s)), key=lambda s: s["id"])
    sick = sorted((s for s in samples if pathologies(s)), key=lambda s: s["id"])
    chosen_healthy = rng.sample(healthy, min(n_healthy, len(healthy)))

    single = [s for s in sick if len(pathologies(s)) == 1]
    multi = [s for s in sick if len(pathologies(s)) > 1]
    multi_target = min(len(multi), round(n_sick * 0.4))
    single_target = min(len(single), n_sick - multi_target)
    if single_target + multi_target < n_sick:
        missing = n_sick - single_target - multi_target
        single_target += min(missing, len(single) - single_target)
        missing = n_sick - single_target - multi_target
        multi_target += min(missing, len(multi) - multi_target)

    chosen_sick = _diverse_sample(single, single_target, rng)
    chosen_sick += _diverse_sample(multi, multi_target, rng)
    labels = {s["id"]: pathologies(s) for s in samples}
    chosen_healthy_ids = sorted(s["id"] for s in chosen_healthy)
    chosen_sick_ids = sorted(s["id"] for s in chosen_sick)
    covered = sorted({label for s in chosen_sick for label in pathologies(s)})
    return {
        "seed": seed,
        "healthy": chosen_healthy_ids,
        "sick": chosen_sick_ids,
        "labels": {
            uid: labels.get(uid, [])
            for uid in chosen_healthy_ids + chosen_sick_ids
        },
        "selection": {
            "strategy": "diagnosis_diversity_with_single_multi_balance",
            "requested": requested,
            "selected": len(chosen_healthy_ids) + len(chosen_sick_ids),
            "healthy": len(chosen_healthy_ids),
            "pathological_single": sum(
                len(labels[uid]) == 1 for uid in chosen_sick_ids
            ),
            "pathological_multi": sum(
                len(labels[uid]) > 1 for uid in chosen_sick_ids
            ),
            "covered_pathologies": covered,
        },
    }


def build_probe(results: Path, n_healthy: int, n_sick: int, seed: int) -> dict[str, Any]:
    samples = load_steps(results)[-1]["samples"]
    probe = select_probe(samples, n_healthy, n_sick, seed)
    probe["built_from"] = str(results.relative_to(ROOT))
    return probe


TOKEN = re.compile(r"[^a-z0-9]+")
_BACKEND: list[Any] = []


def _tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.sub(" ", text.lower()).split() if t]


def _lcs(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(
                previous[j] + 1 if token == other else max(current[j], previous[j + 1])
            )
        previous = current
    return previous[-1]


def _rouge_l_fallback(prediction: str, reference: str) -> float:
    pred, ref = _tokenize(prediction), _tokenize(reference)
    if not pred or not ref:
        return 0.0
    common = _lcs(pred, ref)
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _resolve_backend() -> Any:
    if _BACKEND:
        return _BACKEND[0]
    try:
        from shield.evaluation.metrics import rouge_scores

        rouge_scores(["a"], ["a"])
        _BACKEND.append(lambda p, r: rouge_scores([p], [r])["rougeL"])
    except Exception:
        print("[nota] rouge-score assente: uso l'implementazione LCS interna\n")
        _BACKEND.append(_rouge_l_fallback)
    return _BACKEND[0]


def rouge_l(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    return _resolve_backend()(prediction, reference)


def sample_text(
    sample: dict[str, Any],
    key: str,
    *,
    target: str,
    section: str,
) -> str:
    sections = sample.get(f"{key}_sections") or {}
    if section in {"findings", "impression"}:
        return (sections.get(section) or "").strip()
    if target != "findings_impression":
        return (sections.get("findings") or sample.get(key) or "").strip()
    findings = (sections.get("findings") or "").strip()
    impression = (sections.get("impression") or "").strip()
    if findings or impression:
        return f"Findings:\n{findings}\n\nImpression:\n{impression}".strip()
    return str(sample.get(key) or "").strip()


def section_text(sample: dict[str, Any], key: str) -> str:
    return sample_text(sample, key, target="findings", section="findings")


def wrap(text: str, width: int, limit: int | None) -> list[str]:
    flat = " ".join(text.split())
    if limit and len(flat) > limit:
        flat = flat[:limit].rstrip() + " […]"
    return textwrap.wrap(flat, width) or [""]


def render_case(
    uid: str,
    label: str,
    steps: list[dict[str, Any]],
    best: int | None,
    width: int,
    limit: int | None,
    section: str,
) -> list[float]:
    found = [(d, next((s for s in d["samples"] if s["id"] == uid), None)) for d in steps]
    present = [(d, s) for d, s in found if s is not None]
    if not present:
        print(f"\n{uid}  — assente da queste generazioni")
        return []

    target = present[0][0].get("target", "findings")
    reference = sample_text(
        present[0][1], "reference", target=target, section=section
    )
    print(f"\n{'─' * (width + 22)}")
    print(f"{uid}   [{label}]")
    for line in wrap(reference, width, limit):
        print(f"  rif   │ {line}")

    scores = []
    for payload, sample in present:
        prediction = sample_text(
            sample,
            "prediction",
            target=payload.get("target", target),
            section=section,
        )
        score = rouge_l(prediction, reference)
        scores.append(score)
        mark = " ★" if best is not None and payload["step"] == best else "  "
        head = f"  ep{int(payload['epoch']):>3}{mark}│"
        lines = wrap(prediction, width, limit)
        print(f"{head} {lines[0]}   ({score:.3f})")
        for line in lines[1:]:
            print(f"        │ {line}")
    return scores


def summary(
    steps: list[dict[str, Any]],
    probe: dict[str, Any],
    best: int | None,
    section: str,
) -> None:
    healthy, sick = set(probe["healthy"]), set(probe["sick"])
    print(f"\n{'═' * 78}")
    print("EVOLUZIONE SUL PROBE SET\n")
    print(f"{'ep':>3} {'step':>6} {'sani':>8} {'malati':>8} {'delta':>8} {'val.rougeL':>11}")
    rows = []
    for payload in steps:
        by_id = {s["id"]: s for s in payload["samples"]}
        target = payload.get("target", "findings")

        def mean(ids: set[str]) -> float:
            values = [
                rouge_l(
                    sample_text(
                        by_id[i], "prediction", target=target, section=section
                    ),
                    sample_text(
                        by_id[i], "reference", target=target, section=section
                    ),
                )
                for i in ids
                if i in by_id
            ]
            return sum(values) / len(values) if values else 0.0

        h, s = mean(healthy), mean(sick)
        if section in {"findings", "impression"}:
            overall_metrics = payload.get("sections", {}).get(section) or {}
        else:
            overall_metrics = payload.get("metrics", {})
        overall = overall_metrics.get("rougeL", 0.0)
        mark = " ★" if best is not None and payload["step"] == best else "  "
        print(f"{int(payload['epoch']):>3}{mark}{payload['step']:>6} {h:>8.3f} {s:>8.3f} {h - s:>8.3f} {overall:>11.3f}")
        rows.append((payload["epoch"], payload["step"], h, s, overall))

    if not rows:
        return
    top_sick = max(rows, key=lambda r: r[3])
    top_overall = max(rows, key=lambda r: r[4])
    print()
    print(f"miglior punteggio sui casi malati : epoca {int(top_sick[0])} (step {top_sick[1]}, {top_sick[3]:.3f})")
    print(f"miglior rougeL complessivo        : epoca {int(top_overall[0])} (step {top_overall[1]}, {top_overall[4]:.3f})")
    if best is not None:
        chosen = next((r for r in rows if r[1] == best), None)
        if chosen:
            print(f"best_adapter selezionato          : epoca {int(chosen[0])} (step {best}, malati {chosen[3]:.3f})")
            if chosen[1] != top_sick[1]:
                print("\nATTENZIONE: il best_adapter non coincide con l'epoca migliore sui casi malati.")


REVIEW_FIELDS = (
    "review_id",
    "experiment",
    "section",
    "epoch",
    "step",
    "case_id",
    "group",
    "diagnostic_categories",
    "reference",
    "prediction",
    "false_positive_finding",
    "omitted_finding",
    "incorrect_location",
    "incorrect_severity",
    "spurious_comparison",
    "omitted_comparison",
    "major_errors",
    "minor_errors",
    "notes",
)


def export_review_csv(
    steps: list[dict[str, Any]],
    probe: dict[str, Any],
    destination: Path,
    *,
    section: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    healthy = list(probe.get("healthy", []))
    sick = list(probe.get("sick", []))
    ids = [(uid, "healthy") for uid in healthy]
    ids += [(uid, "pathological") for uid in sick]
    review_id = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for uid, group in ids:
            for payload in steps:
                by_id = {sample["id"]: sample for sample in payload["samples"]}
                target = payload.get("target", "findings")
                sample = by_id.get(uid)
                if sample is None:
                    continue
                review_id += 1
                writer.writerow(
                    {
                        "review_id": f"R{review_id:04d}",
                        "experiment": payload.get("experiment", ""),
                        "section": section,
                        "epoch": payload.get("epoch", ""),
                        "step": payload.get("step", ""),
                        "case_id": uid,
                        "group": group,
                        "diagnostic_categories": " | ".join(
                            probe.get("labels", {}).get(uid, [])
                        ),
                        "reference": sample_text(
                            sample, "reference", target=target, section=section
                        ),
                        "prediction": sample_text(
                            sample, "prediction", target=target, section=section
                        ),
                        "false_positive_finding": "",
                        "omitted_finding": "",
                        "incorrect_location": "",
                        "incorrect_severity": "",
                        "spurious_comparison": "",
                        "omitted_comparison": "",
                        "major_errors": "",
                        "minor_errors": "",
                        "notes": "",
                    }
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confronta le generazioni di un esperimento su un probe set fisso."
    )
    parser.add_argument("path", type=Path, help="cartella esperimento, results, o script .py")
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--build-probe", action="store_true", help="crea il probe set e esci")
    parser.add_argument("--healthy", type=int, default=6)
    parser.add_argument("--sick", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--chars", type=int, default=320, help="0 = testo integrale")
    parser.add_argument("--only-summary", action="store_true")
    parser.add_argument(
        "--section",
        choices=("target", "findings", "impression"),
        default="target",
        help="parte del referto da confrontare (default: target dell'esperimento)",
    )
    parser.add_argument(
        "--export-review",
        type=Path,
        metavar="CSV",
        help="esporta testi integrali e colonne vuote per la revisione manuale",
    )
    args = parser.parse_args(argv)

    results = resolve_results(args.path)
    provenance = mlflow_provenance(results)
    print(f"mlflow run  : {provenance['run_id'] or 'non disponibile'}")
    if provenance["tracking_uri"]:
        print(f"tracking URI: {provenance['tracking_uri']}")
    _resolve_backend()

    if args.build_probe:
        try:
            probe = build_probe(results, args.healthy, args.sick, args.seed)
        except ValueError as exc:
            parser.error(str(exc))
        args.probe.parent.mkdir(parents=True, exist_ok=True)
        args.probe.write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"probe set scritto in {args.probe}")
        print(f"  sani   : {len(probe['healthy'])}  {', '.join(probe['healthy'])}")
        print(f"  malati : {len(probe['sick'])}  {', '.join(probe['sick'])}")
        selection = probe.get("selection", {})
        print(
            f"  mix     : {selection.get('pathological_single', 0)} mono-patologia + "
            f"{selection.get('pathological_multi', 0)} multi-patologia"
        )
        print(
            "  diagnosi: "
            + ", ".join(selection.get("covered_pathologies", []))
        )
        return 0

    if not args.probe.is_file():
        raise SystemExit(
            f"Probe set assente: {args.probe}\n"
            f"Crealo con: python training/inspect_predictions.py {args.path} --build-probe"
        )

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    steps = load_steps(results)
    best = best_step(results)

    print(f"esperimento : {steps[-1].get('experiment')}")
    print(f"risultati   : {results.relative_to(ROOT)}")
    print(f"valutazioni : {len(steps)}  (step {steps[0]['step']} → {steps[-1]['step']})")
    print(f"probe set   : {args.probe.name}  {len(probe['healthy'])} sani + {len(probe['sick'])} malati")
    print(f"sezione     : {args.section}")

    if args.export_review:
        export_review_csv(steps, probe, args.export_review, section=args.section)
        print(f"review CSV  : {args.export_review}")

    if not args.only_summary:
        limit = args.chars or None
        for uid in probe["healthy"]:
            render_case(uid, "SANO", steps, best, args.width, limit, args.section)
        for uid in probe["sick"]:
            labels = ", ".join(probe.get("labels", {}).get(uid, [])) or "patologico"
            render_case(
                uid, labels.upper(), steps, best, args.width, limit, args.section
            )

    summary(steps, probe, best, args.section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
