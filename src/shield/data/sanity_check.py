from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .build import SPLITS, VERSIONS, Version, version_relpath
from .prompts import get_prompts

MAX_REPORTED = 5


def _load_split(version_dir: Path, split: str) -> list[dict[str, Any]]:
    path = version_dir / f"{split}.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_files_present(version_dir: Path, *_: Any) -> list[str]:
    expected = [f"{split}.jsonl" for split in SPLITS] + ["manifest.yaml", "stats.json"]
    return [f"file mancante: {name}" for name in expected if not (version_dir / name).is_file()]


def check_ids_unique(_: Path, records: Mapping[str, list[dict]], __: Version) -> list[str]:
    seen: dict[str, str] = {}
    problems: list[str] = []
    for split, rows in records.items():
        for row in rows:
            sample_id = row["id"]
            if sample_id in seen:
                problems.append(
                    f"id duplicato {sample_id!r}: in {seen[sample_id]} e {split}"
                )
            seen[sample_id] = split
    return problems[:MAX_REPORTED]


def check_no_split_leakage(
    _: Path, records: Mapping[str, list[dict]], __: Version
) -> list[str]:
    ids = {split: {row["id"] for row in rows} for split, rows in records.items()}
    problems: list[str] = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = ids.get(a, set()) & ids.get(b, set())
        if shared:
            problems.append(
                f"{len(shared)} id condivisi tra {a} e {b}: {sorted(shared)[:3]}"
            )
    return problems


def check_image_count(
    _: Path, records: Mapping[str, list[dict]], version: Version
) -> list[str]:
    expected = 2 if version.views == "frontal_lateral" else 1
    problems: list[str] = []
    for split, rows in records.items():
        wrong = [r["id"] for r in rows if len(r["images"]) != expected]
        if wrong:
            problems.append(
                f"{split}: {len(wrong)} record con != {expected} immagini "
                f"({wrong[:3]})"
            )
    return problems


def check_images_exist(
    version_dir: Path, records: Mapping[str, list[dict]], _: Version
) -> list[str]:
    problems: list[str] = []
    for split, rows in records.items():
        missing = [
            f"{r['id']}:{rel}"
            for r in rows
            for rel in r["images"]
            if not (version_dir / rel).is_file()
        ]
        if missing:
            problems.append(
                f"{split}: {len(missing)} file immagine mancanti ({missing[:3]})"
            )
    return problems


def check_message_shape(
    _: Path, records: Mapping[str, list[dict]], __: Version
) -> list[str]:
    problems: list[str] = []
    for split, rows in records.items():
        for row in rows[:]:
            roles = [m["role"] for m in row["messages"]]
            if roles != ["system", "user", "assistant"]:
                problems.append(f"{split}/{row['id']}: ruoli inattesi {roles}")
                break
            n_images = sum(
                1 for c in row["messages"][1]["content"] if c["type"] == "image"
            )
            if n_images != len(row["images"]):
                problems.append(
                    f"{split}/{row['id']}: {n_images} immagini nel prompt "
                    f"ma {len(row['images'])} dichiarate"
                )
                break
    return problems


def check_target_not_empty(
    _: Path, records: Mapping[str, list[dict]], __: Version
) -> list[str]:
    problems: list[str] = []
    for split, rows in records.items():
        empty = [r["id"] for r in rows if not r["messages"][2]["content"].strip()]
        if empty:
            problems.append(f"{split}: {len(empty)} target vuoti ({empty[:3]})")
    return problems


def check_target_sections(
    _: Path, records: Mapping[str, list[dict]], version: Version
) -> list[str]:
    from .prompts import SEP, split_sections

    if version.target != "findings_impression":
        return []
    problems: list[str] = []
    for split, rows in records.items():
        malformed: list[str] = []
        for row in rows:
            text = row["messages"][2]["content"]
            findings, impression = split_sections(text)
            if SEP not in text or not findings.strip() or not (impression or "").strip():
                malformed.append(row["id"])
        if malformed:
            problems.append(
                f"{split}: {len(malformed)} target senza <SEP> o con una sezione vuota "
                f"({malformed[:3]})"
            )
    return problems


def check_prompts_match_scenario(
    _: Path, records: Mapping[str, list[dict]], version: Version
) -> list[str]:
    expected = get_prompts(version.views, version.target, version.lang)
    problems: list[str] = []
    for split, rows in records.items():
        for row in rows[:1]:
            if row["messages"][0]["content"] != expected.system:
                problems.append(f"{split}: system prompt diverso da quello dello scenario")
            user_text = next(
                c["text"] for c in row["messages"][1]["content"] if c["type"] == "text"
            )
            if user_text != expected.user:
                problems.append(f"{split}: user prompt diverso da quello dello scenario")
    return problems


def check_projections(
    _: Path, records: Mapping[str, list[dict]], version: Version
) -> list[str]:
    expected = (
        {"frontal", "lateral"} if version.views == "frontal_lateral" else {"frontal"}
    )
    problems: list[str] = []
    for split, rows in records.items():
        wrong = [
            r["id"]
            for r in rows
            if sorted(r["factors"]["views"]) != sorted(expected)
        ]
        if wrong:
            problems.append(
                f"{split}: {len(wrong)} record con proiezioni non valide ({wrong[:3]})"
            )
    return problems


CHECKS = {
    "file attesi presenti": check_files_present,
    "id univoci": check_ids_unique,
    "nessuna sovrapposizione tra split": check_no_split_leakage,
    "numero di immagini coerente con views": check_image_count,
    "file immagine esistenti": check_images_exist,
    "struttura dei messaggi": check_message_shape,
    "target non vuoto": check_target_not_empty,
    "sezioni del target": check_target_sections,
    "prompt coerenti con lo scenario": check_prompts_match_scenario,
    "proiezioni dichiarate": check_projections,
}


def check_manifest_integrity(
    version_dir: Path, version: Version, project_root: Path
) -> list[str]:
    import yaml

    from .build import integrity_section

    manifest_path = version_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return ["manifest.yaml assente: impossibile verificare l'integrita'"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("integrity")
    if not recorded:
        return [
            "manifest senza sezione 'integrity': ricostruisci la versione per ottenerla"
        ]

    current = integrity_section(
        version,
        version_dir,
        project_root,
        project_root / "dataset/iu-xray/iu_xray_r2gen_final/annotation.json",
        project_root / "dataset/iu-xray/iu_xray_translated.csv",
        project_root / "dataset/iu-xray/iu_xray_r2gen_final/images",
    )

    problems: list[str] = []
    for group, entries in current.items():
        for key, value in entries.items():
            expected = (recorded.get(group) or {}).get(key)
            if expected is None:
                problems.append(f"{group}.{key}: assente nel manifest")
            elif expected != value:
                problems.append(
                    f"{group}.{key}: manifest {expected[7:19]}… vs attuale {value[7:19]}…"
                )
    return problems


def _import_closure(entry: str, src_root: Path) -> set[str]:
    import ast

    def module_file(module: str) -> Path | None:
        direct = src_root / (module.replace(".", "/") + ".py")
        package = src_root / module.replace(".", "/") / "__init__.py"
        return direct if direct.is_file() else (package if package.is_file() else None)

    seen: set[str] = set()
    stack = [entry]
    while stack:
        module = stack.pop()
        path = module_file(module)
        if module in seen or path is None:
            continue
        seen.add(module)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = module.split(".")
                    base = (
                        parts[: len(parts) - node.level + 1]
                        if path.name == "__init__.py"
                        else parts[: -node.level]
                    )
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if target.startswith("shield"):
                    stack.append(target)
            elif isinstance(node, ast.Import):
                stack.extend(a.name for a in node.names if a.name.startswith("shield"))

    return {
        str(module_file(m).relative_to(src_root.parent))
        for m in seen
        if module_file(m)
    }


def check_dependency_declarations(project_root: Path) -> list[str]:
    import yaml

    from .build import CODE_FILES

    problems: list[str] = []
    actual = _import_closure("shield.data.build", project_root / "src")

    missing = actual - set(CODE_FILES)
    extra = set(CODE_FILES) - actual
    problems += [f"CODE_FILES non dichiara {name}" for name in sorted(missing)]
    problems += [f"CODE_FILES dichiara {name}, non piu' importato" for name in sorted(extra)]

    dvc_path = project_root / "dvc.yaml"
    if not dvc_path.is_file():
        return problems
    stages = (yaml.safe_load(dvc_path.read_text(encoding="utf-8")) or {}).get(
        "stages", {}
    )
    entrypoints = {
        "sanity_check": "shield.data.sanity_check",
    }
    for name, stage in stages.items():
        entry = entrypoints.get(name, "shield.data.build")
        declared = {d for d in stage.get("deps", []) if str(d).startswith("src/")}
        expected = _import_closure(entry, project_root / "src")
        for name_missing in sorted(expected - declared):
            problems.append(f"dvc.yaml [{name}] non dichiara {name_missing}")
        for name_stale in sorted(declared - expected):
            problems.append(f"dvc.yaml [{name}] dichiara {name_stale}, non importato")
    return problems


def check_cohort_consistency(datasets: Mapping[str, Mapping[str, list[dict]]]) -> list[str]:
    problems: list[str] = []
    names = sorted(datasets)
    if len(names) < 2:
        return problems
    reference = datasets[names[0]]
    for name in names[1:]:
        for split in SPLITS:
            a = {r["id"] for r in reference.get(split, [])}
            b = {r["id"] for r in datasets[name].get(split, [])}
            if a != b:
                problems.append(
                    f"{split}: {name} differisce da {names[0]} "
                    f"(+{len(b - a)} / -{len(a - b)})"
                )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/iu-xray"))
    parser.add_argument("--version", action="append", choices=sorted(VERSIONS))
    args = parser.parse_args(argv)

    root = args.root.resolve()
    wanted = args.version or sorted(VERSIONS)
    failures = 0
    datasets: dict[str, dict[str, list[dict]]] = {}

    for name in wanted:
        version = VERSIONS[name]
        version_dir = root / args.dataset_dir / version_relpath(version)
        print("=" * 66)
        if not version_dir.is_dir():
            print(f"{name}  — NON COSTRUITO (salto)")
            continue

        records = {split: _load_split(version_dir, split) for split in SPLITS}
        datasets[name] = records
        total = sum(len(rows) for rows in records.values())
        counts = "  ".join(f"{s}={len(records[s])}" for s in SPLITS)
        print(f"{name}  views={version.views}  target={version.target}")
        print(f"  {total} record   {counts}\n")

        for label, check in CHECKS.items():
            problems = check(version_dir, records, version)
            if problems:
                failures += 1
                print(f"  ✗ {label}")
                for problem in problems[:MAX_REPORTED]:
                    print(f"      {problem}")
            else:
                print(f"  ✓ {label}")

        problems = check_manifest_integrity(version_dir, version, root)
        if problems:
            failures += 1
            print("  ✗ integrita': hash del manifest coerenti")
            for problem in problems[:MAX_REPORTED]:
                print(f"      {problem}")
        else:
            print("  ✓ integrita': hash del manifest coerenti")

    print("=" * 66)
    print("DICHIARAZIONI DI DIPENDENZA\n")
    problems = check_dependency_declarations(root)
    if problems:
        failures += 1
        print("  ✗ CODE_FILES e dvc.yaml coprono gli import reali")
        for problem in problems[:MAX_REPORTED]:
            print(f"      {problem}")
    else:
        print("  ✓ CODE_FILES e dvc.yaml coprono gli import reali")

    if len(datasets) > 1:
        print("=" * 66)
        print("COERENZA TRA VERSIONI\n")
        problems = check_cohort_consistency(datasets)
        if problems:
            failures += 1
            print("  ✗ stessa coorte in tutte le versioni")
            for problem in problems[:MAX_REPORTED]:
                print(f"      {problem}")
        else:
            print("  ✓ stessa coorte in tutte le versioni")

    print("=" * 66)
    print("ESITO: TUTTO OK" if not failures else f"ESITO: {failures} CONTROLLI FALLITI")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
