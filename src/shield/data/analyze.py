from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .images import view_symmetry
from .openi import OpenIStudy, parse_report, report_paths

DEFAULT_THRESHOLD = 0.5
PLACEHOLDER = "XXXX"

_FRONTAL_RE = re.compile(
    r"\b(pa|ap|p\.a\.|a\.p\.|frontal|postero-?anterior|antero-?posterior)\b", re.I
)
_LATERAL_RE = re.compile(r"\b(lateral|lat|decubitus|oblique)\b", re.I)
_TWO_VIEWS_RE = re.compile(r"\b(2|two|ii)\s*-?\s*(v|view|views)\b", re.I)


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_placeholder_only(text: str) -> bool:
    return not any(char.isalnum() for char in text.replace(PLACEHOLDER, " "))


def caption_views(caption: str) -> tuple[bool | None, bool | None]:
    if not caption:
        return None, None
    frontal = bool(_FRONTAL_RE.search(caption))
    lateral = bool(_LATERAL_RE.search(caption))
    if _TWO_VIEWS_RE.search(caption):
        return True, True
    if not frontal and not lateral:
        return None, None
    return frontal, lateral


class SymmetryCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._values: dict[str, float] = {}
        if path and path.is_file():
            try:
                self._values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._values = {}

    def score(self, image: Path, key: str) -> float:
        if key not in self._values:
            self._values[key] = view_symmetry(image)
        return self._values[key]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._values, sort_keys=True), encoding="utf-8"
            )


def symmetry_views(
    images: Sequence[tuple[str, Path]],
    cache: SymmetryCache,
    threshold: float,
) -> tuple[bool, bool, int, dict[str, float]]:
    scores: dict[str, float] = {}
    for key, path in images:
        if path.is_file():
            scores[key] = cache.score(path, key)
    frontal = any(value >= threshold for value in scores.values())
    lateral = any(value < threshold for value in scores.values())
    return frontal, lateral, len(scores), scores


@dataclass
class Row:
    sample_id: str
    uid: str
    split: str
    caption_frontal: bool | None
    caption_lateral: bool | None
    sym_frontal: bool
    sym_lateral: bool
    has_findings: bool
    has_impression: bool
    n_images_declared: int
    n_images_found: int
    findings_placeholder_only: bool
    impression_placeholder_only: bool
    caption: str = ""
    symmetry: dict[str, float] = field(default_factory=dict)
    report_equals_findings: bool | None = None
    impression: str = ""

    def frontal(self, source: str) -> bool | None:
        return self.caption_frontal if source == "caption" else self.sym_frontal

    def lateral(self, source: str) -> bool | None:
        return self.caption_lateral if source == "caption" else self.sym_lateral

    def as_record(self) -> dict[str, Any]:
        def flag(value: bool | None) -> str:
            return "" if value is None else str(int(value))

        return {
            "sample_id": self.sample_id,
            "uid": self.uid,
            "split": self.split,
            "caption_frontal": flag(self.caption_frontal),
            "caption_lateral": flag(self.caption_lateral),
            "sym_frontal": int(self.sym_frontal),
            "sym_lateral": int(self.sym_lateral),
            "has_findings": int(self.has_findings),
            "has_impression": int(self.has_impression),
            "n_images_declared": self.n_images_declared,
            "n_images_found": self.n_images_found,
            "findings_placeholder_only": int(self.findings_placeholder_only),
            "impression_placeholder_only": int(self.impression_placeholder_only),
            "caption": self.caption,
            "symmetry": json.dumps(
                {key: round(value, 4) for key, value in sorted(self.symmetry.items())}
            ),
        }


def _build_row(
    *,
    sample_id: str,
    uid: str,
    split: str,
    study: OpenIStudy | None,
    candidates: Sequence[tuple[str, Path]],
    cache: SymmetryCache,
    threshold: float,
    with_images: bool,
    report: str | None = None,
) -> Row:
    caption = study.caption if study else ""
    caption_frontal, caption_lateral = caption_views(caption)

    if with_images:
        sym_frontal, sym_lateral, found, scores = symmetry_views(
            candidates, cache, threshold
        )
    else:
        found = sum(1 for _, path in candidates if path.is_file())
        sym_frontal = sym_lateral = False
        scores = {}

    findings = study.findings if study else ""
    impression = study.impression if study else ""
    return Row(
        sample_id=sample_id,
        uid=uid,
        split=split,
        caption_frontal=caption_frontal,
        caption_lateral=caption_lateral,
        sym_frontal=sym_frontal,
        sym_lateral=sym_lateral,
        has_findings=bool(findings),
        has_impression=bool(impression),
        n_images_declared=len(candidates),
        n_images_found=found,
        findings_placeholder_only=bool(findings) and _is_placeholder_only(findings),
        impression_placeholder_only=bool(impression)
        and _is_placeholder_only(impression),
        caption=caption,
        symmetry=scores,
        report_equals_findings=(
            None if report is None else _normalize(report) == _normalize(findings)
        ),
        impression=impression,
    )


def analyze_original(
    reports_dir: Path,
    images_dir: Path,
    cache: SymmetryCache,
    threshold: float,
    with_images: bool,
) -> tuple[list[Row], dict[str, OpenIStudy]]:
    rows: list[Row] = []
    by_uid: dict[str, OpenIStudy] = {}

    files = report_paths(reports_dir)
    for index, xml_path in enumerate(files, start=1):
        study = parse_report(xml_path)
        by_uid[study.uid] = study
        rows.append(
            _build_row(
                sample_id=study.uid,
                uid=study.uid,
                split="original",
                study=study,
                candidates=[
                    (name, images_dir / f"{name}.png") for name in study.parent_images
                ],
                cache=cache,
                threshold=threshold,
                with_images=with_images,
            )
        )
        if index % 500 == 0:
            print(f"  … {index}/{len(files)} referti", file=sys.stderr)

    return rows, by_uid


def analyze_r2gen(
    annotation_path: Path,
    images_dir: Path,
    by_uid: Mapping[str, OpenIStudy],
    cache: SymmetryCache,
    threshold: float,
    with_images: bool,
) -> list[Row]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows: list[Row] = []

    for split in ("train", "val", "test"):
        records = annotation.get(split, [])
        for index, record in enumerate(records, start=1):
            sample_id = str(record["id"])
            uid = sample_id.split("_", 1)[0]
            rows.append(
                _build_row(
                    sample_id=sample_id,
                    uid=uid,
                    split=split,
                    study=by_uid.get(uid),
                    candidates=[
                        (rel, images_dir / rel) for rel in record.get("image_path", [])
                    ],
                    cache=cache,
                    threshold=threshold,
                    with_images=with_images,
                    report=str(record.get("report", "")),
                )
            )
            if index % 500 == 0:
                print(f"  … {split} {index}/{len(records)}", file=sys.stderr)

    return rows


def _pct(part: int, total: int) -> str:
    return f"{part / total:6.1%}" if total else "     —"


def _tally(values: Iterable[bool | None]) -> tuple[int, int, int]:
    present = absent = unknown = 0
    for value in values:
        if value is None:
            unknown += 1
        elif value:
            present += 1
        else:
            absent += 1
    return present, absent, unknown


def _coverage_table(
    rows: Sequence[Row],
    title: str,
    with_images: bool,
    text_suffix: str = "",
) -> list[str]:
    total = len(rows)
    checks: list[tuple[str, tuple[int, int, int]]] = []
    if with_images:
        checks += [
            ("frontale [caption]", _tally(r.caption_frontal for r in rows)),
            ("laterale [caption]", _tally(r.caption_lateral for r in rows)),
            ("frontale [simmetria]", _tally(r.sym_frontal for r in rows)),
            ("laterale [simmetria]", _tally(r.sym_lateral for r in rows)),
        ]
    else:
        checks += [
            ("frontale [caption]", _tally(r.caption_frontal for r in rows)),
            ("laterale [caption]", _tally(r.caption_lateral for r in rows)),
        ]
    checks += [
        (f"findings{text_suffix}", _tally(r.has_findings for r in rows)),
        (f"impression{text_suffix}", _tally(r.has_impression for r in rows)),
    ]

    lines = [f"{title}  (N = {total})", ""]
    lines.append(
        f"  {'campo':<24}{'presenti':>9} {'%':>7}{'assenti':>9} {'%':>7}{'ignoti':>8}"
    )
    lines.append(f"  {'-' * 64}")
    for label, (present, absent, unknown) in checks:
        lines.append(
            f"  {label:<24}{present:>9} {_pct(present, total):>7}"
            f"{absent:>9} {_pct(absent, total):>7}{unknown:>8}"
        )
    return lines


def _combination_table(rows: Sequence[Row], source: str, with_images: bool) -> list[str]:
    total = len(rows)
    lines: list[str] = []

    if with_images:
        combos: Counter[str] = Counter()
        for row in rows:
            frontal, lateral = row.frontal(source), row.lateral(source)
            if frontal is None or lateral is None:
                combos["indeterminato"] += 1
            elif frontal and lateral:
                combos["entrambe le viste"] += 1
            elif frontal:
                combos["solo frontale"] += 1
            elif lateral:
                combos["solo laterale"] += 1
            else:
                combos["nessuna vista"] += 1
        lines.append(f"  viste  [fonte: {source}]")
        for label in (
            "entrambe le viste",
            "solo frontale",
            "solo laterale",
            "nessuna vista",
            "indeterminato",
        ):
            count = combos.get(label, 0)
            lines.append(f"    {label:<20}{count:>9} {_pct(count, total):>7}")
        lines.append("")

    text_combos = Counter(
        (
            "findings + impression"
            if r.has_findings and r.has_impression
            else "solo findings"
            if r.has_findings
            else "solo impression"
            if r.has_impression
            else "nessuna sezione"
        )
        for r in rows
    )
    lines.append("  testo")
    for label in (
        "findings + impression",
        "solo findings",
        "solo impression",
        "nessuna sezione",
    ):
        count = text_combos.get(label, 0)
        lines.append(f"    {label:<20}{count:>9} {_pct(count, total):>7}")

    complete = sum(
        1
        for r in rows
        if r.has_findings
        and r.has_impression
        and (not with_images or (r.frontal(source) and r.lateral(source)))
    )
    lines.append("")
    lines.append(
        f"  {'STUDI COMPLETI':<24}{complete:>9} {_pct(complete, total):>7}"
        + (
            "  (2 viste + findings + impression)"
            if with_images
            else "  (findings + impression)"
        )
    )

    counts = Counter(r.n_images_found for r in rows)
    lines.append("")
    lines.append("  file immagine trovati per studio")
    for n in sorted(counts):
        lines.append(
            f"    {n:>2} immagini        {counts[n]:>9} {_pct(counts[n], total):>7}"
        )
    return lines


def exclusion_reason(row: Row) -> str:
    if not row.has_findings:
        return "findings assenti"
    if row.n_images_found < 2:
        return f"studio con solo {row.n_images_found} radiografie"
    return "non spiegato"


def analyze_exclusions(
    original_rows: Sequence[Row], r2gen_rows: Sequence[Row]
) -> tuple[list[Row], list[Row], list[str]]:
    used = {row.uid for row in r2gen_rows}
    included = [row for row in original_rows if row.uid in used]
    excluded = [row for row in original_rows if row.uid not in used]

    total_o, total_e = len(original_rows), len(excluded)
    lines = [
        f"OpenI {total_o} studi  →  R2Gen {len(used)} studi  "
        f"(scartati {total_e}, {total_e / total_o:.1%})",
        "",
        f"  {'campo':<24}{'ripresi':>10}{'scartati':>12}",
        f"  {'-' * 46}",
    ]

    def share(rows: Sequence[Row], predicate) -> str:
        return _pct(sum(1 for r in rows if predicate(r)), len(rows)) if rows else "  —"

    comparisons = [
        ("findings", lambda r: r.has_findings),
        ("impression", lambda r: r.has_impression),
        ("frontale [caption]", lambda r: r.caption_frontal is True),
        ("laterale [caption]", lambda r: r.caption_lateral is True),
        ("almeno 2 file immagine", lambda r: r.n_images_found >= 2),
    ]
    for label, predicate in comparisons:
        lines.append(
            f"  {label:<24}{share(included, predicate):>10}{share(excluded, predicate):>12}"
        )

    reasons = Counter(exclusion_reason(row) for row in excluded)
    lines += ["", "  motivo dello scarto (cascata a priorita')", ""]
    for reason, count in reasons.most_common():
        lines.append(f"    {reason:<32}{count:>7} {_pct(count, total_e):>8}")

    rule = {
        row.uid
        for row in original_rows
        if row.has_findings and row.n_images_found >= 2
    }
    lines += [
        "",
        "  regola dedotta: FINDINGS presenti  AND  almeno 2 file immagine",
        f"    studi che la soddisfano       : {len(rule)}",
        f"    studi effettivamente in R2Gen : {len(used)}",
        f"    falsi positivi (regola si', R2Gen no) : {len(rule - used)}",
        f"    falsi negativi (regola no, R2Gen si') : {len(used - rule)}",
        "    → la regola ricostruisce lo split ESATTAMENTE"
        if rule == used
        else "    → ATTENZIONE: la regola NON ricostruisce lo split",
    ]

    incomplete = [
        row for row in original_rows if row.n_images_declared != row.n_images_found
    ]
    lines += [
        "",
        f"  integrita': studi con immagini dichiarate != trovate su disco: "
        f"{len(incomplete)}/{total_o}",
        "    → gli studi scartati per le immagini hanno davvero meno di 2 radiografie,"
        if not incomplete
        else "    → ATTENZIONE: alcuni file immagine mancano dal disco,",
        "      non e' un problema di download"
        if not incomplete
        else "      lo scarto potrebbe dipendere dai file mancanti",
    ]

    distribution = Counter(row.n_images_found for row in excluded)
    lines += ["", "  radiografie per studio, negli studi scartati"]
    for n in sorted(distribution):
        lines.append(
            f"    {n:>2} immagini        {distribution[n]:>7} {_pct(distribution[n], total_e):>8}"
        )
    return included, excluded, lines


IMPRESSION_RECOVERABLE = "recuperabile da OpenI"
IMPRESSION_PLACEHOLDER = "presente ma solo placeholder"
IMPRESSION_MISSING = "assente anche in OpenI"


def impression_status(row: Row) -> str:
    if not row.has_impression:
        return IMPRESSION_MISSING
    if row.impression_placeholder_only:
        return IMPRESSION_PLACEHOLDER
    return IMPRESSION_RECOVERABLE


def analyze_impression(r2gen_rows: Sequence[Row]) -> tuple[list[Row], list[str]]:
    total = len(r2gen_rows)
    checked = [row for row in r2gen_rows if row.report_equals_findings is not None]
    equal = sum(1 for row in checked if row.report_equals_findings)

    lines = [
        "  premessa verificata sui dati:",
        f"    'report' di annotation.json identico ai FINDINGS OpenI : "
        f"{equal}/{len(checked)} ({_pct(equal, len(checked)).strip()})",
    ]
    if equal == len(checked):
        lines.append(
            "    → l'impression non e' MAI nel report R2Gen: manca per costruzione"
        )
        lines.append("      su tutti e 2955 gli studi. La domanda vera e' se sia")
        lines.append("      recuperabile dal referto OpenI di origine.")
    else:
        lines.append(
            "    → ATTENZIONE: in alcuni studi il report NON coincide con i findings,"
        )
        lines.append("      il campo potrebbe contenere anche altro.")

    statuses = Counter(impression_status(row) for row in r2gen_rows)
    lines += ["", f"  {'stato dell impression':<32}{'studi':>8}{'%':>9}", f"  {'-' * 49}"]
    for label in (IMPRESSION_RECOVERABLE, IMPRESSION_PLACEHOLDER, IMPRESSION_MISSING):
        count = statuses.get(label, 0)
        lines.append(f"  {label:<32}{count:>8}{_pct(count, total):>9}")

    lines += ["", "  ripartizione per split", ""]
    lines.append(
        f"    {'split':<8}{'totale':>8}{'recuperabile':>14}{'placeholder':>13}{'assente':>9}"
    )
    for split in ("train", "val", "test"):
        subset = [row for row in r2gen_rows if row.split == split]
        per_split = Counter(impression_status(row) for row in subset)
        lines.append(
            f"    {split:<8}{len(subset):>8}"
            f"{per_split.get(IMPRESSION_RECOVERABLE, 0):>14}"
            f"{per_split.get(IMPRESSION_PLACEHOLDER, 0):>13}"
            f"{per_split.get(IMPRESSION_MISSING, 0):>9}"
        )

    unrecoverable = [
        row for row in r2gen_rows if impression_status(row) != IMPRESSION_RECOVERABLE
    ]
    if unrecoverable:
        lines += ["", "  studi senza impression utilizzabile:", ""]
        for row in unrecoverable:
            lines.append(
                f"    {row.sample_id:<24}{row.split:<7}{impression_status(row)}"
            )
    return unrecoverable, lines


def _missing_lists(rows: Sequence[Row], with_images: bool) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    if with_images:
        lists["senza_frontale_caption"] = [
            r.sample_id for r in rows if r.caption_frontal is False
        ]
        lists["senza_laterale_caption"] = [
            r.sample_id for r in rows if r.caption_lateral is False
        ]
        lists["viste_indeterminate_caption"] = [
            r.sample_id for r in rows if r.caption_frontal is None
        ]
        lists["senza_frontale_simmetria"] = [
            r.sample_id for r in rows if not r.sym_frontal
        ]
        lists["senza_laterale_simmetria"] = [
            r.sample_id for r in rows if not r.sym_lateral
        ]
        lists["senza_file_immagine"] = [r.sample_id for r in rows if not r.n_images_found]
    lists["senza_findings"] = [r.sample_id for r in rows if not r.has_findings]
    lists["senza_impression"] = [r.sample_id for r in rows if not r.has_impression]
    return lists


def write_rows_csv(rows: Iterable[Row], path: Path) -> None:
    records = [row.as_record() for row in rows]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_missing_lists(lists: Mapping[str, Sequence[str]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, ids in lists.items():
        (directory / f"{name}.txt").write_text(
            "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
        )


def _emit(lines: Sequence[str], sink: list[str]) -> None:
    for line in lines:
        print(line)
        sink.append(line)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--original", type=Path, default=Path("dataset/iu-xray/iu_xray_original"))
    parser.add_argument("--r2gen", type=Path, default=Path("dataset/iu-xray/iu_xray_r2gen"))
    parser.add_argument("--out", type=Path, default=Path("outputs/dataset_analysis"))
    parser.add_argument(
        "--view-source",
        choices=("caption", "symmetry"),
        default="caption",
        help="segnale usato nelle tabelle di combinazione delle viste",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="soglia del proxy di simmetria: >= frontale, < laterale",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="salta la lettura delle immagini (veloce, solo testo e caption)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    original_dir = root / args.original
    r2gen_dir = root / args.r2gen
    out_dir = root / args.out
    with_images = not args.no_images
    source = args.view_source

    reports_dir = original_dir / "reports"
    if not reports_dir.is_dir():
        parser.error(f"cartella referti non trovata: {reports_dir}")

    cache = SymmetryCache(out_dir / "symmetry_cache.json" if with_images else None)
    transcript: list[str] = []

    print("[analyze] parsing referti OpenI…", file=sys.stderr)
    original_rows, by_uid = analyze_original(
        reports_dir, original_dir / "images", cache, args.threshold, with_images
    )
    cache.save()

    _emit(["", "=" * 72], transcript)
    _emit(
        _coverage_table(
            original_rows, "SORGENTE OpenI — dataset/iu-xray/iu_xray_original", with_images
        ),
        transcript,
    )
    _emit([""], transcript)
    _emit(_combination_table(original_rows, source, with_images), transcript)

    write_rows_csv(original_rows, out_dir / "original_samples.csv")
    write_missing_lists(
        _missing_lists(original_rows, with_images), out_dir / "original_missing"
    )

    annotation_path = r2gen_dir / "annotation.json"
    if annotation_path.is_file():
        print("[analyze] analisi split R2Gen…", file=sys.stderr)
        r2gen_rows = analyze_r2gen(
            annotation_path,
            r2gen_dir / "images",
            by_uid,
            cache,
            args.threshold,
            with_images,
        )
        cache.save()

        for split in ("train", "val", "test"):
            subset = [row for row in r2gen_rows if row.split == split]
            _emit(["", "=" * 72], transcript)
            _emit(
                _coverage_table(
                    subset, f"SPLIT R2Gen — {split}", with_images, " [OpenI]"
                ),
                transcript,
            )
            _emit([""], transcript)
            _emit(_combination_table(subset, source, with_images), transcript)
            write_missing_lists(
                _missing_lists(subset, with_images), out_dir / "r2gen_missing" / split
            )

        _emit(["", "=" * 72], transcript)
        _emit(
            _coverage_table(
                r2gen_rows, "SPLIT R2Gen — totale", with_images, " [OpenI]"
            ),
            transcript,
        )
        _emit([""], transcript)
        _emit(_combination_table(r2gen_rows, source, with_images), transcript)

        orphans = [row.sample_id for row in r2gen_rows if row.uid not in by_uid]
        _emit(["", f"  studi R2Gen senza XML OpenI corrispondente: {len(orphans)}"], transcript)

        _emit(
            ["", "=" * 72, "IMPRESSION NEGLI STUDI R2Gen (recuperata da OpenI)", ""],
            transcript,
        )
        unrecoverable, impression_lines = analyze_impression(r2gen_rows)
        _emit(impression_lines, transcript)

        impression_dir = out_dir / "r2gen_impression"
        impression_dir.mkdir(parents=True, exist_ok=True)
        with (impression_dir / "stato_impression.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "split", "stato", "impression"])
            for row in r2gen_rows:
                writer.writerow(
                    [row.sample_id, row.split, impression_status(row), row.impression]
                )
        for label, filename in (
            (IMPRESSION_MISSING, "assente_anche_in_openi.txt"),
            (IMPRESSION_PLACEHOLDER, "solo_placeholder.txt"),
        ):
            ids = [
                f"{row.sample_id}\t{row.split}"
                for row in r2gen_rows
                if impression_status(row) == label
            ]
            (impression_dir / filename).write_text(
                "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
            )

        _emit(["", "=" * 72, "STUDI OpenI ESCLUSI DALLO SPLIT R2Gen", ""], transcript)
        _, excluded, exclusion_lines = analyze_exclusions(original_rows, r2gen_rows)
        _emit(exclusion_lines, transcript)

        (out_dir / "esclusi_da_r2gen.txt").write_text(
            "\n".join(row.uid for row in excluded) + "\n", encoding="utf-8"
        )
        with (out_dir / "esclusi_da_r2gen.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["motivo", *excluded[0].as_record()]
            )
            writer.writeheader()
            for row in excluded:
                writer.writerow({"motivo": exclusion_reason(row), **row.as_record()})

        write_rows_csv(r2gen_rows, out_dir / "r2gen_samples.csv")
        write_missing_lists(
            _missing_lists(r2gen_rows, with_images), out_dir / "r2gen_missing" / "all"
        )
        (out_dir / "r2gen_missing" / "senza_xml_openi.txt").write_text(
            "\n".join(orphans) + ("\n" if orphans else ""), encoding="utf-8"
        )
    else:
        _emit(["", f"[analyze] annotation.json non trovato: {annotation_path}"], transcript)

    _emit(
        [
            "",
            "=" * 72,
            "  viste [caption]    : <caption> dei <parentImage> — etichetta di studio,",
            "                       dice quali proiezioni comprende l'esame",
            f"  viste [simmetria]  : proxy per-immagine, soglia {args.threshold} — INDICATIVO:",
            "                       la distribuzione non e' bimodale, non separa le classi",
            "  findings/impression: <AbstractText Label=...> dell'XML OpenI.",
            "                       ATTENZIONE sulle tabelle R2Gen: annotation.json NON",
            "                       contiene ne' findings ne' impression, ha un solo campo",
            "                       'report' (che coincide al 100% con i FINDINGS OpenI).",
            "                       Le colonne [OpenI] dicono se lo studio DI ORIGINE aveva",
            "                       quella sezione, recuperata col join sull'uId: serve a",
            "                       sapere quali studi sarebbero inutilizzabili per una",
            "                       variante con target impression. Di conseguenza la riga",
            "                       'findings [OpenI]' e' tautologicamente al 100%.",
            f"  output             : {out_dir}",
            "=" * 72,
        ],
        transcript,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text("\n".join(transcript) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
