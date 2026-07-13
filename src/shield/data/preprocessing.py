from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

_NORMAL_TOKENS = {"normal"}
_UNLABELED_TOKENS = {"no indexing", "technical quality of image unsatisfactory"}

OTHER_LABEL = "Other"
NO_FINDING_LABEL = "No Finding"
UNLABELED_LABEL = "Unlabeled"

_CHEXPERT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Pneumothorax", ("pneumothorax",)),
    ("Pleural Effusion", ("pleural effusion", "effusion", "hydrothorax")),
    ("Edema", ("edema", "pulmonary congestion", "vascular congestion")),
    ("Consolidation", ("consolidation", "airspace disease", "air space disease")),
    ("Pneumonia", ("pneumonia",)),
    ("Atelectasis", ("atelectasis", "atelectases")),
    ("Lung Lesion", ("nodule", "mass", "granuloma", "granulomatous", "lesion")),
    (
        "Lung Opacity",
        (
            "opacity",
            "opacities",
            "infiltrate",
            "infiltration",
            "density",
            "densities",
            "fibrosis",
            "scarring",
            "interstitial",
            "reticular",
        ),
    ),
    ("Cardiomegaly", ("cardiomegaly",)),
    ("Enlarged Cardiomediastinum", ("mediastinum", "mediastinal", "cardiomediastin")),
    ("Fracture", ("fracture",)),
    ("Pleural Other", ("pleural thickening", "pleural", "pleura")),
    (
        "Support Devices",
        (
            "catheter",
            "device",
            "pacemaker",
            "tube",
            "picc",
            "port",
            "sternotomy",
            "surgical",
            "suture",
            "clip",
            "stent",
            "prosthesis",
            "implant",
        ),
    ),
]

CHEXPERT_CATEGORIES: list[str] = [label for label, _ in _CHEXPERT_RULES] + [
    NO_FINDING_LABEL,
    OTHER_LABEL,
    UNLABELED_LABEL,
]


def map_mesh_to_chexpert(mesh_majors: list[str]) -> list[str]:
    matched: list[str] = []
    for label, keywords in _CHEXPERT_RULES:
        for tag in mesh_majors:
            lowered = tag.lower()
            if any(keyword in lowered for keyword in keywords):
                matched.append(label)
                break
    if matched:
        return matched
    lowered_tags = [tag.lower() for tag in mesh_majors]
    if not mesh_majors or all(tag in _NORMAL_TOKENS for tag in lowered_tags):
        return [NO_FINDING_LABEL]
    if all(tag in _NORMAL_TOKENS | _UNLABELED_TOKENS for tag in lowered_tags):
        return [UNLABELED_LABEL]
    return [OTHER_LABEL]


def _clean(text: str | None) -> str:
    return " ".join(text.split()) if text else ""


def parse_openi_report(
    xml_path: str | Path,
    findings_label: str = "FINDINGS",
    impression_label: str = "IMPRESSION",
) -> dict[str, Any] | None:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return None

    uid_node = root.find("uId")
    uid = uid_node.get("id") if uid_node is not None else Path(xml_path).stem

    categories: list[str] = []
    for major in root.findall("MeSH/major"):
        value = _clean(major.text)
        if value and value.lower() not in _NORMAL_TOKENS:
            categories.append(value.lower())

    findings = ""
    impression = ""
    for node in root.findall(".//AbstractText"):
        label = (node.get("Label") or "").upper()
        if label == findings_label.upper():
            findings = _clean(node.text)
        elif label == impression_label.upper():
            impression = _clean(node.text)

    images: list[dict[str, str]] = []
    for parent in root.findall("parentImage"):
        image_id = parent.get("id")
        if not image_id:
            continue
        images.append({"filename": f"{image_id}.png"})

    return {
        "uid": uid,
        "findings": findings,
        "impression": impression,
        "categories": categories or ["normal"],
        "images": images,
    }


def get_translator(
    backend: str, source_lang: str = "en", target_lang: str = "en"
) -> Callable[[str], str]:
    if backend == "passthrough":
        return lambda text: text
    raise NotImplementedError(
        f"Backend di traduzione '{backend}' non configurato. "
        f"Implementalo in get_translator() o usa 'passthrough' per lo scaffolding."
    )


def default_system_prompt(
    target_lang: str,
    target: str = "findings_impression",
    target_source: str = "openi",
) -> str:
    if target_lang == "it":
        persona = (
            "Sei un radiologo esperto. "
            "Ti verranno fornite una o più immagini mediche. "
        )
        if target_source == "r2gen":
            return persona + (
                "Il tuo compito è descrivere in modo conciso i reperti visibili "
                "nell'immagine o nelle immagini, incluse le strutture anatomiche, "
                "le anomalie e le osservazioni rilevanti. "
                "NON includere altre sezioni o preamboli."
            )
        if target == "findings_impression":
            return persona + (
                "Il tuo compito è: "
                "1) Descrivere in modo conciso i reperti visibili nell'immagine o nelle immagini, "
                "incluse le strutture anatomiche, le anomalie e le osservazioni rilevanti. "
                "2) Fornire un'impressione clinica concisa che riassuma i reperti principali. "
                "Fornisci la risposta in ESATTAMENTE questo formato:\n"
                "Reperti:\n<i tuoi reperti dettagliati qui>\n\n"
                "Impressione:\n<la tua impressione concisa qui>\n\n"
                "NON includere altre sezioni o preamboli."
            )
        return persona + (
            "Il tuo compito è descrivere in modo conciso i reperti visibili "
            "nell'immagine o nelle immagini, incluse le strutture anatomiche, "
            "le anomalie e le osservazioni rilevanti. "
            "Fornisci la risposta in ESATTAMENTE questo formato:\n"
            "Reperti:\n<i tuoi reperti dettagliati qui>\n\n"
            "NON includere altre sezioni o preamboli."
        )
    persona = (
        "You are an expert radiologist. "
        "You will be provided with one or more medical images. "
    )
    if target_source == "r2gen":
        return persona + (
            "Your task is to concisely describe the findings visible in the image "
            "or images, including anatomical structures, abnormalities, and relevant "
            "observations. "
            "Do not include any other sections or preamble."
        )
    if target == "findings_impression":
        return persona + (
            "Your task is to: "
            "1) Concisely describe the findings visible in the image or images, "
            "including anatomical structures, abnormalities, and relevant observations. "
            "2) Provide a concise clinical impression that summarizes the main findings. "
            "Provide your answer in EXACTLY this format:\n"
            "Findings:\n<your detailed findings here>\n\n"
            "Impression:\n<your concise impression here>\n\n"
            "Do not include any other sections or preamble."
        )
    return persona + (
        "Your task is to concisely describe the findings visible in the image "
        "or images, including anatomical structures, abnormalities, and relevant "
        "observations. "
        "Provide your answer in EXACTLY this format:\n"
        "Findings:\n<your detailed findings here>\n\n"
        "Do not include any other sections or preamble."
    )


def default_user_prompt(
    target_lang: str, target: str, views: str, target_source: str = "openi"
) -> str:
    lang = "it" if target_lang == "it" else "en"
    n_views = 2 if views == "frontal_lateral" else 1
    if lang == "it":
        imgs = (
            "una radiografia del torace in proiezione frontale e una in proiezione laterale"
            if n_views == 2
            else "una radiografia del torace in proiezione frontale"
        )
        intro = f"Ti vengono fornite {imgs}. "
        if target_source == "r2gen":
            return intro + "Descrivi in modo conciso i reperti visibili."
        if target == "findings_impression":
            return (
                intro
                + "Descrivi i reperti visibili e fornisci un'impressione clinica concisa. "
                "Rispondi ESATTAMENTE in questo formato:\n"
                "Reperti:\n<i tuoi reperti qui>\n\n"
                "Impressione:\n<la tua impressione qui>\n"
                "NON includere altre sezioni o preamboli."
            )
        return (
            intro + "Descrivi in modo conciso i reperti visibili. "
            "Rispondi ESATTAMENTE in questo formato:\n"
            "Reperti:\n<i tuoi reperti qui>\n"
            "NON includere altre sezioni o preamboli."
        )
    imgs = (
        "a frontal and a lateral chest X-ray"
        if n_views == 2
        else "a frontal chest X-ray"
    )
    intro = f"You are given {imgs}. "
    if target_source == "r2gen":
        return intro + "Concisely describe the visible findings."
    if target == "findings_impression":
        return (
            intro
            + "Describe the visible findings and provide a concise clinical impression. "
            "Answer EXACTLY in this format:\n"
            "Findings:\n<your findings here>\n\n"
            "Impression:\n<your impression here>\n"
            "Do not include any other sections or preamble."
        )
    return (
        intro + "Concisely describe the visible findings. "
        "Answer EXACTLY in this format:\n"
        "Findings:\n<your findings here>\n"
        "Do not include any other sections or preamble."
    )


def normalize_image(src: str | Path, dst: str | Path) -> None:
    from PIL import Image

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(dst)


_HEADERS: dict[tuple[str, str], tuple[str, ...]] = {
    ("en", "findings"): ("Findings:",),
    ("en", "findings_impression"): ("Findings:", "Impression:"),
    ("it", "findings"): ("Reperti:",),
    ("it", "findings_impression"): ("Reperti:", "Impressione:"),
}


def format_report(findings: str, impression: str, target_lang: str, target: str) -> str:
    lang = "it" if target_lang == "it" else "en"
    if target == "findings_impression":
        h_find, h_impr = _HEADERS[(lang, "findings_impression")]
        return f"{h_find}\n{findings.strip()}\n\n{h_impr}\n{impression.strip()}"
    (h_find,) = _HEADERS[(lang, "findings")]
    return f"{h_find}\n{findings.strip()}"


_SECTION_HEADERS = ("findings:", "impression:", "reperti:", "impressione:")
_R2GEN_PUNCT = re.compile(r"[.,?;*!%^&_+():-\[\]{}]")


def clean_report_r2gen(text: str) -> str:
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        header_only = False
        for header in _SECTION_HEADERS:
            if low == header or low == header[:-1]:
                header_only = True
                break
            if low.startswith(header):
                stripped = stripped[len(header) :].strip()
                break
        if header_only:
            continue
        kept.append(stripped)
    joined = " ".join(kept)
    for _ in range(3):
        joined = joined.replace("..", ".")
    for pattern, replacement in (
        ("1. ", ""),
        (". 2. ", ". "),
        (". 3. ", ". "),
        (". 4. ", ". "),
        (". 5. ", ". "),
        (" 2. ", ". "),
        (" 3. ", ". "),
        (" 4. ", ". "),
        (" 5. ", ". "),
    ):
        joined = joined.replace(pattern, replacement)
    tokens: list[str] = []
    for sentence in joined.strip().lower().split(". "):
        cleaned = (
            sentence.replace('"', "")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
        )
        cleaned = _R2GEN_PUNCT.sub("", cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            tokens.append(cleaned)
    return " . ".join(tokens) + " ." if tokens else ""


def build_record(
    report: Mapping[str, Any],
    images: list[Mapping[str, str]],
    rel_images: list[str],
    assistant_text: str,
    system_prompt: str,
    user_prompt: str,
    source_lang: str,
    target_lang: str,
    translation_method: str,
) -> dict[str, Any]:
    projections = [im["projection"] for im in images]
    content: list[dict[str, str]] = [
        {"type": "image", "image": rel} for rel in rel_images
    ]
    content.append({"type": "text", "text": user_prompt})
    return {
        "id": report["uid"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_text},
        ],
        "images": list(rel_images),
        "factors": {
            "diagnostic_category": list(report["categories"]),
            "projection": (
                "+".join(projections) if len(projections) > 1 else projections[0]
            ),
            "views": projections,
            "task_type": "report_generation",
        },
        "mesh_raw": list(report.get("mesh_raw", [])),
        "provenance": {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation_method": translation_method,
        },
    }


def compute_stats(
    records_by_split: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    category_counts: dict[str, int] = defaultdict(int)
    projection_counts: dict[str, int] = defaultdict(int)
    for records in records_by_split.values():
        for record in records:
            factors = record["factors"]
            for category in factors["diagnostic_category"]:
                category_counts[category] += 1
            projection_counts[factors["projection"]] += 1
    return {
        "n_examples": {
            split: len(records) for split, records in records_by_split.items()
        },
        "diagnostic_category": dict(category_counts),
        "projection": dict(projection_counts),
    }


def category_coverage_by_split(
    records_by_split: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0}
    )
    for split, records in records_by_split.items():
        for record in records:
            for category in record["factors"]["diagnostic_category"]:
                coverage[category][split] += 1
    return {category: dict(counts) for category, counts in sorted(coverage.items())}


def categories_absent_from_split(
    coverage: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    absent: dict[str, list[str]] = {}
    for category, counts in coverage.items():
        missing = [
            split for split in ("train", "val", "test") if counts.get(split, 0) == 0
        ]
        if missing:
            absent[category] = missing
    return absent


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _materialize_images(
    srcs: list[Path], rel_names: list[str], output_dir: Path
) -> list[str]:
    rel_images: list[str] = []
    for src, name in zip(srcs, rel_names):
        rel = f"images_normalized/{name}"
        normalize_image(src, output_dir / rel)
        rel_images.append(rel)
    return rel_images


def _view_symmetry(png: str | Path) -> float:
    import numpy as np
    from PIL import Image

    with Image.open(png) as image:
        arr = np.asarray(image.convert("L").resize((128, 128)), dtype="float32")
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return float((arr * np.fliplr(arr)).mean())


def _r2gen_image_pair(
    images_dir: Path,
    study_id: str,
    views_mode: str,
    frontal_selection: str = "order",
) -> tuple[list[Path], list[str], list[dict[str, str]], bool]:
    pair = [images_dir / study_id / "0.png", images_dir / study_id / "1.png"]
    swapped = False
    if (
        frontal_selection == "symmetry"
        and views_mode == "frontal"
        and pair[0].exists()
        and pair[1].exists()
        and _view_symmetry(pair[1]) > _view_symmetry(pair[0])
    ):
        pair = [pair[1], pair[0]]
        swapped = True
    srcs = pair if views_mode == "frontal_lateral" else pair[:1]
    projections = ["frontal", "lateral"][: len(srcs)]
    selected = [
        {"filename": src.name, "projection": proj}
        for src, proj in zip(srcs, projections)
    ]
    rel_names = [f"{study_id}/{src.name}" for src in srcs]
    return srcs, rel_names, selected, swapped


def _coverage_guard(
    n_missing: int, total: int, coverage_tolerance: float, context: str
) -> None:
    if n_missing <= 0:
        return
    frac = n_missing / total if total else 0.0
    print(
        f"[preprocess] ⚠️ copertura R2Gen ({context}): {n_missing}/{total} studi "
        f"senza record ({frac:.2%}); tolleranza={coverage_tolerance:.2%}."
    )
    if frac > coverage_tolerance:
        raise RuntimeError(
            f"Copertura R2Gen insufficiente ({context}): {n_missing}/{total} studi mancanti "
            f"({frac:.2%}) oltre la tolleranza {coverage_tolerance:.2%} (split.coverage_tolerance). "
            f"Verifica raw/filtri/target, oppure alza la tolleranza dichiarando il sottoinsieme "
            f"— o passa a split.target_source='r2gen' per la copertura per costruzione."
        )


def _make_record(
    report: Mapping[str, Any],
    selected: list[Mapping[str, str]],
    srcs: list[Path],
    rel_names: list[str],
    output_dir: Path,
    system_prompt: str,
    user_prompt: str,
    source_lang: str,
    target_lang: str,
    backend: str,
    target: str,
    record_id: str,
) -> dict[str, Any]:
    rel_images = _materialize_images(srcs, rel_names, output_dir)
    assistant = format_report(
        report["findings_tr"], report["impression_tr"], target_lang, target
    )
    record = build_record(
        report,
        selected,
        rel_images,
        assistant,
        system_prompt,
        user_prompt,
        source_lang,
        target_lang,
        backend,
    )
    record["id"] = record_id
    return record


_R2GEN_IMAGE_ID = re.compile(r"^(CXR\d+(?:_\d+)?_IM-\d+(?:-\d+)?)-\d+$")
_R2GEN_STUDY_ID = re.compile(r"^CXR\d+(?:_\d+)?_IM-\d+(?:-\d+)?$")


def r2gen_study_id(image_filename: str) -> str:
    stem = image_filename.rsplit(".", 1)[0]
    image_match = _R2GEN_IMAGE_ID.match(stem)
    if image_match is not None:
        return image_match.group(1)
    study_match = _R2GEN_STUDY_ID.match(stem)
    if study_match is None:
        raise ValueError(
            f"Filename immagine fuori dal pattern R2Gen atteso "
            f"'CXR<n>[_<k>]_IM-<n>[-<study>][-<img>]': "
            f"{image_filename!r}. Verifica i filename reali o adegua r2gen_study_id."
        )
    return study_match.group(0)


def load_r2gen_annotation(annotation_path: str | Path) -> dict[str, dict[str, str]]:
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        for record in data.get(split, []):
            mapping[record["id"]] = {"split": split, "report": record.get("report", "")}
    return mapping


def r2gen_coverage_diagnostic(
    reports: list[Mapping[str, Any]],
    annotation: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    by_study: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        for image in report.get("images", []):
            by_study.setdefault(r2gen_study_id(image["filename"]), report)

    total = len(annotation)
    n_no_openi = n_with_findings = 0
    n_empty_with_impr = n_empty_no_impr = 0
    for study_id in annotation:
        report = by_study.get(study_id)
        if report is None:
            n_no_openi += 1
        elif (report.get("findings") or "").strip():
            n_with_findings += 1
        elif (report.get("impression") or "").strip():
            n_empty_with_impr += 1
        else:
            n_empty_no_impr += 1
    n_gap = total - n_with_findings
    return {
        "n_r2gen_studies": total,
        "n_with_findings": n_with_findings,
        "n_empty_findings_with_impression": n_empty_with_impr,
        "n_empty_findings_no_impression": n_empty_no_impr,
        "n_no_openi_report": n_no_openi,
        "n_gap_findings": n_gap,
        "gap_fraction": round(n_gap / total, 4) if total else 0.0,
    }


def build_records_r2gen(
    reports: list[dict[str, Any]],
    *,
    split_cfg: Mapping[str, Any],
    project_root: Path,
    output_dir: Path,
    views_mode: str,
    target: str,
    target_lang: str,
    source_lang: str,
    backend: str,
    system_prompt: str,
    user_prompt: str,
    enforce_full_coverage: bool,
    attach_r2gen_reference: bool,
    coverage_tolerance: float = 0.0,
    frontal_selection: str = "order",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    annotation_path = _resolve(project_root, split_cfg["annotation"])
    r2gen_images_dir = _resolve(project_root, split_cfg["images_dir"])
    annotation = load_r2gen_annotation(annotation_path)

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    n_dropped_missing_image = 0
    n_frontal_swapped = 0
    n_duplicate_openi_refs = 0
    seen_ids: set[str] = set()
    processed_ids: set[str] = set()

    for report in reports:
        study_ids = sorted({r2gen_study_id(im["filename"]) for im in report["images"]})
        for study_id in study_ids:
            entry = annotation.get(study_id)
            if entry is None:
                continue
            if study_id in processed_ids:
                n_duplicate_openi_refs += 1
                continue
            processed_ids.add(study_id)
            srcs, rel_names, selected, swapped = _r2gen_image_pair(
                r2gen_images_dir, study_id, views_mode, frontal_selection
            )
            if not all(src.exists() for src in srcs):
                n_dropped_missing_image += 1
                continue
            n_frontal_swapped += int(swapped)
            record = _make_record(
                report,
                selected,
                srcs,
                rel_names,
                output_dir,
                system_prompt,
                user_prompt,
                source_lang,
                target_lang,
                backend,
                target,
                record_id=study_id,
            )
            if attach_r2gen_reference:
                record["r2gen_report"] = entry["report"]
            records_by_split[entry["split"]].append(record)
            seen_ids.add(study_id)

    missing = set(annotation) - seen_ids
    if enforce_full_coverage:
        _coverage_guard(
            len(missing),
            len(annotation),
            coverage_tolerance,
            f"OpenI-driven, views={views_mode}, target={target}",
        )
    if n_duplicate_openi_refs:
        print(
            f"[preprocess] ℹ️ {n_duplicate_openi_refs} riferimenti duplicati allo stesso "
            f"studio R2Gen da report OpenI distinti: emesso solo il primo (deterministico)."
        )
    n_records = sum(len(records) for records in records_by_split.values())
    assert n_records == len(
        seen_ids
    ), f"record emessi ({n_records}) != studi emessi ({len(seen_ids)}): duplicati negli split"
    coverage = {
        "n_r2gen_studies": len(annotation),
        "n_emitted": len(seen_ids),
        "n_missing": len(missing),
        "n_dropped_missing_image": n_dropped_missing_image,
        "n_dropped_missing_text": len(missing) - n_dropped_missing_image,
        "n_duplicate_openi_refs": n_duplicate_openi_refs,
        "n_frontal_swapped": n_frontal_swapped,
        "coverage_tolerance": coverage_tolerance,
    }
    return records_by_split, coverage


def build_records_r2gen_annotation_driven(
    all_reports: list[dict[str, Any]],
    *,
    split_cfg: Mapping[str, Any],
    project_root: Path,
    output_dir: Path,
    views_mode: str,
    target_lang: str,
    source_lang: str,
    backend: str,
    system_prompt: str,
    user_prompt: str,
    coverage_tolerance: float = 0.0,
    frontal_selection: str = "order",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    annotation_path = _resolve(project_root, split_cfg["annotation"])
    r2gen_images_dir = _resolve(project_root, split_cfg["images_dir"])
    annotation = load_r2gen_annotation(annotation_path)

    openi_by_study: dict[str, Mapping[str, Any]] = {}
    for report in all_reports:
        for image in report["images"]:
            openi_by_study.setdefault(r2gen_study_id(image["filename"]), report)

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    n_dropped_missing_image = 0
    n_frontal_swapped = 0

    for study_id, entry in sorted(annotation.items()):
        srcs, rel_names, selected, swapped = _r2gen_image_pair(
            r2gen_images_dir, study_id, views_mode, frontal_selection
        )
        if not all(src.exists() for src in srcs):
            n_dropped_missing_image += 1
            continue
        n_frontal_swapped += int(swapped)
        rel_images = _materialize_images(srcs, rel_names, output_dir)
        openi = openi_by_study.get(study_id)
        synthetic_report = {
            "uid": study_id,
            "categories": list(openi["categories"]) if openi else [UNLABELED_LABEL],
            "mesh_raw": list(openi.get("mesh_raw", [])) if openi else [],
        }
        record = build_record(
            synthetic_report,
            selected,
            rel_images,
            entry["report"],
            system_prompt,
            user_prompt,
            source_lang,
            target_lang,
            backend,
        )
        record["id"] = study_id
        record["r2gen_report"] = entry["report"]
        records_by_split[entry["split"]].append(record)

    _coverage_guard(
        n_dropped_missing_image,
        len(annotation),
        coverage_tolerance,
        f"annotation-driven, views={views_mode}, immagini R2Gen mancanti",
    )
    coverage = {
        "n_r2gen_studies": len(annotation),
        "n_emitted": len(annotation) - n_dropped_missing_image,
        "n_missing": n_dropped_missing_image,
        "n_dropped_missing_image": n_dropped_missing_image,
        "n_dropped_missing_text": 0,
        "n_frontal_swapped": n_frontal_swapped,
        "coverage_tolerance": coverage_tolerance,
    }
    return records_by_split, coverage


def run_preprocessing(
    config: Mapping[str, Any],
    project_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import yaml

    from ..cards import write_dataset_card
    from ..tracking import dvc_dataset_hash, git_metadata

    project_root = Path(project_root)
    ds_name = config.get("dataset", "iu-xray")
    version = config["version"]
    source = config["source"]
    reports_dir = _resolve(project_root, source["reports_dir"])

    if output_dir is None:
        output_dir = (
            _resolve(project_root, config.get("output_dir", f"dataset/{ds_name}"))
            / version
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = config.get("fields", {})
    findings_label = fields.get("findings_label", "FINDINGS")
    impression_label = fields.get("impression_label", "IMPRESSION")

    report_cfg = config.get("report", {})
    target = report_cfg.get("target", "findings")
    views_mode = report_cfg.get("views", "frontal_lateral")
    frontal_selection = report_cfg.get("frontal_selection", "order")
    if target not in {"findings", "findings_impression"}:
        raise ValueError(
            f"report.target non valido: {target!r} (usa 'findings' o 'findings_impression')"
        )
    if views_mode not in {"frontal", "frontal_lateral"}:
        raise ValueError(
            f"report.views non valido: {views_mode!r} (usa 'frontal' o 'frontal_lateral')"
        )
    if frontal_selection not in {"order", "symmetry"}:
        raise ValueError(
            f"report.frontal_selection non valido: {frontal_selection!r} (usa 'order' o 'symmetry')"
        )
    if views_mode == "frontal" and frontal_selection == "order":
        print(
            "[preprocess] ⚠️ views=frontal + frontal_selection=order: 0.png assunta frontale, "
            "ordinamento R2Gen NON garantito (~15% invertito, cfr. view_ordering_diagnostic.py). "
            "Rischio misalignment vista↔testo: usa report.frontal_selection: 'symmetry'."
        )
    elif views_mode == "frontal":
        print(
            "[preprocess] views=frontal + frontal_selection=symmetry: la frontale è scelta "
            "per-studio col proxy di simmetria (n_frontal_swapped nel manifest)."
        )

    translation = config.get("translation", {})
    backend = translation.get("backend", "passthrough")
    source_lang = translation.get("source_lang", "en")
    target_lang = translation.get("target_lang", "en")
    translate = get_translator(backend, source_lang, target_lang)

    split_cfg = config.get("split", {})
    coverage_tolerance = float(split_cfg.get("coverage_tolerance", 0.0))
    target_source = split_cfg.get("target_source", "openi")
    if target_source not in {"openi", "r2gen"}:
        raise ValueError(
            f"split.target_source non valido: {target_source!r} (usa 'openi' o 'r2gen')"
        )
    if target_source == "r2gen" and target != "findings":
        raise ValueError(
            f"Config incoerente: split.target_source='r2gen' usa il testo R2Gen (findings-based) "
            f"e IGNORA report.target={target!r}. Con 'r2gen' imposta report.target='findings' "
            f"(o rimuovilo); per un target con impression usa split.target_source='openi'."
        )
    require_text = config.get("filter", {}).get("require_report_text", True)

    prompt_cfg = config.get("prompt", {}) or {}
    system_prompt = prompt_cfg.get("system") or default_system_prompt(
        target_lang, target, target_source
    )
    user_prompt = prompt_cfg.get("user") or default_user_prompt(
        target_lang, target, views_mode, target_source
    )

    all_reports: list[dict[str, Any]] = []
    for xml_path in sorted(reports_dir.glob("*.xml")):
        report = parse_openi_report(xml_path, findings_label, impression_label)
        if report is None:
            continue
        report["findings_tr"] = translate(report["findings"])
        report["impression_tr"] = translate(report["impression"])
        report["mesh_raw"] = report["categories"]
        report["categories"] = map_mesh_to_chexpert(report["mesh_raw"])
        all_reports.append(report)

    if not all_reports:
        raise RuntimeError(f"Nessun report valido trovato in {reports_dir}")

    if target_source == "r2gen":
        is_standard = views_mode == "frontal_lateral"
        reports = all_reports
        records_by_split, coverage = build_records_r2gen_annotation_driven(
            all_reports,
            split_cfg=split_cfg,
            project_root=project_root,
            output_dir=output_dir,
            views_mode=views_mode,
            target_lang=target_lang,
            source_lang=source_lang,
            backend=backend,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            coverage_tolerance=coverage_tolerance,
            frontal_selection=frontal_selection,
        )
    else:
        is_standard = views_mode == "frontal_lateral" and target == "findings"
        reports = [
            report
            for report in all_reports
            if (not require_text)
            or (target == "findings" and report["findings"])
            or (
                target == "findings_impression"
                and report["findings"]
                and report["impression"]
            )
        ]
        if not reports:
            raise RuntimeError(
                f"Nessun report con il testo richiesto (target={target!r}, "
                f"require_report_text={require_text}) in {reports_dir}"
            )
        records_by_split, coverage = build_records_r2gen(
            reports,
            split_cfg=split_cfg,
            project_root=project_root,
            output_dir=output_dir,
            views_mode=views_mode,
            target=target,
            target_lang=target_lang,
            source_lang=source_lang,
            backend=backend,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enforce_full_coverage=is_standard,
            attach_r2gen_reference=is_standard,
            coverage_tolerance=coverage_tolerance,
            frontal_selection=frontal_selection,
        )
    n_dropped_missing_image = coverage["n_dropped_missing_image"]
    if not is_standard:
        print(
            f"[preprocess] ⚠️ variante non-standard (views={views_mode}, target={target}, "
            f"target_source={target_source}): NON confrontabile con la letteratura."
        )

    for split_name in ("val", "test"):
        if not records_by_split[split_name]:
            raise RuntimeError(
                f"Split '{split_name}' vuoto dopo filtri/scarti: "
                f"rivedi annotation R2Gen, report.target o filtri."
            )

    for split, records in records_by_split.items():
        _write_jsonl(output_dir / f"{split}.jsonl", records)

    stats = compute_stats(records_by_split)
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cat_coverage = category_coverage_by_split(records_by_split)
    cat_absent = categories_absent_from_split(cat_coverage)
    if cat_absent:
        print(
            "[preprocess] ⚠️ alcune categorie CheXpert non compaiono in tutti gli split "
            "(lo split R2Gen non stratifica — atteso; cfr. docs/GUIDA_FINE_TUNING.md §B.4):"
        )
        for category, missing_splits in cat_absent.items():
            print(f"    - {category}: assente da {', '.join(missing_splits)}")

    unmatched: dict[str, int] = defaultdict(int)
    n_other = 0
    for report in reports:
        if report["categories"] == [OTHER_LABEL]:
            n_other += 1
            for tag in report["mesh_raw"]:
                unmatched[tag] += 1
    top_unmatched = dict(
        sorted(unmatched.items(), key=lambda kv: kv[1], reverse=True)[:25]
    )

    manifest = {
        "dataset": ds_name,
        "version": version,
        "preprocessing": {
            "type": "preprocess_v2",
            "script_git_sha": git_metadata(project_root).get("git.commit"),
            "target": target,
            "target_source": target_source,
            "target_text": "r2gen_report" if target_source == "r2gen" else "openi",
            "views": views_mode,
            "frontal_selection": frontal_selection,
            "n_frontal_swapped": coverage.get("n_frontal_swapped", 0),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation_method": backend,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_resize": "native",
            "n_dropped_missing_image": n_dropped_missing_image,
        },
        "categories": {
            "scheme": "chexpert14+other+unlabeled",
            "n_reports_other": n_other,
            "top_unmatched_mesh": top_unmatched,
            "coverage_by_split": cat_coverage,
            "absent_from_split": cat_absent,
        },
        "source": {
            "raw_path": str(source["reports_dir"]),
            "raw_dvc_hash": dvc_dataset_hash(
                config.get("output_dir", f"dataset/{ds_name}") + "/original",
                project_root,
            )
            or "unavailable",
        },
        "split": {
            "strategy": "r2gen_official",
            "build": (
                "annotation_driven" if target_source == "r2gen" else "openi_driven"
            ),
            "annotation": str(split_cfg["annotation"]),
            "reference": "Chen et al., 2020 (R2Gen)",
            "literature_comparable": is_standard,
            "variant": f"views={views_mode}, target={target}, target_source={target_source}",
            "n_r2gen_studies": coverage["n_r2gen_studies"],
            "n_emitted": coverage["n_emitted"],
            "n_missing": coverage["n_missing"],
            "n_dropped_missing_text": coverage["n_dropped_missing_text"],
            "n_duplicate_openi_refs": coverage.get("n_duplicate_openi_refs", 0),
            "coverage_tolerance": coverage_tolerance,
        },
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_dataset_card(manifest, stats, output_dir / "dataset_card.md")

    return {
        "output_dir": output_dir,
        "n_reports": len(reports),
        "n_examples": stats["n_examples"],
        "categories": len(stats["diagnostic_category"]),
        "n_dropped_missing_image": n_dropped_missing_image,
    }
