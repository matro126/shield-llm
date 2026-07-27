from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

FINDINGS_LABEL = "FINDINGS"
IMPRESSION_LABEL = "IMPRESSION"

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
    ("Pleural Other", ("pleural thickening",)),
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


def clean(text: str | None) -> str:
    return " ".join(text.split()) if text else ""


PLEURAL_OTHER_LABEL = "Pleural Other"

_PATTERN_KEYWORDS = frozenset({"interstitial", "reticular"})

_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "pneumothorax": ("pneumothorax", "hydropneumothorax", "hemopneumothorax"),
}

_PLEURAL_TERMS = ("pleura", "pleural")


def _mentions(keyword: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(keyword), text) is not None


def chexpert_labels_for_tag(tag: str) -> list[str]:
    lowered = tag.lower()
    finding = lowered.split("/", 1)[0]

    labels: list[str] = []
    for label, keywords in _CHEXPERT_RULES:
        for keyword in keywords:
            scope = lowered if keyword in _PATTERN_KEYWORDS else finding
            if any(
                _mentions(alias, scope)
                for alias in _KEYWORD_ALIASES.get(keyword, (keyword,))
            ):
                labels.append(label)
                break

    if not labels and any(_mentions(term, lowered) for term in _PLEURAL_TERMS):
        labels.append(PLEURAL_OTHER_LABEL)
    return labels


def map_mesh_to_chexpert(mesh_majors: list[str]) -> list[str]:
    found = {label for tag in mesh_majors for label in chexpert_labels_for_tag(tag)}
    if found:
        order = [label for label, _ in _CHEXPERT_RULES]
        return [label for label in order if label in found]
    lowered_tags = [tag.lower() for tag in mesh_majors]
    if not mesh_majors or all(tag in _NORMAL_TOKENS for tag in lowered_tags):
        return [NO_FINDING_LABEL]
    if all(tag in _NORMAL_TOKENS | _UNLABELED_TOKENS for tag in lowered_tags):
        return [UNLABELED_LABEL]
    return [OTHER_LABEL]


@dataclass
class OpenIStudy:
    uid: str
    report_file: str
    findings: str = ""
    impression: str = ""
    caption: str = ""
    parent_images: list[str] = field(default_factory=list)
    mesh_majors: list[str] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def has_impression(self) -> bool:
        return bool(self.impression)

    @property
    def image_filenames(self) -> list[str]:
        return [f"{name}.png" for name in self.parent_images]

    @property
    def chexpert_categories(self) -> list[str]:
        return map_mesh_to_chexpert(self.mesh_majors)


def parse_report(
    xml_path: str | Path,
    findings_label: str = FINDINGS_LABEL,
    impression_label: str = IMPRESSION_LABEL,
) -> OpenIStudy:
    path = Path(xml_path)
    root = ET.parse(str(path)).getroot()

    uid_node = root.find("uId")
    uid = (uid_node.get("id") if uid_node is not None else None) or path.stem

    sections: dict[str, str] = {}
    for node in root.iter("AbstractText"):
        label = (node.get("Label") or "").strip().upper()
        if label:
            sections[label] = clean(node.text)

    mesh_majors = [
        value
        for major in root.findall("MeSH/major")
        if (value := clean(major.text).lower()) and value not in _NORMAL_TOKENS
    ]

    images: list[str] = []
    captions: list[str] = []
    for node in root.iter("parentImage"):
        image_id = node.get("id")
        if image_id:
            images.append(image_id)
        caption = clean(node.findtext("caption"))
        if caption and caption not in captions:
            captions.append(caption)

    return OpenIStudy(
        uid=uid,
        report_file=path.name,
        findings=sections.get(findings_label.upper(), ""),
        impression=sections.get(impression_label.upper(), ""),
        caption=" | ".join(captions),
        parent_images=images,
        mesh_majors=mesh_majors or ["normal"],
    )


def report_paths(reports_dir: str | Path) -> list[Path]:
    return sorted(
        Path(reports_dir).glob("*.xml"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )


def iter_reports(
    reports_dir: str | Path, skip_malformed: bool = True
) -> Iterator[OpenIStudy]:
    for path in report_paths(reports_dir):
        try:
            yield parse_report(path)
        except ET.ParseError:
            if not skip_malformed:
                raise
