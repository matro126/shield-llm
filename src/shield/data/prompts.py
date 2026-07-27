from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

SEP = "<SEP>"

FINDINGS_HEADER = {"en": "Findings:", "it": "Reperti:"}
IMPRESSION_HEADER = {"en": "Impression:", "it": "Impressione:"}


_PERSONA = {
    "en": "You are an expert radiologist. "
          "You will be provided with one or more medical images. ",
    "it": "Sei un radiologo esperto. "
          "Ti verranno fornite una o più immagini mediche. ",
}

_TASK_FINDINGS = {
    "en": "Your task is to concisely describe the findings visible in the image "
          "or images, including anatomical structures, abnormalities, and relevant "
          "observations. ",
    "it": "Il tuo compito è descrivere in modo conciso i reperti visibili "
          "nell'immagine o nelle immagini, incluse le strutture anatomiche, "
          "le anomalie e le osservazioni rilevanti. ",
}

_TASK_FINDINGS_IMPRESSION = {
    "en": "Your task is to: "
          "1) Concisely describe the findings visible in the image or images, "
          "including anatomical structures, abnormalities, and relevant observations. "
          "2) Provide a concise clinical impression that summarizes the main findings. ",
    "it": "Il tuo compito è: "
          "1) Descrivere in modo conciso i reperti visibili nell'immagine o nelle immagini, "
          "incluse le strutture anatomiche, le anomalie e le osservazioni rilevanti. "
          "2) Fornire un'impressione clinica concisa che riassuma i reperti principali. ",
}

_SYS_FMT_FINDINGS = {
    "en": "Provide your answer in EXACTLY this format:\n"
          "Findings:\n<your detailed findings here>\n\n"
          "Do not include any other sections or preamble.",
    "it": "Fornisci la risposta in ESATTAMENTE questo formato:\n"
          "Reperti:\n<i tuoi reperti dettagliati qui>\n\n"
          "NON includere altre sezioni o preamboli.",
}

_SYS_FMT_FINDINGS_IMPRESSION = {
    "en": "Provide your answer in EXACTLY this format:\n"
          "Findings:\n<your detailed findings here>\n"
          f"{SEP}\n"
          "Impression:\n<your concise impression here>\n\n"
          "Do not include any other sections or preamble.",
    "it": "Fornisci la risposta in ESATTAMENTE questo formato:\n"
          "Reperti:\n<i tuoi reperti dettagliati qui>\n"
          f"{SEP}\n"
          "Impressione:\n<la tua impressione concisa qui>\n\n"
          "NON includere altre sezioni o preamboli.",
}

_USER_INTRO = {
    ("frontal_lateral", "en"): "You are given a frontal and a lateral chest X-ray. ",
    ("frontal", "en"): "You are given a frontal chest X-ray. ",
    ("frontal_lateral", "it"): "Ti vengono fornite una radiografia del torace in "
                               "proiezione frontale e una in proiezione laterale. ",
    ("frontal", "it"): "Ti vengono fornite una radiografia del torace in "
                       "proiezione frontale. ",
}

_USER_FMT_FINDINGS = {
    "en": "Concisely describe the visible findings. Answer EXACTLY in this format:\n"
          "Findings:\n<your findings here>\n"
          "Do not include any other sections or preamble.",
    "it": "Descrivi in modo conciso i reperti visibili. "
          "Rispondi ESATTAMENTE in questo formato:\n"
          "Reperti:\n<i tuoi reperti qui>\n"
          "NON includere altre sezioni o preamboli.",
}

_USER_FMT_FINDINGS_IMPRESSION = {
    "en": "Describe the visible findings and provide a concise clinical impression. "
          "Answer EXACTLY in this format:\n"
          "Findings:\n<your findings here>\n"
          f"{SEP}\n"
          "Impression:\n<your impression here>\n"
          "Do not include any other sections or preamble.",
    "it": "Descrivi i reperti visibili e fornisci un'impressione clinica concisa. "
          "Rispondi ESATTAMENTE in questo formato:\n"
          "Reperti:\n<i tuoi reperti qui>\n"
          f"{SEP}\n"
          "Impressione:\n<la tua impressione qui>\n"
          "NON includere altre sezioni o preamboli.",
}


def system_findings_en() -> str:
    return _PERSONA["en"] + _TASK_FINDINGS["en"] + _SYS_FMT_FINDINGS["en"]

def system_findings_it() -> str:
    return _PERSONA["it"] + _TASK_FINDINGS["it"] + _SYS_FMT_FINDINGS["it"]

def system_findings_impression_en() -> str:
    return _PERSONA["en"] + _TASK_FINDINGS_IMPRESSION["en"] + _SYS_FMT_FINDINGS_IMPRESSION["en"]

def system_findings_impression_it() -> str:
    return _PERSONA["it"] + _TASK_FINDINGS_IMPRESSION["it"] + _SYS_FMT_FINDINGS_IMPRESSION["it"]


def user_frontal_lateral_findings_en() -> str:
    return _USER_INTRO[("frontal_lateral", "en")] + _USER_FMT_FINDINGS["en"]

def user_frontal_findings_en() -> str:
    return _USER_INTRO[("frontal", "en")] + _USER_FMT_FINDINGS["en"]

def user_frontal_lateral_findings_impression_en() -> str:
    return _USER_INTRO[("frontal_lateral", "en")] + _USER_FMT_FINDINGS_IMPRESSION["en"]

def user_frontal_findings_impression_en() -> str:
    return _USER_INTRO[("frontal", "en")] + _USER_FMT_FINDINGS_IMPRESSION["en"]

def user_frontal_lateral_findings_it() -> str:
    return _USER_INTRO[("frontal_lateral", "it")] + _USER_FMT_FINDINGS["it"]

def user_frontal_findings_it() -> str:
    return _USER_INTRO[("frontal", "it")] + _USER_FMT_FINDINGS["it"]

def user_frontal_lateral_findings_impression_it() -> str:
    return _USER_INTRO[("frontal_lateral", "it")] + _USER_FMT_FINDINGS_IMPRESSION["it"]

def user_frontal_findings_impression_it() -> str:
    return _USER_INTRO[("frontal", "it")] + _USER_FMT_FINDINGS_IMPRESSION["it"]


PromptBuilder = Callable[[], str]

PROMPT_BUILDERS: dict[tuple[str, str, str], tuple[PromptBuilder, PromptBuilder]] = {
    ("frontal_lateral", "findings", "en"): (system_findings_en, user_frontal_lateral_findings_en),
    ("frontal", "findings", "en"): (system_findings_en, user_frontal_findings_en),
    ("frontal_lateral", "findings_impression", "en"): (system_findings_impression_en, user_frontal_lateral_findings_impression_en),
    ("frontal", "findings_impression", "en"): (system_findings_impression_en, user_frontal_findings_impression_en),
    ("frontal_lateral", "findings", "it"): (system_findings_it, user_frontal_lateral_findings_it),
    ("frontal", "findings", "it"): (system_findings_it, user_frontal_findings_it),
    ("frontal_lateral", "findings_impression", "it"): (system_findings_impression_it, user_frontal_lateral_findings_impression_it),
    ("frontal", "findings_impression", "it"): (system_findings_impression_it, user_frontal_findings_impression_it),
}


@dataclass(frozen=True)
class PromptPair:
    system: str
    user: str


def get_prompts(views: str, target: str, lang: str = "en") -> PromptPair:
    try:
        build_system, build_user = PROMPT_BUILDERS[(views, target, lang)]
    except KeyError:
        raise ValueError(
            f"Nessun prompt per lo scenario views={views!r}, target={target!r}, "
            f"lang={lang!r}. Scenari: {sorted(PROMPT_BUILDERS)}"
        ) from None
    return PromptPair(system=build_system(), user=build_user())


def format_target(findings: str, impression: str, target: str, lang: str = "en") -> str:
    fh = FINDINGS_HEADER[lang]
    if target == "findings":
        return f"{fh}\n{findings.strip()}"
    ih = IMPRESSION_HEADER[lang]
    return f"{fh}\n{findings.strip()}\n{SEP}\n{ih}\n{impression.strip()}"


_HEADER_RE = re.compile(
    r"^\s*(findings|impression|reperti|impressione)\s*:\s*", re.IGNORECASE
)


def strip_section_header(text: str) -> str:
    return _HEADER_RE.sub("", text.strip(), count=1).strip()


def split_sections(text: str) -> tuple[str, str | None]:
    if SEP in text:
        head, _, tail = text.partition(SEP)
        return strip_section_header(head), strip_section_header(tail)

    match = re.search(r"(impression|impressione)\s*:", text, re.IGNORECASE)
    if match:
        return (
            strip_section_header(text[: match.start()]),
            strip_section_header(text[match.start():]),
        )
    return strip_section_header(text), None
