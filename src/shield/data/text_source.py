from __future__ import annotations

import csv
from pathlib import Path

from .openi import _NORMAL_TOKENS

_FINDINGS = {"en": "findings", "it": "findings_it"}
_IMPRESSION = {"en": "impression", "it": "impression_it"}


def csv_uid(sample_id: str) -> str:
    return sample_id.split("_", 1)[0].replace("CXR", "")


class TextSource:
    def __init__(self, csv_path: str | Path) -> None:
        with Path(csv_path).open(encoding="utf-8") as handle:
            self.rows = {row["uid"]: row for row in csv.DictReader(handle)}

    def _row(self, sample_id: str) -> dict[str, str]:
        uid = csv_uid(sample_id)
        try:
            return self.rows[uid]
        except KeyError:
            raise KeyError(
                f"Studio {sample_id!r} (uid {uid!r}) assente dal CSV di traduzione."
            ) from None

    def report(self, sample_id: str, lang: str) -> tuple[str, str]:
        row = self._row(sample_id)
        return row[_FINDINGS[lang]].strip(), row[_IMPRESSION[lang]].strip()

    def has_impression(self, sample_id: str, lang: str = "en") -> bool:
        return bool(self._row(sample_id)[_IMPRESSION[lang]].strip())

    def mesh_majors(self, sample_id: str) -> list[str]:
        raw = self._row(sample_id)["MeSH"]
        majors = [
            part.strip().lower()
            for part in raw.split(";")
            if part.strip() and part.strip().lower() not in _NORMAL_TOKENS
        ]
        return majors or ["normal"]
