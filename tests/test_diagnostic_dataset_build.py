from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import shield.data.build_diagnostic as diagnostic_build
import shield.tracking as shield_tracking


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-image")


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def sample(sample_id: str, labels: list[str]) -> dict:
    return {
        "id": sample_id,
        "report": f"Report {sample_id}.",
        "impression": f"Impression {sample_id}.",
        "image_path": [
            f"{sample_id}/frontal.png",
            f"{sample_id}/lateral.png",
        ],
        "split": "train",
        "diagnostic_label": labels,
        "diagnostic_category": ["Legacy"],
        "mesh_raw": ["legacy mesh"],
    }


def prepare_source(root: Path) -> tuple[Path, dict]:
    source = root / "source"
    annotation = {
        "train": [
            sample("healthy", ["No Finding"]),
            sample("mixed", ["Atelectasis", "Other"]),
        ],
        "val": [sample("cardio", ["Cardiomegaly"])],
        "test": [sample("other", ["Other"])],
    }
    source.mkdir(parents=True)
    (source / "annotation_labeled.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    for records in annotation.values():
        for record in records:
            for relative in record["image_path"]:
                write_image(source / "images" / relative)
    return source, annotation


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_other_uses_diagnostic_labels_and_omits_mesh_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = prepare_source(tmp_path)
    monkeypatch.setattr(diagnostic_build, "normalize_image", copy_image)
    monkeypatch.setattr(
        shield_tracking,
        "git_metadata",
        lambda root: {
            "git.commit": "abc123",
            "git.branch": "main",
            "git.is_dirty": False,
            "git.changed_files": 0,
            "git.dirty_files": [],
            "git.dirty_truncated": False,
        },
    )

    result = diagnostic_build.main(
        [
            "--root",
            str(tmp_path),
            "--source",
            str(source),
            "--out",
            "generated",
            "--variant",
            "other",
            "--version",
            "iu_xray_en_FL-FI",
        ]
    )

    assert result == 0
    output = tmp_path / "generated" / "other" / "iu_xray_en_FL-FI"
    records = read_jsonl(output / "train.jsonl")
    assert [record["id"] for record in records] == ["healthy", "mixed"]
    assert records[0]["factors"]["diagnostic_category"] == ["No Finding"]
    assert records[1]["factors"]["diagnostic_category"] == ["Atelectasis", "Other"]
    assert all("mesh_raw" not in record for record in records)
    assert records[1]["images"] == [
        "images_normalized/mixed/frontal.png",
        "images_normalized/mixed/lateral.png",
    ]
    assert records[1]["messages"][2]["content"] == (
        "Findings:\nReport mixed.\n<SEP>\nImpression:\nImpression mixed."
    )
    manifest = yaml.safe_load(
        (output / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert "annotation_labeled.json" in manifest["split_source"]
    assert manifest["label_policy"] == "other"
    assert manifest["provenance"] == {
        "git.commit": "abc123",
        "git.branch": "main",
        "git.is_dirty": False,
        "git.changed_files": 0,
        "git.dirty_files": [],
        "git.dirty_truncated": False,
    }


def test_no_other_removes_every_sample_containing_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, annotation = prepare_source(tmp_path)
    monkeypatch.setattr(diagnostic_build, "normalize_image", copy_image)

    result = diagnostic_build.main(
        [
            "--root",
            str(tmp_path),
            "--source",
            str(source),
            "--out",
            "generated",
            "--variant",
            "no_other",
            "--version",
            "iu_xray_en_F-F",
        ]
    )

    assert result == 0
    output = tmp_path / "generated" / "no_other" / "iu_xray_en_F-F"
    assert [record["id"] for record in read_jsonl(output / "train.jsonl")] == [
        "healthy"
    ]
    assert [record["id"] for record in read_jsonl(output / "val.jsonl")] == [
        "cardio"
    ]
    assert read_jsonl(output / "test.jsonl") == []
    record = read_jsonl(output / "train.jsonl")[0]
    assert record["images"] == ["images_normalized/healthy/frontal.png"]
    assert record["messages"][2]["content"] == "Findings:\nReport healthy."
    assert len(annotation["train"]) == 2
    assert (source / "images" / "mixed" / "frontal.png").is_file()


def test_unknown_diagnostic_label_stops_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = prepare_source(tmp_path)
    monkeypatch.setattr(diagnostic_build, "normalize_image", copy_image)
    annotation_path = source / "annotation_labeled.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["val"][0]["diagnostic_label"] = ["Unknown"]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    try:
        diagnostic_build.main(
            [
                "--root",
                str(tmp_path),
                "--source",
                str(source),
                "--out",
                "generated",
                "--variant",
                "other",
                "--version",
                "iu_xray_en_F-F",
            ]
        )
    except ValueError as exc:
        assert "Unknown" in str(exc)
        assert "cardio" in str(exc)
    else:
        raise AssertionError("unknown diagnostic label accepted")

    assert not (tmp_path / "generated").exists()
