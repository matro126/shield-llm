from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_tree(root: str | Path, pattern: str = "*") -> str:
    base = Path(root)
    if not base.is_dir():
        return "sha256:absent"
    digest = hashlib.sha256()
    for path in sorted(base.rglob(pattern)):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata(root: str | Path = ".") -> dict[str, str | bool | int | None]:
    cwd = Path(root)
    status = _git(["status", "--short"], cwd) or ""
    righe = [line for line in status.splitlines() if line.strip()]
    return {
        "git.commit": _git(["rev-parse", "HEAD"], cwd),
        "git.branch": _git(["branch", "--show-current"], cwd),
        "git.is_dirty": bool(status),
        "git.changed_files": len(righe),
        "git.dirty_files": [r.strip() for r in righe[:30]],
        "git.dirty_truncated": len(righe) > 30,
    }


def _read_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _out_hash(out: Mapping[str, Any]) -> str | None:
    for key in ("md5", "checksum", "hash"):
        value = out.get(key)
        if isinstance(value, str) and key != "hash":
            return value
    return None


def dvc_dataset_hash(
    dataset_root: str | Path,
    project_root: str | Path = ".",
) -> str | None:
    root = Path(project_root)
    target = Path(dataset_root)

    dvc_file = root / f"{target}.dvc"
    parsed = _read_yaml(dvc_file)
    if isinstance(parsed, Mapping):
        outs = parsed.get("outs", [])
        if outs and isinstance(outs[0], Mapping):
            found = _out_hash(outs[0])
            if found:
                return found

    lock = _read_yaml(root / "dvc.lock")
    if isinstance(lock, Mapping):
        target_name = target.name
        for stage in (lock.get("stages") or {}).values():
            if not isinstance(stage, Mapping):
                continue
            for out in stage.get("outs", []) or []:
                if not isinstance(out, Mapping):
                    continue
                path = str(out.get("path", ""))
                if Path(path).name == target_name or path.rstrip("/").endswith(
                    str(target)
                ):
                    found = _out_hash(out)
                    if found:
                        return found
    return None


def sha256_manifest(
    root: str | Path,
    max_bytes: int = 1 << 20,
    pattern: str = "*",
) -> dict[str, Any]:
    base = Path(root)
    if not base.is_dir():
        return {"hash": "sha256:absent", "files": 0, "bytes": 0}
    digest = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(base.rglob(pattern)):
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("utf-8"))
        digest.update(b"\0")
        if size <= max_bytes:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        files += 1
        total += size
    return {"hash": "sha256:" + digest.hexdigest(), "files": files, "bytes": total}


def _resolve(candidate: str | Path, root: Path) -> Path:
    path = Path(candidate)
    return path if path.is_absolute() else root / path


def model_provenance(
    config: Mapping[str, Any],
    root: str | Path = ".",
) -> dict[str, Any]:
    base = Path(root)
    out: dict[str, Any] = {
        "base_model": config.get("base_model") or "",
        "path": str(config.get("model_path") or ""),
        "load_in_4bit": bool(config.get("load_in_4bit")),
    }
    if not out["path"]:
        return out
    resolved = _resolve(out["path"], base)
    out.update(sha256_manifest(resolved))
    if config.get("hash_base_model_full") and resolved.is_dir():
        out["hash_full"] = sha256_tree(resolved)
    return out


def metric_provenance(
    config: Mapping[str, Any],
    root: str | Path = ".",
) -> dict[str, Any]:
    base = Path(root)
    out: dict[str, Any] = {}

    metrics = set(config.get("eval_metrics") or ()) | set(config.get("test_metrics") or ())
    if "chexbert" not in metrics:
        return out

    translator = config.get("chexbert_translator") or ""
    if config.get("chexbert_translate") and translator:
        resolved = _resolve(translator, base)
        out["translator"] = {
            "name": str(translator),
            "local": resolved.is_dir(),
            "hash": sha256_tree(resolved) if resolved.is_dir() else "not-local",
        }

    for candidate in (
        base / "models" / "others" / "chexbert" / "chexbert.pth",
        base / "models" / "evaluation" / "chexbert" / "chexbert.pth",
        Path.home() / ".cache" / "chexbert" / "chexbert.pth",
    ):
        if candidate.is_file():
            out["chexbert_checkpoint"] = {
                "path": str(candidate),
                "hash": sha256_file(candidate),
            }
            break
    else:
        out["chexbert_checkpoint"] = {"path": None, "hash": "unavailable"}

    glossary = config.get("chexbert_glossary") or ""
    if glossary:
        resolved = _resolve(glossary, base)
        out["glossary"] = {
            "path": str(glossary),
            "hash": sha256_file(resolved) if resolved.is_file() else "absent",
        }
    return out


def dataset_provenance(
    config: Mapping[str, Any],
    root: str | Path = ".",
) -> dict[str, str]:
    dataset = config.get("dataset", {})
    if not isinstance(dataset, Mapping):
        return {}

    tags: dict[str, str] = {}
    version = dataset.get("version")
    if version:
        tags["dataset.version"] = str(version)

    dataset_root = dataset.get("root")
    if dataset_root:
        dataset_hash = dvc_dataset_hash(dataset_root, root)
        tags["dvc.dataset_hash"] = dataset_hash if dataset_hash else "unavailable"
    return tags
