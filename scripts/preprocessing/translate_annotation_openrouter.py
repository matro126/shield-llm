from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.labeling.label_annotation_anthropic import (
    OpenRouterBatchClient,
    atomic_write_json,
    resolved_model,
    sha256_file,
    wait_for_batch,
    write_raw_results,
)

SYSTEM_PROMPT = """You are a professional medical translator specialized in radiology. Your task is to translate chest radiology reports from English into Italian. This is not a word-for-word translation: you must adapt sentence structure and phrasing to match standard Italian radiology report conventions, while preserving the exact clinical meaning. Use strictly standard Italian radiology report language as used in clinical PACS systems, with impersonal and formal constructions, and avoid conversational forms. Do not add, remove, or interpret information. Maintain a neutral, objective, and technical radiological report style. Keep abbreviations unchanged unless a standard Italian equivalent is commonly used, and apply it consistently. Do not summarize, simplify, explain, or comment on the content. Translate report and impression independently. If an input field is empty, return an empty string for its translation. Return only the requested structured JSON object."""


class TranslationSample(NamedTuple):
    split: str
    index: int
    sample_id: str
    report: str
    impression: str


def load_annotation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation root must be a JSON object")
    return payload


def iter_samples(
    annotation: Mapping[str, Any], overwrite: bool
) -> tuple[TranslationSample, ...]:
    samples: list[TranslationSample] = []
    seen: set[str] = set()
    for split, records in annotation.items():
        if not isinstance(records, list):
            raise ValueError(f"split {split!r} must be a list")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"non-object sample in split {split} at index {index}")
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"invalid sample id in split {split} at index {index}")
            if sample_id in seen:
                raise ValueError(f"duplicate sample id: {sample_id}")
            if not overwrite and ("report_it" in record or "impression_it" in record):
                raise ValueError(f"translation already exists for {sample_id}")
            report = record.get("report", "")
            impression = record.get("impression", "")
            if not isinstance(report, str) or not isinstance(impression, str):
                raise ValueError(
                    f"report and impression must be strings for {sample_id}"
                )
            if not report.strip() and not impression.strip():
                raise ValueError(
                    f"report and impression are both empty for {sample_id}"
                )
            seen.add(sample_id)
            samples.append(
                TranslationSample(str(split), index, sample_id, report, impression)
            )
    if not samples:
        raise ValueError("annotation contains no samples")
    return tuple(samples)


def select_samples(
    samples: Sequence[TranslationSample], limit: int | None
) -> tuple[TranslationSample, ...]:
    if limit is None:
        return tuple(samples)
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if limit > len(samples):
        raise ValueError(f"limit {limit} exceeds available samples {len(samples)}")
    return tuple(samples[:limit])


def build_request(
    sample: TranslationSample, model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "report_it": {"type": "string"},
            "impression_it": {"type": "string"},
        },
        "required": ["report_it", "impression_it"],
        "additionalProperties": False,
    }
    return {
        "custom_id": sample.sample_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "report": sample.report,
                            "impression": sample.impression,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        },
    }


def message_payload(
    row: Mapping[str, Any],
) -> tuple[dict[str, str], Mapping[str, Any]]:
    result = row.get("result")
    if not isinstance(result, dict) or result.get("type") != "succeeded":
        result_type = result.get("type") if isinstance(result, dict) else "invalid"
        raise ValueError(f"batch request did not succeed: {result_type}")
    message = result.get("message")
    if not isinstance(message, dict):
        raise ValueError("successful result has no message")
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("message content must be a list")
    texts = [
        part.get("text")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("message must contain exactly one text block")
    payload = json.loads(texts[0])
    expected = {"report_it", "impression_it"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("message JSON must contain report_it and impression_it")
    if not all(isinstance(payload[field], str) for field in expected):
        raise ValueError("translated fields must be strings")
    usage = message.get("usage")
    return (
        {field: payload[field].strip() for field in ("report_it", "impression_it")},
        usage if isinstance(usage, dict) else {},
    )


def parse_batch_results(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise ValueError("expected IDs are not unique")
    parsed: dict[str, dict[str, str]] = {}
    input_tokens = 0
    output_tokens = 0
    for row in rows:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError("result has invalid custom_id")
        if custom_id not in expected:
            raise ValueError(f"unknown result id: {custom_id}")
        if custom_id in parsed:
            raise ValueError(f"duplicate result id: {custom_id}")
        payload, usage = message_payload(row)
        parsed[custom_id] = payload
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
    missing = [sample_id for sample_id in expected_ids if sample_id not in parsed]
    if missing:
        raise ValueError(f"missing result id: {missing[0]}")
    return parsed, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def validate_translation(
    sample_id: str,
    source: Mapping[str, Any],
    translation: Mapping[str, Any],
) -> dict[str, str]:
    expected = {"report_it", "impression_it"}
    if set(translation) != expected:
        raise ValueError(f"{sample_id}: translation fields are invalid")
    normalized: dict[str, str] = {}
    for source_field, target_field in (
        ("report", "report_it"),
        ("impression", "impression_it"),
    ):
        value = translation[target_field]
        if not isinstance(value, str):
            raise ValueError(f"{sample_id}: {target_field} must be a string")
        clean = value.strip()
        source_value = str(source.get(source_field, "")).strip()
        if source_value and not clean:
            raise ValueError(f"{sample_id}: {target_field} is empty")
        if not source_value and clean:
            raise ValueError(
                f"{sample_id}: {target_field} adds content to empty source"
            )
        normalized[target_field] = clean
    return normalized


def apply_translations(
    annotation: Mapping[str, Any],
    translations_by_id: Mapping[str, Mapping[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    samples = iter_samples(annotation, overwrite=overwrite)
    by_id = {sample.sample_id: sample for sample in samples}
    supplied = set(translations_by_id)
    if not supplied:
        raise ValueError("translations are empty")
    unknown = sorted(supplied - set(by_id))
    if unknown:
        raise ValueError(f"unknown translation id: {unknown[0]}")
    result = copy.deepcopy(dict(annotation))
    for records in result.values():
        for record in records:
            sample_id = record["id"]
            if sample_id not in supplied:
                continue
            translated = validate_translation(
                sample_id, record, translations_by_id[sample_id]
            )
            record["report_it"] = translated["report_it"]
            record["impression_it"] = translated["impression_it"]
    return result


def create_state(
    input_path: Path,
    samples: Sequence[TranslationSample],
    batch_id: str,
    model: str,
    effort: str,
    overwrite: bool,
) -> dict[str, Any]:
    if not batch_id:
        raise ValueError("batch id is empty")
    return {
        "schema_version": 1,
        "task": "radiology_translation_en_it",
        "provider": "openrouter",
        "batch_id": batch_id,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "model": model,
        "effort": effort,
        "overwrite": overwrite,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("batch state must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported state schema")
    if payload.get("task") != "radiology_translation_en_it":
        raise ValueError("state belongs to another task")
    if payload.get("provider") != "openrouter":
        raise ValueError("state provider must be openrouter")
    return payload


def validate_state_input(state: Mapping[str, Any], input_path: Path) -> None:
    if state.get("schema_version") != 1:
        raise ValueError("unsupported state schema")
    if state.get("input") != str(input_path.resolve()):
        raise ValueError("state input path differs")
    if state.get("input_sha256") != sha256_file(input_path):
        raise ValueError("input hash differs from submitted batch")


def download_batch(
    input_path: Path,
    state_path: Path,
    output_path: Path,
    raw_results_path: Path,
    error_report_path: Path,
    client: Any,
    replace_output: bool,
) -> dict[str, Any]:
    state = load_state(state_path)
    validate_state_input(state, input_path)
    if output_path.exists() and not replace_output:
        raise FileExistsError(f"output already exists: {output_path}")
    batch_id = str(state.get("batch_id", ""))
    status = client.get_batch(batch_id)
    if status.get("id") != batch_id:
        raise ValueError("batch status ID differs from state")
    phase = client.batch_phase(status)
    if phase == "failed":
        raise RuntimeError(f"batch failed with status: {status.get('status')}")
    if phase != "complete":
        raise RuntimeError(f"batch is not complete: {status.get('status')}")
    rows = client.get_results(status)
    write_raw_results(raw_results_path, rows)
    try:
        translations, usage = parse_batch_results(
            rows, tuple(state.get("sample_ids", ()))
        )
        source = load_annotation(input_path)
        translated = apply_translations(
            source, translations, bool(state.get("overwrite"))
        )
    except Exception as error:
        atomic_write_json(
            error_report_path,
            {
                "schema_version": 1,
                "batch_id": batch_id,
                "error": str(error),
                "raw_results": str(raw_results_path.resolve()),
            },
        )
        raise
    atomic_write_json(output_path, translated)
    return {
        "status": "completed",
        "batch_id": batch_id,
        "sample_count": len(translations),
        "usage": usage,
        "output": str(output_path.resolve()),
        "raw_results": str(raw_results_path.resolve()),
    }


def client_from_args(args: argparse.Namespace) -> OpenRouterBatchClient:
    return OpenRouterBatchClient(
        os.getenv(
            "OPENROUTER_API_KEY",
            "",
        ),
        args.base_url or "https://openrouter.ai",
        timeout=args.http_timeout,
        max_retries=args.max_retries,
    )


def submit_batch(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    if args.state.exists() and not args.replace_state:
        raise FileExistsError(f"state already exists: {args.state}")
    annotation = load_annotation(args.input)
    samples = select_samples(iter_samples(annotation, args.overwrite), args.limit)
    model = resolved_model("openrouter", args.model)
    requests = [
        build_request(sample, model, args.effort, args.max_tokens) for sample in samples
    ]
    response = client.create_batch(requests)
    batch_id = response.get("id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("OpenRouter create response has no batch ID")
    state = create_state(
        args.input,
        samples,
        batch_id,
        model,
        args.effort,
        args.overwrite,
    )
    state["create_response"] = response
    atomic_write_json(args.state, state)
    return state


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url")
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)


def add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("annotation_complete.json"))
    parser.add_argument(
        "--state", type=Path, default=Path("annotation_translation_it.batch.json")
    )
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument(
        "--effort", choices=("low", "medium", "high", "max"), default="low"
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-state", action="store_true")
    add_api_arguments(parser)


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("annotation_complete.json"))
    parser.add_argument(
        "--state", type=Path, default=Path("annotation_translation_it.batch.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("annotation_translated_it.json")
    )
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=Path("annotation_translation_it.batch_results.jsonl"),
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=Path("annotation_translation_it.batch_errors.json"),
    )
    parser.add_argument("--replace-output", action="store_true")
    add_api_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    add_submit_arguments(submit)
    status = subparsers.add_parser("status")
    status.add_argument(
        "--state", type=Path, default=Path("annotation_translation_it.batch.json")
    )
    add_api_arguments(status)
    download = subparsers.add_parser("download")
    add_download_arguments(download)
    run = subparsers.add_parser("run")
    add_submit_arguments(run)
    run.add_argument(
        "--output", type=Path, default=Path("annotation_translated_it.json")
    )
    run.add_argument(
        "--raw-results",
        type=Path,
        default=Path("annotation_translation_it.batch_results.jsonl"),
    )
    run.add_argument(
        "--error-report",
        type=Path,
        default=Path("annotation_translation_it.batch_errors.json"),
    )
    run.add_argument("--replace-output", action="store_true")
    run.add_argument("--poll-interval", type=float, default=60.0)
    run.add_argument("--wait-timeout", type=float, default=86400.0)
    return parser


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "submit":
        return submit_batch(args, client_from_args(args))
    if args.command == "status":
        state = load_state(args.state)
        return client_from_args(args).get_batch(str(state.get("batch_id", "")))
    if args.command == "download":
        return download_batch(
            args.input,
            args.state,
            args.output,
            args.raw_results,
            args.error_report,
            client_from_args(args),
            args.replace_output,
        )
    client = client_from_args(args)
    state = submit_batch(args, client)
    wait_for_batch(
        client,
        state["batch_id"],
        args.poll_interval,
        args.wait_timeout,
        sample_count=int(state["sample_count"]),
    )
    return download_batch(
        args.input,
        args.state,
        args.output,
        args.raw_results,
        args.error_report,
        client,
        args.replace_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as error:
        print(f"errore: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
