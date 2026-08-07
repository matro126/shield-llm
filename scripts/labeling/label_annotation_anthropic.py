from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from tqdm import tqdm

LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "No Finding",
    "Other",
)

SYSTEM_PROMPT = """You classify chest X-ray samples using the textual content of the report and impression fields.

Return a JSON object with exactly one key named \"diagnostic_label\". Its value must be a non-empty JSON list containing one or more of these exact English labels:
Atelectasis
Cardiomegaly
Consolidation
Edema
Enlarged Cardiomediastinum
Fracture
Lung Lesion
Lung Opacity
Pleural Effusion
Pleural Other
Pneumonia
Pneumothorax
Support Devices
No Finding
Other

Classification rules:
1. Read both report and impression before assigning labels.
2. This is a multi-label classification task. Assign all labels directly supported by the radiological findings described in the text.
3. Do not choose only the most important finding. If multiple supported findings belong to different categories, include all corresponding labels.
4. Do not infer diagnoses that are not supported by the text.
5. Do not assign a label when the corresponding condition is explicitly negated. For example, \"No pneumothorax\" does not support Pneumothorax, \"No pleural effusion\" does not support Pleural Effusion, and \"No focal consolidation\" does not support Consolidation.
6. Handle uncertain findings carefully. Statements such as \"possible\", \"may represent\", \"suggestive of\", or \"cannot exclude\" should only generate a label when the finding is clearly presented as a relevant possible abnormality. Do not convert weak or purely hypothetical statements into definite diagnoses.
7. Assign every applicable predefined category.
8. Use Other when there is at least one clinically relevant positive abnormality that cannot reasonably be mapped to a predefined category.
9. Other may coexist with predefined labels when the report contains covered abnormalities and an additional clinically relevant abnormality outside the predefined categories.
10. Use No Finding only when there is no significant positive radiographic abnormality that maps to a predefined category or Other.
11. No Finding must not coexist with any other label.
12. Do not assign Other for normal anatomical descriptions, technical information, or explicitly negated findings. Other requires a genuine positive abnormal finding outside the predefined categories.
13. Do not add duplicate labels.
14. Use the exact label names and capitalization. Do not translate, abbreviate, or create categories.
15. Return only the JSON object and no additional text."""


class SampleRef(NamedTuple):
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
) -> tuple[SampleRef, ...]:
    samples: list[SampleRef] = []
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
            if "diagnostic_label" in record and not overwrite:
                raise ValueError(f"diagnostic_label already exists for {sample_id}")
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
            samples.append(SampleRef(str(split), index, sample_id, report, impression))
    if not samples:
        raise ValueError("annotation contains no samples")
    return tuple(samples)


def select_samples(
    samples: Sequence[SampleRef], limit: int | None
) -> tuple[SampleRef, ...]:
    if limit is None:
        return tuple(samples)
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if limit > len(samples):
        raise ValueError(f"limit {limit} exceeds available samples {len(samples)}")
    return tuple(samples[:limit])


def build_request(
    sample: SampleRef, model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "diagnostic_label": {
                "type": "array",
                "items": {"type": "string", "enum": list(LABELS)},
                "minItems": 1,
            }
        },
        "required": ["diagnostic_label"],
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
                        {"report": sample.report, "impression": sample.impression},
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


def validate_labels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("diagnostic_label must be a non-empty list")
    if not all(isinstance(label, str) for label in value):
        raise ValueError("diagnostic_label values must be strings")
    unknown = [label for label in value if label not in LABELS]
    if unknown:
        raise ValueError(f"unknown label: {unknown[0]}")
    selected = set(value)
    if "No Finding" in selected and len(selected) > 1:
        selected.remove("No Finding")
    return tuple(label for label in LABELS if label in selected)


def _message_payload(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
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
    if not isinstance(payload, dict) or set(payload) != {"diagnostic_label"}:
        raise ValueError("message JSON must contain only diagnostic_label")
    usage = message.get("usage")
    return payload, usage if isinstance(usage, dict) else {}


def parse_batch_results(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise ValueError("expected IDs are not unique")
    parsed: dict[str, tuple[str, ...]] = {}
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
        payload, usage = _message_payload(row)
        parsed[custom_id] = validate_labels(payload["diagnostic_label"])
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
    missing = [sample_id for sample_id in expected_ids if sample_id not in parsed]
    if missing:
        raise ValueError(f"missing result id: {missing[0]}")
    return parsed, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_state(
    input_path: Path,
    samples: Sequence[SampleRef],
    batch_id: str,
    model: str,
    effort: str,
    overwrite: bool,
) -> dict[str, Any]:
    if not batch_id:
        raise ValueError("batch id is empty")
    return {
        "schema_version": 1,
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


def validate_state_input(state: Mapping[str, Any], input_path: Path) -> None:
    if state.get("schema_version") != 1:
        raise ValueError("unsupported state schema")
    recorded_path = state.get("input")
    if recorded_path != str(input_path.resolve()):
        raise ValueError("state input path differs")
    recorded_hash = state.get("input_sha256")
    if recorded_hash != sha256_file(input_path):
        raise ValueError("input hash differs from submitted batch")


def apply_labels(
    annotation: Mapping[str, Any],
    labels_by_id: Mapping[str, Sequence[str]],
    overwrite: bool,
) -> dict[str, Any]:
    samples = iter_samples(annotation, overwrite=overwrite)
    expected = {sample.sample_id for sample in samples}
    supplied = set(labels_by_id)
    unknown = sorted(supplied - expected)
    if not supplied:
        raise ValueError("labels are empty")
    if unknown:
        raise ValueError(f"unknown labels id: {unknown[0]}")
    result = copy.deepcopy(dict(annotation))
    for records in result.values():
        for record in records:
            sample_id = record["id"]
            if sample_id not in supplied:
                continue
            normalized = validate_labels(list(labels_by_id[sample_id]))
            record.pop("diagnosticl_label", None)
            record["diagnostic_label"] = list(normalized)
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _urlopen_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Anthropic request failed: {error.reason}") from error


class AnthropicBatchClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        api_version: str,
        timeout: float = 120.0,
        max_retries: int = 5,
        transport: Any = None,
    ):
        if not api_key:
            raise ValueError("API key is not set")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("invalid HTTP retry configuration")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.transport = transport or _urlopen_transport

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def _request(self, method: str, url: str, payload: Any = None) -> bytes:
        body = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if payload is not None
            else None
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.transport(method, url, self._headers(), body, self.timeout)
            except RuntimeError as error:
                last_error = error
                retryable = any(
                    marker in str(error)
                    for marker in (
                        "HTTP 408",
                        "HTTP 409",
                        "HTTP 429",
                        "HTTP 500",
                        "HTTP 502",
                        "HTTP 503",
                        "HTTP 504",
                        "request failed",
                    )
                )
                if not retryable or attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
        raise RuntimeError("Anthropic request exhausted retries") from last_error

    def _json(self, method: str, url: str, payload: Any = None) -> dict[str, Any]:
        value = json.loads(self._request(method, url, payload).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Anthropic response must be a JSON object")
        return value

    def create_batch(self, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not requests:
            raise ValueError("batch must contain at least one request")
        return self._json(
            "POST", f"{self.base_url}/v1/messages/batches", {"requests": list(requests)}
        )

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            raise ValueError("batch id is empty")
        return self._json("GET", f"{self.base_url}/v1/messages/batches/{batch_id}")

    def download_results(self, results_url: str) -> list[dict[str, Any]]:
        if not results_url.startswith("https://"):
            raise ValueError("results URL must use HTTPS")
        text = self._request("GET", results_url).decode("utf-8")
        rows: list[dict[str, Any]] = []
        for index, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object result at line {index + 1}")
            rows.append(value)
        if not rows:
            raise ValueError("Anthropic results are empty")
        return rows

    def batch_phase(self, status: Mapping[str, Any]) -> str:
        return "complete" if status.get("processing_status") == "ended" else "pending"

    def get_results(self, status: Mapping[str, Any]) -> list[dict[str, Any]]:
        results_url = status.get("results_url")
        if not isinstance(results_url, str) or not results_url:
            raise ValueError("completed batch has no results URL")
        return self.download_results(results_url)


class OpenRouterBatchClient(AnthropicBatchClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 5,
        transport: Any = None,
    ):
        super().__init__(
            api_key,
            base_url,
            "",
            timeout=timeout,
            max_retries=max_retries,
            transport=transport,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_batch(self, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not requests:
            raise ValueError("batch must contain at least one request")
        models = {
            request.get("params", {}).get("model")
            for request in requests
            if isinstance(request.get("params"), Mapping)
        }
        if len(models) != 1 or not all(
            isinstance(model, str) and model for model in models
        ):
            raise ValueError("OpenRouter batch requests must use one model")
        model = next(iter(models))
        converted = [
            {"custom_id": request.get("custom_id"), "body": request.get("params")}
            for request in requests
        ]
        return self._json(
            "POST",
            f"{self.base_url}/api/beta/batches",
            {
                "endpoint": "/v1/messages",
                "model": model,
                "requests": converted,
            },
        )

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            raise ValueError("batch id is empty")
        return self._json("GET", f"{self.base_url}/api/beta/batches/{batch_id}")

    def batch_phase(self, status: Mapping[str, Any]) -> str:
        value = status.get("status")
        if value == "completed":
            return "complete"
        if value in ("failed", "cancelled", "expired"):
            return "failed"
        return "pending"

    def get_results(self, status: Mapping[str, Any]) -> list[dict[str, Any]]:
        results = status.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("completed OpenRouter batch has no inlined results")
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                raise ValueError(f"non-object OpenRouter result at index {index}")
            custom_id = row.get("custom_id")
            if "result" in row:
                normalized.append(dict(row))
                continue
            response = row.get("response")
            error = row.get("error")
            status_code = (
                response.get("status_code") if isinstance(response, dict) else None
            )
            body = response.get("body") if isinstance(response, dict) else None
            if isinstance(response, dict) and body is None and "content" in response:
                body = response
                status_code = 200
            if (
                error is None
                and isinstance(status_code, int)
                and 200 <= status_code < 300
                and isinstance(body, dict)
            ):
                normalized.append(
                    {
                        "custom_id": custom_id,
                        "result": {"type": "succeeded", "message": body},
                    }
                )
            else:
                normalized.append(
                    {
                        "custom_id": custom_id,
                        "result": {
                            "type": "errored",
                            "error": error
                            or {"status_code": status_code, "body": body},
                        },
                    }
                )
        return normalized


def load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("batch state must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported state schema")
    return payload


def write_raw_results(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    value = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, value)


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
    batch_id = state.get("batch_id")
    status = client.get_batch(batch_id)
    if status.get("id") != batch_id:
        raise ValueError("batch status ID differs from state")
    phase = (
        client.batch_phase(status)
        if hasattr(client, "batch_phase")
        else "complete" if status.get("processing_status") == "ended" else "pending"
    )
    if phase == "failed":
        raise RuntimeError(f"batch failed with status: {status.get('status')}")
    if phase != "complete":
        value = status.get("processing_status", status.get("status"))
        raise RuntimeError(f"batch is not complete: {value}")
    rows = (
        client.get_results(status)
        if hasattr(client, "get_results")
        else client.download_results(status.get("results_url"))
    )
    write_raw_results(raw_results_path, rows)
    try:
        labels, usage = parse_batch_results(rows, tuple(state.get("sample_ids", ())))
        annotation = load_annotation(input_path)
        labeled = apply_labels(annotation, labels, bool(state.get("overwrite")))
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
    atomic_write_json(output_path, labeled)
    return {
        "status": "completed",
        "batch_id": batch_id,
        "sample_count": len(labels),
        "usage": usage,
        "output": str(output_path.resolve()),
        "raw_results": str(raw_results_path.resolve()),
    }


def resolved_provider(
    args: argparse.Namespace, state: Mapping[str, Any] | None = None
) -> str:
    requested = getattr(args, "provider", None)
    recorded = state.get("provider") if isinstance(state, Mapping) else None
    provider = requested or recorded or "anthropic"
    if provider not in ("anthropic", "openrouter"):
        raise ValueError(f"unsupported provider: {provider}")
    if requested and recorded and requested != recorded:
        raise ValueError(f"provider {requested} differs from state provider {recorded}")
    return provider


def client_from_args(
    args: argparse.Namespace, state: Mapping[str, Any] | None = None
) -> AnthropicBatchClient:
    provider = resolved_provider(args, state)
    if provider == "openrouter":
        return OpenRouterBatchClient(
            os.getenv(
                "OPENROUTER_API_KEY",
                "",
            ),
            args.base_url or "https://openrouter.ai",
            timeout=args.http_timeout,
            max_retries=args.max_retries,
        )
    return AnthropicBatchClient(
        os.getenv("ANTHROPIC_API_KEY", ""),
        args.base_url or "https://api.anthropic.com",
        args.api_version,
        timeout=args.http_timeout,
        max_retries=args.max_retries,
    )


def resolved_model(provider: str, model: str) -> str:
    if provider == "openrouter":
        if model == "claude-sonnet-5":
            return "anthropic/claude-sonnet-5"
        return model.removesuffix(":batch")
    return model


def submit_batch(args: argparse.Namespace, client: Any) -> dict[str, Any]:
    if args.state.exists() and not args.replace_state:
        raise FileExistsError(f"state already exists: {args.state}")
    annotation = load_annotation(args.input)
    all_samples = iter_samples(annotation, args.overwrite)
    samples = select_samples(all_samples, args.limit)
    provider = resolved_provider(args)
    model = resolved_model(provider, args.model)
    requests = [
        build_request(sample, model, args.effort, args.max_tokens) for sample in samples
    ]
    response = client.create_batch(requests)
    batch_id = response.get("id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("Anthropic create response has no batch ID")
    state = create_state(
        args.input,
        samples,
        batch_id,
        model,
        args.effort,
        args.overwrite,
    )
    state["provider"] = provider
    state["create_response"] = response
    atomic_write_json(args.state, state)
    return state


def wait_for_batch(
    client: Any,
    batch_id: str,
    poll_interval: float,
    wait_timeout: float,
    sample_count: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    progress = tqdm(
        total=sample_count,
        desc=f"batch {batch_id}",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        while True:
            try:
                status = client.get_batch(batch_id)
            except RuntimeError as error:
                transient_not_found = isinstance(
                    client, OpenRouterBatchClient
                ) and "HTTP 404" in str(error)
                if not transient_not_found:
                    raise
                progress.set_postfix(stato="propagazione")
                progress.refresh()
                if time.monotonic() - started >= wait_timeout:
                    raise TimeoutError(
                        f"batch did not become visible within {wait_timeout} seconds"
                    ) from error
                time.sleep(poll_interval)
                continue
            if status.get("id") != batch_id:
                raise ValueError("batch status ID differs from submitted ID")
            phase = (
                client.batch_phase(status)
                if hasattr(client, "batch_phase")
                else (
                    "complete"
                    if status.get("processing_status") == "ended"
                    else "pending"
                )
            )
            counts = status.get("request_counts")
            completed = 0
            failed = 0
            if isinstance(counts, Mapping):
                completed = int(counts.get("completed", counts.get("succeeded", 0)))
                failed = int(counts.get("failed", counts.get("errored", 0)))
                failed += int(counts.get("canceled", 0))
                failed += int(counts.get("expired", 0))
                if progress.total is None and isinstance(counts.get("total"), int):
                    progress.total = counts["total"]
                if progress.total is not None:
                    progress.n = min(completed + failed, progress.total)
            status_name = status.get("status", status.get("processing_status", phase))
            progress.set_postfix(
                stato=status_name,
                completati=completed,
                falliti=failed,
            )
            progress.refresh()
            if phase == "complete":
                return status
            if phase == "failed":
                raise RuntimeError(f"batch failed with status: {status.get('status')}")
            if time.monotonic() - started >= wait_timeout:
                raise TimeoutError(
                    f"batch did not complete within {wait_timeout} seconds"
                )
            time.sleep(poll_interval)
    finally:
        progress.close()


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("anthropic", "openrouter"))
    parser.add_argument("--base-url")
    parser.add_argument("--api-version", default="2023-06-01")
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)


def _add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("annotation_complete.json"))
    parser.add_argument(
        "--state", type=Path, default=Path("annotation_complete.batch.json")
    )
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument(
        "--effort", choices=("low", "medium", "high", "max"), default="medium"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-state", action="store_true")
    _add_api_arguments(parser)


def _add_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("annotation_complete.json"))
    parser.add_argument(
        "--state", type=Path, default=Path("annotation_complete.batch.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("annotation_complete_labeled.json")
    )
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=Path("annotation_complete.batch_results.jsonl"),
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=Path("annotation_complete.batch_errors.json"),
    )
    parser.add_argument("--replace-output", action="store_true")
    _add_api_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    _add_submit_arguments(submit)
    status = subparsers.add_parser("status")
    status.add_argument(
        "--state", type=Path, default=Path("annotation_complete.batch.json")
    )
    _add_api_arguments(status)
    download = subparsers.add_parser("download")
    _add_download_arguments(download)
    run = subparsers.add_parser("run")
    _add_submit_arguments(run)
    run.add_argument(
        "--output", type=Path, default=Path("annotation_complete_labeled.json")
    )
    run.add_argument(
        "--raw-results",
        type=Path,
        default=Path("annotation_complete.batch_results.jsonl"),
    )
    run.add_argument(
        "--error-report",
        type=Path,
        default=Path("annotation_complete.batch_errors.json"),
    )
    run.add_argument("--replace-output", action="store_true")
    run.add_argument("--poll-interval", type=float, default=60.0)
    run.add_argument("--wait-timeout", type=float, default=86400.0)
    return parser


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "submit":
        client = client_from_args(args)
        return submit_batch(args, client)
    if args.command == "status":
        state = load_state(args.state)
        client = client_from_args(args, state)
        return client.get_batch(str(state.get("batch_id", "")))
    if args.command == "download":
        state = load_state(args.state)
        client = client_from_args(args, state)
        return download_batch(
            args.input,
            args.state,
            args.output,
            args.raw_results,
            args.error_report,
            client,
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
