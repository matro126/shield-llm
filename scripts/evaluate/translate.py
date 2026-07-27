#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL_ID = "qwen/qwen3-235b-a22b-2507"
BASE_URL = "https://openrouter.ai/api/v1"

NON_RECOVERABLE_CODES = ["401", "402", "403", "404"]

SYSTEM_PROMPT = (
    "You are a professional medical translator specialized in radiology. Your task "
    "is to translate chest radiology reports from Italian into English. This is not "
    "a word-for-word translation: you must adapt sentence structure and phrasing to "
    "match standard English radiology report conventions, while preserving the exact "
    "clinical meaning. Use strictly standard English radiology report language as "
    "used in clinical PACS systems, with impersonal and formal constructions, and "
    "avoid conversational forms. Do not add, remove, or interpret information. "
    "Maintain a neutral, objective, and technical radiological report style. Keep "
    "abbreviations unchanged unless a standard English equivalent is commonly used, "
    "and apply it consistently. Do not summarize, simplify, explain, or comment on "
    "the content. Output only the translated report text in English."
)

_thread_local = threading.local()


def text_key(text: str) -> str:
    return hashlib.md5(" ".join(text.split()).encode("utf-8")).hexdigest()


def _client():
    if not hasattr(_thread_local, "client"):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY non impostata. Esporta la chiave prima di "
                "eseguire: export OPENROUTER_API_KEY=..."
            )
        _thread_local.client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _thread_local.client


def translate_text(text: str, max_retries: int = 30, initial_wait: float = 5.0) -> str:
    if not text or not isinstance(text, str) or not text.strip():
        return ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text.strip()},
    ]
    client = _client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            error = str(exc)
            if any(code in error for code in NON_RECOVERABLE_CODES):
                print(f"  ❌ errore non recuperabile: {error[:160]}")
                return ""
            wait = initial_wait * (2 ** min(attempt, 6))
            if attempt < max_retries - 1:
                print(f"  ⚠️  tentativo {attempt + 1}/{max_retries}: {error[:120]}")
                print(f"  ⏳ attesa {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"  ❌ falliti {max_retries} tentativi: {text[:70]}…")
    return ""


class TranslationCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  
                    self.entries[row["key"]] = row["translation"]

    def get(self, text: str) -> str | None:
        return self.entries.get(text_key(text))

    def put(self, text: str, translation: str) -> None:
        key = text_key(text)
        with self._lock:
            if key in self.entries:
                return
            self.entries[key] = translation
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"key": key, "source": text, "translation": translation},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def translate_many(
    texts: Sequence[str],
    cache: TranslationCache | None = None,
    workers: int = 8,
    progress: bool = True,
) -> tuple[list[str], dict[str, int]]:
    unique: dict[str, str | None] = {}
    for text in texts:
        if text and text.strip():
            unique.setdefault(text, None)

    stats = {"totali": len(texts), "unici": len(unique), "da_cache": 0, "tradotti": 0}

    todo = []
    for text in unique:
        cached = cache.get(text) if cache else None
        if cached is not None:
            unique[text] = cached
            stats["da_cache"] += 1
        else:
            todo.append(text)

    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(translate_text, text): text for text in todo}
            for future in as_completed(futures):
                text = futures[future]
                translation = future.result()
                unique[text] = translation
                if cache is not None and translation:
                    cache.put(text, translation)
                done += 1
                stats["tradotti"] += 1
                if progress and (done % 25 == 0 or done == len(todo)):
                    print(f"    tradotti {done}/{len(todo)}", flush=True)

    return [unique.get(t, "") if t and t.strip() else "" for t in texts], stats


def cache_stats(texts: Iterable[str], cache: TranslationCache) -> dict[str, int]:
    items = [t for t in texts if t and t.strip()]
    unique = {text_key(t) for t in items}
    missing = {k for k in unique if k not in cache.entries}
    return {
        "totali": len(items),
        "unici": len(unique),
        "gia_in_cache": len(unique) - len(missing),
        "da_tradurre": len(missing),
    }
