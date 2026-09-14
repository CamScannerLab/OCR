from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from nid_ocr_lab.engines.lmstudio_vision import VISION_PROMPT, load_dotenv, normalize_fields
from nid_ocr_lab.models import OCRResult

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def config_value(name: str, default: str = "") -> str:
    file_env = load_dotenv()
    return os.environ.get(name) or file_env.get(name) or default


def api_key() -> str:
    return config_value("GEMINI_API_KEY", "").strip()


def model_name(selected_model: str | None = None) -> str:
    if selected_model:
        return selected_model.strip().removeprefix("models/") or DEFAULT_MODEL
    return config_value("GEMINI_VISION_MODEL", DEFAULT_MODEL).strip().removeprefix("models/") or DEFAULT_MODEL


def base_url() -> str:
    return config_value("GEMINI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def is_available() -> bool:
    return bool(api_key())


def status() -> dict[str, Any]:
    has_key = bool(api_key())
    models: list[str] = []
    note = None if has_key else "Set GEMINI_API_KEY in .env to enable Gemini Vision."
    if has_key:
        try:
            models = list_models(timeout=2.0)
        except RuntimeError as exc:
            note = str(exc)
    selected = model_name()
    if models and selected not in models:
        selected = preferred_model(models)
    return {
        "available": has_key,
        "base_url": base_url(),
        "model": selected,
        "models": models or [selected],
        "note": note,
    }


def preferred_model(models: list[str]) -> str:
    for candidate in (model_name(), "gemini-3.6-flash", "gemini-3.5-flash"):
        if candidate in models:
            return candidate
    flash_models = [model for model in models if "flash" in model.lower()]
    return flash_models[0] if flash_models else (models[0] if models else DEFAULT_MODEL)


def list_models(timeout: float = 8.0) -> list[str]:
    key = api_key()
    if not key:
        return []
    request = urllib.request.Request(
        f"{base_url()}/models",
        headers={"x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load Gemini model list: {exc}") from exc
    output = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = str(item.get("name") or "").removeprefix("models/")
        if name and "embedding" not in name.lower():
            output.append(name)
    return sorted(set(output))


class GeminiVisionEngine:
    name = "gemini_vision"

    def recognize_fields(
        self,
        image_path: Path,
        preprocessing: str | None = None,
        selected_model: str | None = None,
    ) -> tuple[dict[str, Any], OCRResult]:
        key = api_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env before using Gemini Vision.")

        model = model_name(selected_model)
        started = time.perf_counter()
        response = _generate_content(model=model, image_path=image_path, key=key)
        content = _extract_content(response)
        fields = _parse_field_json(content)
        normalized = normalize_fields(fields)
        latency_ms = (time.perf_counter() - started) * 1000
        full_text = json.dumps(normalized, ensure_ascii=False, indent=2)
        return normalized, OCRResult(
            blocks=[],
            full_text=full_text,
            engine=self.name,
            language="vision",
            preprocessing=preprocessing,
            latency_ms=latency_ms,
            metadata={
                "source_image": str(image_path),
                "provider": "gemini",
                "base_url": base_url(),
                "model": model,
                "raw_content": content,
                "raw_response": response,
            },
        )


def _generate_content(model: str, image_path: Path, key: str) -> dict[str, Any]:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": _mime_type(image_path), "data": _image_base64(image_path)}},
                    {"text": VISION_PROMPT},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    encoded_body = json.dumps(body).encode("utf-8")
    retryable_codes = {429, 500, 502, 503, 504}
    last_error = ""
    for attempt, delay_seconds in enumerate((0.0, 1.0, 2.0), start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        request = urllib.request.Request(
            f"{base_url()}/models/{model}:generateContent",
            data=encoded_body,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"Gemini returned HTTP {exc.code}: {detail[:500]}"
            if exc.code not in retryable_codes or attempt == 3:
                if exc.code == 503:
                    raise RuntimeError(
                        f"{last_error} Gemini is temporarily unavailable. Try again in a minute, or switch to LM Studio Vision/PaddleOCR while the service recovers."
                    ) from exc
                raise RuntimeError(last_error) from exc
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"Gemini vision OCR failed: {exc}"
            if attempt == 3:
                raise RuntimeError(last_error) from exc
    raise RuntimeError(last_error or "Gemini vision OCR failed.")


def _image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _mime_type(image_path: Path) -> str:
    return "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"


def _extract_content(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not parts:
        raise RuntimeError("Gemini returned no content parts.")
    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    if not texts:
        raise RuntimeError("Gemini returned no text content.")
    return "\n".join(texts).strip()


def _parse_field_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        raise RuntimeError(f"Gemini did not return JSON fields: {content[:300]}")
    if end == -1 or end <= start:
        salvaged = _salvage_field_json(text)
        if salvaged:
            return salvaged
        raise RuntimeError(f"Gemini did not return JSON fields: {content[:300]}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        salvaged = _salvage_field_json(text)
        if salvaged:
            return salvaged
        raise RuntimeError(f"Gemini returned invalid JSON: {content[:300]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini field response must be a JSON object.")
    return parsed


def _salvage_field_json(content: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for match in re.finditer(r'"([a-z_]+)"\s*:\s*(null|"(?:\\.|[^"\\])*")', content):
        key = match.group(1)
        raw_value = match.group(2)
        if key not in {
            "nid_number",
            "name_bangla",
            "name_english",
            "father_name_bangla",
            "father_name_english",
            "mother_name_bangla",
            "mother_name_english",
            "date_of_birth",
            "address_bangla",
            "address_english",
        }:
            continue
        if raw_value == "null":
            fields[key] = None
            continue
        try:
            fields[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            continue
    return fields
