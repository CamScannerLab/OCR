from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from nid_ocr_lab.models import FIELD_NAMES, OCRResult

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"

VISION_PROMPT = """Extract Bangladesh National ID card fields from this image.

Return only one valid JSON object. Do not include markdown fences or explanation.
Use these exact keys:
- nid_number
- name_bangla
- name_english
- father_name_bangla
- father_name_english
- mother_name_bangla
- mother_name_english
- date_of_birth
- address_bangla
- address_english

Rules:
- The image can be the front side or the back side of a Bangladesh NID card.
- Read the card body, not only the header.
- If this is the back side, extract the printed address into address_bangla or address_english and keep unavailable front-side fields as null.
- If a field is not visible, use null.
- Keep Bangla fields in Bangla script.
- Keep English names in English script.
- For date_of_birth, preserve the printed value if you cannot confidently convert it.
- For nid_number, return only the ID number digits when visible.
- Keep address values concise and exactly as printed; do not add explanation.
- Always close the JSON object.
"""


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    env_path = path or Path.cwd() / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def config_value(name: str, default: str = "") -> str:
    file_env = load_dotenv()
    return os.environ.get(name) or file_env.get(name) or default


def base_url() -> str:
    return config_value("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def api_key() -> str:
    return config_value("LM_STUDIO_API_KEY", DEFAULT_API_KEY)


def configured_model() -> str:
    return config_value("LM_STUDIO_VISION_MODEL", "").strip()


def is_available(timeout: float = 0.75) -> bool:
    return status(timeout=timeout)["available"]


def status(timeout: float = 0.75) -> dict[str, Any]:
    try:
        models = list_models(timeout=timeout)
    except RuntimeError as exc:
        return {
            "available": False,
            "base_url": base_url(),
            "model": configured_model() or None,
            "models": [],
            "note": str(exc),
        }
    model = configured_model() or (models[0] if models else "")
    return {
        "available": bool(model),
        "base_url": base_url(),
        "model": model or None,
        "models": models,
        "note": "Start LM Studio Local Server and load a vision model." if not model else None,
    }


def list_models(timeout: float = 2.0) -> list[str]:
    request = urllib.request.Request(f"{base_url()}/models", headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio is not reachable at {base_url()}. Start the Local Server in LM Studio.") from exc
    models = payload.get("data") if isinstance(payload, dict) else []
    output = []
    for model in models or []:
        model_id = model.get("id") if isinstance(model, dict) else None
        if model_id:
            output.append(str(model_id))
    return output


class LMStudioVisionEngine:
    name = "lmstudio_vision"

    def recognize_fields(
        self,
        image_path: Path,
        preprocessing: str | None = None,
        selected_model: str | None = None,
    ) -> tuple[dict[str, Any], OCRResult]:
        model = (selected_model or "").strip() or configured_model() or _first_available_model()
        if not model:
            raise RuntimeError(
                "No LM Studio model is available. Start LM Studio Local Server and load Qwen2.5-VL."
            )

        started = time.perf_counter()
        response = _chat_completion(model=model, image_path=image_path)
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
                "provider": "lm_studio",
                "base_url": base_url(),
                "model": model,
                "raw_content": content,
                "raw_response": response,
            },
        )


def normalize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name in FIELD_NAMES:
        value = fields.get(name)
        if value is None:
            normalized[name] = None
        else:
            text = str(value).strip()
            normalized[name] = text or None
    return normalized


def _first_available_model() -> str:
    models = list_models(timeout=2.0)
    return models[0] if models else ""


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }


def _chat_completion(model: str, image_path: Path) -> dict[str, Any]:
    data_url = _image_data_url(image_path)
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 900,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        f"{base_url()}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio returned HTTP {exc.code}: {detail[:500]}") from exc
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio vision OCR failed: {exc}") from exc


def _image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise RuntimeError("LM Studio returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    raise RuntimeError("LM Studio returned an unsupported response format.")


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
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"LM Studio did not return JSON fields: {content[:300]}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LM Studio returned invalid JSON: {content[:300]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("LM Studio field response must be a JSON object.")
    return parsed
