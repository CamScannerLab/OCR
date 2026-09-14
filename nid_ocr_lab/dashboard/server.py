from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import subprocess
import tempfile
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

from nid_ocr_lab.dashboard.filters import (
    FILTER_MODES,
    render_image,
    render_mask_overlay,
    render_sdk_crop,
    save_filtered_image,
    save_sdk_crop,
)
from nid_ocr_lab.dashboard.indexer import build_index, dataset_health, load_annotation
from nid_ocr_lab.engines.gemini_vision import GeminiVisionEngine, status as gemini_status
from nid_ocr_lab.engines.lmstudio_vision import LMStudioVisionEngine, status as lmstudio_status
from nid_ocr_lab.engines.paddleocr import PaddleOCREngine, is_available as paddleocr_available
from nid_ocr_lab.engines.tesseract import TesseractEngine, ocr_result_to_json
from nid_ocr_lab.models import FIELD_NAMES, FieldResult, NIDData, OCRResult
from nid_ocr_lab.pipeline import OCRPipeline

STATIC_DIR = Path(__file__).with_name("static")


class DashboardHandler(BaseHTTPRequestHandler):
    index_cache: list[dict] | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static("index.html")
        elif parsed.path == "/api/samples":
            self.send_json(self.samples())
        elif parsed.path == "/api/health":
            self.send_json(dataset_health())
        elif parsed.path == "/api/ocr/status":
            self.send_json(ocr_status())
        elif parsed.path == "/api/annotation":
            query = parse_qs(parsed.query)
            self.send_json(load_annotation(first(query, "path")) or {})
        elif parsed.path == "/image":
            self.send_image(parsed.query)
        elif parsed.path == "/mask-overlay":
            self.send_mask_overlay(parsed.query)
        elif parsed.path == "/sdk-crop":
            self.send_sdk_crop(parsed.query)
        elif parsed.path.startswith("/static/"):
            self.send_static(parsed.path.removeprefix("/static/"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/ocr/run":
            self.run_ocr()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def samples(self) -> list[dict]:
        if DashboardHandler.index_cache is None:
            DashboardHandler.index_cache = build_index()
        return DashboardHandler.index_cache

    def send_image(self, query_string: str) -> None:
        query = parse_qs(query_string)
        path = first(query, "path")
        mode = first(query, "mode") or "original"
        if not path or mode not in FILTER_MODES:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid image request")
            return
        try:
            payload, _info = render_image(unquote(path), mode=mode)
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def send_mask_overlay(self, query_string: str) -> None:
        query = parse_qs(query_string)
        image_path = first(query, "image")
        mask_path = first(query, "mask")
        if not image_path or not mask_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid mask overlay request")
            return
        try:
            payload, _info = render_mask_overlay(unquote(image_path), unquote(mask_path))
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Image or mask not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def send_sdk_crop(self, query_string: str) -> None:
        query = parse_qs(query_string)
        image_path = first(query, "image")
        annotation_path = first(query, "annotation")
        mode = first(query, "mode") or "original"
        if not image_path or not annotation_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid SDK crop request")
            return
        annotation = load_annotation(unquote(annotation_path)) or {}
        points = first_quad_points(annotation)
        if not points:
            self.send_error(HTTPStatus.BAD_REQUEST, "No 4-point annotation quad found")
            return
        try:
            payload, _info = render_sdk_crop(unquote(image_path), points, mode=mode)
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND, "SDK crop source not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def send_json(self, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def run_ocr(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = run_ocr_payload(payload)
        except RuntimeError as exc:
            self.send_json({"ok": False, "error": str(exc)})
            return
        except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            self.send_json({"ok": False, "error": f"OCR failed: {exc}"})
            return
        self.send_json({"ok": True, **result})

    def send_static(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def first_quad_points(annotation: dict) -> list[list[float]] | None:
    for shape in annotation.get("shapes", []):
        points = shape.get("points") or []
        if len(points) == 4:
            return points
    return None


def ocr_status() -> dict:
    tesseract_path = shutil.which("tesseract")
    lmstudio = lmstudio_status()
    gemini = gemini_status()
    return {
        "engines": [
            {
                "id": "tesseract",
                "label": "Tesseract",
                "available": bool(tesseract_path),
                "binary": tesseract_path,
                "languages": tesseract_languages() if tesseract_path else [],
            },
            {
                "id": "paddleocr",
                "label": "PaddleOCR",
                "available": paddleocr_available(),
                "binary": "python package",
                "languages": ["eng"],
                "note": "Install paddleocr and paddlepaddle to enable.",
            },
            {
                "id": "lmstudio_vision",
                "label": "LM Studio Vision",
                "available": lmstudio["available"],
                "binary": lmstudio["base_url"],
                "languages": ["vision"],
                "model": lmstudio["model"],
                "models": lmstudio["models"],
                "note": lmstudio["note"],
            },
            {
                "id": "gemini_vision",
                "label": "Gemini Vision",
                "available": gemini["available"],
                "binary": gemini["base_url"],
                "languages": ["vision"],
                "model": gemini["model"],
                "models": gemini["models"],
                "note": gemini["note"],
            },
        ]
    }


def tesseract_languages() -> list[str]:
    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in completed.stdout.splitlines()[1:] if line.strip()]


def run_ocr_payload(payload: dict) -> dict:
    engine = payload.get("engine") or "tesseract"
    image_path = payload.get("image_path")
    annotation_path = payload.get("annotation_path")
    mode = payload.get("mode") or "enhance"
    language = payload.get("language") or "eng+ben"
    rotation = payload.get("rotation") or "auto"
    selected_model = payload.get("model") or None
    if not image_path:
        raise ValueError("image_path is required")

    with tempfile.TemporaryDirectory(prefix="nid-ocr-dashboard-") as tmp:
        filtered_path = Path(tmp) / "filtered.jpg"
        input_mode = "filtered"
        if annotation_path:
            annotation = load_annotation(annotation_path) or {}
            points = first_quad_points(annotation)
            if not points:
                raise ValueError("No 4-point annotation quad found")
            image_info = save_sdk_crop(image_path, points, filtered_path, mode=mode)
            input_mode = "filter_panel_crop"
        else:
            image_info = save_filtered_image(image_path, filtered_path, mode=mode)
        if engine == "tesseract":
            ocr, selected_rotation, selected_psm, candidates, parse_ocr = run_tesseract_with_rotation(
                filtered_path,
                language=language,
                mode=f"{input_mode}_{mode}",
                rotation=rotation,
            )
            parsed = OCRPipeline().parse_ocr_result(parse_ocr)
        elif engine == "paddleocr":
            ocr, selected_rotation, selected_psm, candidates, parse_ocr = run_paddle_with_rotation(
                filtered_path,
                language=language,
                mode=f"{input_mode}_{mode}",
                rotation=rotation,
            )
            parsed = OCRPipeline().parse_ocr_result(parse_ocr)
        elif engine == "lmstudio_vision":
            ocr, selected_rotation, selected_psm, candidates, fields = run_lmstudio_vision(
                filtered_path,
                mode=f"{input_mode}_{mode}",
                rotation=rotation,
                selected_model=selected_model,
            )
            parsed = nid_data_from_vision_fields(fields, raw_text=ocr.full_text, source="lmstudio_vision")
        elif engine == "gemini_vision":
            ocr, selected_rotation, selected_psm, candidates, fields = run_gemini_vision(
                filtered_path,
                mode=f"{input_mode}_{mode}",
                rotation=rotation,
                selected_model=selected_model,
            )
            parsed = nid_data_from_vision_fields(fields, raw_text=ocr.full_text, source="gemini_vision")
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")

    return {
        "image": {
            "width": image_info.width,
            "height": image_info.height,
            "mode": image_info.mode,
        },
        "rotation": selected_rotation,
        "psm": selected_psm,
        "rotation_candidates": candidates,
        "ocr": ocr_result_to_json(ocr),
        "parsed": nid_data_to_json(parsed),
    }


def run_tesseract_with_rotation(
    image_path: Path,
    language: str,
    mode: str,
    rotation: str,
) -> tuple[OCRResult, int, int, list[dict], list[tuple[int, int, OCRResult]]]:
    rotations = [0, 90, 180, 270] if rotation == "auto" else [int(rotation)]
    psm_candidates = [6, 11, 12, 3]
    engine = TesseractEngine()
    best = None
    candidates = []
    candidate_ocrs = []
    for degrees in rotations:
        rotated_path = image_path.with_name(f"filtered-rot{degrees}.jpg")
        rotate_image(image_path, rotated_path, degrees)
        for psm in psm_candidates:
            ocr = engine.recognize(
                rotated_path,
                language.split("+"),
                preprocessing=f"filtered_{mode}_rot{degrees}_psm{psm}",
                psm=psm,
            )
            score = score_ocr_result(ocr)
            candidate_ocrs.append((degrees, psm, ocr))
            candidates.append(
                {
                    "rotation": degrees,
                    "psm": psm,
                    "score": round(score, 4),
                    "blocks": len(ocr.blocks),
                    "text_preview": ocr.full_text[:160],
                }
            )
            if best is None or score > best[0]:
                best = (score, degrees, psm, ocr)
    assert best is not None
    selected = best[3]
    return selected, best[1], best[2], candidates, combine_selected_rotation_ocr(selected, candidate_ocrs, best[1])


def run_paddle_with_rotation(
    image_path: Path,
    language: str,
    mode: str,
    rotation: str,
) -> tuple[OCRResult, int, None, list[dict], OCRResult]:
    rotations = [0, 90, 180, 270] if rotation == "auto" else [int(rotation)]
    engine = PaddleOCREngine()
    best = None
    candidates = []
    for degrees in rotations:
        rotated_path = image_path.with_name(f"filtered-rot{degrees}.jpg")
        rotate_image(image_path, rotated_path, degrees)
        ocr = engine.recognize(
            rotated_path,
            language.split("+"),
            preprocessing=f"filtered_{mode}_rot{degrees}",
        )
        score = score_ocr_result(ocr)
        candidates.append(
            {
                "rotation": degrees,
                "psm": None,
                "score": round(score, 4),
                "blocks": len(ocr.blocks),
                "text_preview": ocr.full_text[:160],
            }
        )
        if best is None or score > best[0]:
            best = (score, degrees, ocr)
    assert best is not None
    return best[2], best[1], None, candidates, best[2]


def run_lmstudio_vision(
    image_path: Path,
    mode: str,
    rotation: str,
    selected_model: str | None = None,
) -> tuple[OCRResult, int, None, list[dict], dict]:
    degrees = 0 if rotation == "auto" else int(rotation)
    rotated_path = image_path.with_name(f"filtered-rot{degrees}.jpg")
    rotate_image(image_path, rotated_path, degrees)
    fields, ocr = LMStudioVisionEngine().recognize_fields(
        rotated_path,
        preprocessing=f"filtered_{mode}_rot{degrees}",
        selected_model=selected_model,
    )
    score = score_vision_fields(fields)
    candidates = [
        {
            "rotation": degrees,
            "psm": None,
            "score": round(score, 4),
            "blocks": 0,
            "text_preview": ocr.full_text[:160],
        }
    ]
    return ocr, degrees, None, candidates, fields


def run_gemini_vision(
    image_path: Path,
    mode: str,
    rotation: str,
    selected_model: str | None = None,
) -> tuple[OCRResult, int, None, list[dict], dict]:
    degrees = 0 if rotation == "auto" else int(rotation)
    rotated_path = image_path.with_name(f"filtered-rot{degrees}.jpg")
    rotate_image(image_path, rotated_path, degrees)
    fields, ocr = GeminiVisionEngine().recognize_fields(
        rotated_path,
        preprocessing=f"filtered_{mode}_rot{degrees}",
        selected_model=selected_model,
    )
    score = score_vision_fields(fields)
    candidates = [
        {
            "rotation": degrees,
            "psm": None,
            "score": round(score, 4),
            "blocks": 0,
            "text_preview": ocr.full_text[:160],
        }
    ]
    return ocr, degrees, None, candidates, fields


def score_vision_fields(fields: dict) -> float:
    return sum(1.0 for name in FIELD_NAMES if fields.get(name)) / len(FIELD_NAMES)


def nid_data_from_vision_fields(fields: dict, raw_text: str, source: str) -> NIDData:
    values = {}
    for name in FIELD_NAMES:
        value = fields.get(name)
        text = str(value).strip() if value is not None else None
        values[name] = FieldResult(
            raw_value=text or None,
            confidence=0.80 if text else 0.0,
            needs_review=True,
            source=source,
        )
    return NIDData(raw_text=raw_text, **values)


def combine_selected_rotation_ocr(
    selected: OCRResult,
    candidates: list[tuple[int, int, OCRResult]],
    selected_rotation: int,
) -> OCRResult:
    same_rotation = [ocr for degrees, _psm, ocr in candidates if degrees == selected_rotation]
    if not same_rotation:
        return selected
    full_text = "\n".join(ocr.full_text for ocr in same_rotation if ocr.full_text)
    blocks = []
    for ocr in same_rotation:
        blocks.extend(ocr.blocks)
    metadata = dict(selected.metadata)
    metadata["field_parse_sources"] = [
        {"preprocessing": ocr.preprocessing, "psm": ocr.metadata.get("psm")}
        for ocr in same_rotation
    ]
    return OCRResult(
        blocks=blocks,
        full_text=full_text,
        engine=selected.engine,
        language=selected.language,
        preprocessing=f"{selected.preprocessing}_field_parse_union",
        latency_ms=selected.latency_ms,
        metadata=metadata,
    )


def rotate_image(source: Path, target: Path, degrees: int) -> None:
    with Image.open(source) as image:
        if degrees == 0:
            image.save(target, format="JPEG", quality=92, optimize=True)
            return
        image.rotate(-degrees, expand=True).save(target, format="JPEG", quality=92, optimize=True)


def score_ocr_result(ocr: object) -> float:
    import re

    confidences = [block.confidence for block in ocr.blocks if block.confidence is not None]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    text = ocr.full_text
    lower = text.lower()
    words = [block.text for block in ocr.blocks if block.text]
    digit_words = sum(1 for word in words if re.search(r"\d|[০-৯]", word))
    low_confidence_words = sum(
        1 for block in ocr.blocks if block.confidence is not None and block.confidence < 0.45
    )
    score = confidence * 0.50
    score += min(len(ocr.blocks), 80) * 0.005
    if re.search(r"\b\d{10,17}\b", text):
        score += 1.50
    if re.search(r"\b(?:ID|NID)\s*(?:NO|NUMBER)?\s*[:：]?\s*\d{8,17}\b", text, re.IGNORECASE):
        score += 1.00
    if re.search(r"\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\-|/)", lower):
        score += 0.80
    if re.search(r"\d{1,2}\s+[a-z]{3,9}\.?,?\s+\d{4}", lower):
        score += 0.90
    for keyword in ("name", "date", "birth", "নাম", "পিতা", "মাতা", "জন্ম", "ঠিকানা"):
        if keyword in lower or keyword in text:
            score += 0.30
    for header_keyword in ("bangladesh", "government", "জাতীয়", "গণপ্রজাতন্ত্রী", "সরকার"):
        if header_keyword in lower or header_keyword in text:
            score -= 0.05
    if digit_words > 8:
        score -= (digit_words - 8) * 0.12
    if low_confidence_words > 8:
        score -= (low_confidence_words - 8) * 0.04
    return score


def nid_data_to_json(result: object) -> dict:
    output = {"raw_text": result.raw_text}
    for key, value in result.__dict__.items():
        if key == "raw_text":
            continue
        output[key] = {
            "raw_value": value.raw_value,
            "corrected_value": value.corrected_value,
            "value": value.value,
            "confidence": value.confidence,
            "needs_review": value.needs_review,
            "source": value.source,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SmartScan OCR R&D dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
