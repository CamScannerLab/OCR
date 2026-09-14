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
    save_sdk_crop,
)
from nid_ocr_lab.dashboard.indexer import build_index, dataset_health, load_annotation
from nid_ocr_lab.engines.tesseract import TesseractEngine, ocr_result_to_json
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
            result = run_tesseract_payload(payload)
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
    return {
        "engines": [
            {
                "id": "tesseract",
                "label": "Tesseract",
                "available": bool(tesseract_path),
                "binary": tesseract_path,
                "languages": tesseract_languages() if tesseract_path else [],
            }
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


def run_tesseract_payload(payload: dict) -> dict:
    if not shutil.which("tesseract"):
        raise RuntimeError("Tesseract is not installed or is not on PATH.")

    image_path = payload.get("image_path")
    annotation_path = payload.get("annotation_path")
    mode = payload.get("mode") or "enhance"
    language = payload.get("language") or "eng+ben"
    rotation = payload.get("rotation") or "auto"
    if not image_path or not annotation_path:
        raise ValueError("image_path and annotation_path are required")

    annotation = load_annotation(annotation_path) or {}
    points = first_quad_points(annotation)
    if not points:
        raise ValueError("No 4-point annotation quad found")

    with tempfile.TemporaryDirectory(prefix="nid-ocr-dashboard-") as tmp:
        crop_path = Path(tmp) / "sdk-crop.jpg"
        crop_info = save_sdk_crop(image_path, points, crop_path, mode=mode)
        ocr, selected_rotation, candidates = run_tesseract_with_rotation(
            crop_path,
            language=language,
            mode=mode,
            rotation=rotation,
        )

    parsed = OCRPipeline().parse_ocr_result(ocr)
    return {
        "crop": {
            "width": crop_info.width,
            "height": crop_info.height,
            "mode": crop_info.mode,
        },
        "rotation": selected_rotation,
        "rotation_candidates": candidates,
        "ocr": ocr_result_to_json(ocr),
        "parsed": nid_data_to_json(parsed),
    }


def run_tesseract_with_rotation(
    crop_path: Path,
    language: str,
    mode: str,
    rotation: str,
) -> tuple[object, int, list[dict]]:
    rotations = [0, 90, 180, 270] if rotation == "auto" else [int(rotation)]
    engine = TesseractEngine()
    best = None
    candidates = []
    for degrees in rotations:
        rotated_path = crop_path.with_name(f"sdk-crop-rot{degrees}.jpg")
        rotate_image(crop_path, rotated_path, degrees)
        ocr = engine.recognize(
            rotated_path,
            language.split("+"),
            preprocessing=f"sdk_crop_{mode}_rot{degrees}",
        )
        score = score_ocr_result(ocr)
        candidates.append(
            {
                "rotation": degrees,
                "score": round(score, 4),
                "blocks": len(ocr.blocks),
                "text_preview": ocr.full_text[:160],
            }
        )
        if best is None or score > best[0]:
            best = (score, degrees, ocr)
    assert best is not None
    return best[2], best[1], candidates


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
    score = confidence
    score += min(len(ocr.blocks), 30) * 0.01
    if re.search(r"\b\d{10,17}\b", text):
        score += 0.35
    if re.search(r"\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\-|/)", lower):
        score += 0.20
    for keyword in ("bangladesh", "government", "name", "date", "birth", "id", "জাতীয়", "নাম", "পিতা", "মাতা"):
        if keyword in lower or keyword in text:
            score += 0.08
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
