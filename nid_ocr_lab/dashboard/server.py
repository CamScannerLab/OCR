from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageOps

from nid_ocr_lab.dashboard.filters import (
    FILTER_MODES,
    ImageInfo,
    apply_filter,
    render_image,
    render_mask_overlay,
    render_sdk_crop,
    save_filtered_image,
    save_sdk_crop,
)
from nid_ocr_lab.dashboard.indexer import build_index, dataset_health, load_annotation
from nid_ocr_lab.dashboard.uploads import MAX_UPLOAD_BYTES, delete_upload, list_uploads, save_upload, upload_samples
from nid_ocr_lab.engines.easyocr import EasyOCREngine, is_available as easyocr_available
from nid_ocr_lab.engines.gemini_vision import GeminiVisionEngine, status as gemini_status
from nid_ocr_lab.engines.lmstudio_vision import LMStudioVisionEngine, status as lmstudio_status
from nid_ocr_lab.engines.paddleocr import PaddleOCREngine, is_available as paddleocr_available
from nid_ocr_lab.engines.tessdata import list_variants, model_version
from nid_ocr_lab.engines.tesseract import TesseractEngine, ocr_result_to_json
from nid_ocr_lab.models import FIELD_NAMES, FieldResult, NIDData, OCRResult
from nid_ocr_lab.pipeline import OCRPipeline
from nid_ocr_lab.training.dataset import crop_line, dataset_stats, save_run, save_training_lines

STATIC_DIR = Path(__file__).with_name("static")
SWEEP_PSMS = [3, 4, 6, 11]
OSD_MIN_CONFIDENCE = 2.0
RUN_CROP_PATTERN = re.compile(r"^/api/runs/([0-9a-z-]+)/crop$")
UPLOAD_ITEM_PATTERN = re.compile(r"^/api/uploads/([0-9a-z-]+)$")


class DashboardHandler(BaseHTTPRequestHandler):
    index_cache: list[dict] | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static("index.html")
        elif parsed.path == "/api/samples":
            self.send_json(upload_samples() + self.samples())
        elif parsed.path == "/api/uploads":
            self.send_json(list_uploads())
        elif parsed.path == "/api/training/datasets":
            self.send_json(dataset_stats())
        elif RUN_CROP_PATTERN.match(parsed.path):
            self.send_run_crop(RUN_CROP_PATTERN.match(parsed.path).group(1), parsed.query)
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
        elif parsed.path == "/api/uploads":
            self.receive_upload()
        elif parsed.path == "/api/training/lines":
            self.save_lines()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_DELETE(self) -> None:  # noqa: N802
        match = UPLOAD_ITEM_PATTERN.match(urlparse(self.path).path)
        if not match:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            delete_upload(match.group(1))
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"ok": True})

    def read_body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length > limit:
            raise ValueError(f"Request body exceeds {limit // (1024 * 1024)} MB")
        return self.rfile.read(length)

    def receive_upload(self) -> None:
        try:
            data = self.read_body(MAX_UPLOAD_BYTES)
            filename = unquote(self.headers.get("X-Filename", "upload"))
            record = save_upload(filename, data)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"ok": True, "upload": record, "sample_id": f"upload:{record['id']}"})

    def save_lines(self) -> None:
        try:
            payload = json.loads(self.read_body(5 * 1024 * 1024).decode("utf-8"))
            saved = save_training_lines(
                payload.get("dataset") or "",
                payload.get("run_id") or "",
                payload.get("card_id") or "",
                payload.get("lines") or [],
            )
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"ok": True, "saved": saved, "datasets": dataset_stats()})

    def send_run_crop(self, run_id: str, query_string: str) -> None:
        query = parse_qs(query_string)
        try:
            bbox = {key: float(first(query, key) or "") for key in ("x", "y", "width", "height")}
            image = crop_line(run_id, bbox)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        payload = BytesIO()
        image.save(payload, format="PNG")
        self.send_bytes(payload.getvalue(), "image/png")

    def samples(self) -> list[dict]:
        if DashboardHandler.index_cache is None:
            DashboardHandler.index_cache = build_index()
        return DashboardHandler.index_cache

    def send_image(self, query_string: str) -> None:
        query = parse_qs(query_string)
        path = first(query, "path")
        mode = first(query, "mode") or "original"
        rotation = preview_rotation(query)
        if not path or mode not in FILTER_MODES or rotation is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid image request")
            return
        try:
            payload, _info = render_image(unquote(path), mode=mode, rotation=rotation)
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
            rotation = preview_rotation(query)
            if rotation is None:
                raise ValueError("Invalid rotation")
            payload, _info = render_sdk_crop(unquote(image_path), points, mode=mode, rotation=rotation)
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

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
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


def preview_rotation(query: dict[str, list[str]]) -> int | None:
    value = first(query, "rotate") or "0"
    return int(value) if value in ("0", "90", "180", "270") else None


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
                "variants": tesseract_variants() if tesseract_path else [],
            },
            {
                "id": "easyocr",
                "label": "EasyOCR",
                "available": easyocr_available(),
                "binary": "python package · CPU",
                "languages": ["eng", "ben"],
                "note": "Install requirements-easyocr.txt in the dashboard Python environment.",
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


def tesseract_variants() -> list[dict]:
    variants = []
    for variant in list_variants():
        languages = variant.languages()
        variants.append(
            {
                "id": variant.id,
                "label": variant.label,
                "directory": str(variant.directory),
                "languages": languages,
                "versions": {
                    language: model_version(variant.model_path(language))
                    for language in languages
                    if language in ("ben", "eng") or variant.id.startswith("custom/")
                },
            }
        )
    return variants


def run_ocr_payload(payload: dict) -> dict:
    request_started = time.perf_counter()
    engine = payload.get("engine") or "tesseract"
    image_path = payload.get("image_path")
    annotation_path = payload.get("annotation_path")
    mode = payload.get("mode") or "enhance"
    input_kind = payload.get("input") or "filter"
    language = payload.get("language") or "eng+ben"
    rotation = str(payload.get("rotation") or "auto")
    selected_model = payload.get("model") or None
    if not image_path:
        raise ValueError("image_path is required")
    if rotation not in ("auto", "0", "90", "180", "270"):
        raise ValueError("Rotation must be auto, 0, 90, 180, or 270")

    with tempfile.TemporaryDirectory(prefix="nid-ocr-dashboard-") as tmp:
        if input_kind == "as_is":
            ocr_input = Path(tmp) / "input.png"
            mode = payload.get("mode") or "original"
            image_info = save_as_uploaded(image_path, ocr_input, mode=mode)
            input_mode = "as_uploaded"
        elif annotation_path:
            ocr_input = Path(tmp) / "filtered.png"
            annotation = load_annotation(annotation_path) or {}
            points = first_quad_points(annotation)
            if not points:
                raise ValueError("No 4-point annotation quad found")
            image_info = save_sdk_crop(image_path, points, ocr_input, mode=mode)
            input_mode = "filter_panel_crop"
        else:
            ocr_input = Path(tmp) / "filtered.png"
            image_info = save_filtered_image(image_path, ocr_input, mode=mode)
            input_mode = "filtered"
        label = f"{input_mode}_{mode}"
        extra: dict = {}
        if engine == "tesseract":
            tess = run_tesseract(
                ocr_input,
                language=language,
                mode=label,
                rotation=rotation,
                variant=payload.get("tesseract_variant") or "system",
                psm=int(payload.get("psm") or 6),
                strategy=payload.get("strategy") or "single",
            )
            ocr, selected_rotation, selected_psm, candidates = tess["ocr"], tess["rotation"], tess["psm"], tess["candidates"]
            parsed = OCRPipeline().parse_ocr_result(ocr)
            extra = {"ocr_calls": tess["calls"], "orientation": tess["orientation"]}
            extra["run_id"] = save_run(
                tess["input_path"],
                ocr_result_to_json(ocr),
                {
                    "sample_id": payload.get("sample_id"),
                    "image_path": str(image_path),
                    "input_mode": input_mode,
                    "filter_mode": mode,
                    "rotation": selected_rotation,
                    "variant": ocr.metadata.get("variant"),
                    "language": ocr.language,
                    "psm": selected_psm,
                },
            )
        elif engine == "easyocr":
            degrees = 0 if rotation == "auto" else int(rotation)
            easy_path = rotated_copy(ocr_input, degrees)
            ocr = EasyOCREngine().recognize(
                easy_path, language.split("+"),
                preprocessing=f"{label}_rot{degrees}",
                auto_rotate=rotation == "auto",
            )
            selected_rotation, selected_psm, candidates = degrees, None, []
            parsed = OCRPipeline().parse_ocr_result(ocr)
        elif engine == "paddleocr":
            ocr, selected_rotation, selected_psm, candidates, parse_ocr = run_paddle_with_rotation(
                ocr_input,
                language=language,
                mode=label,
                rotation=rotation,
            )
            parsed = OCRPipeline().parse_ocr_result(parse_ocr)
        elif engine == "lmstudio_vision":
            ocr, selected_rotation, selected_psm, candidates, fields = run_lmstudio_vision(
                ocr_input,
                mode=label,
                rotation=rotation,
                selected_model=selected_model,
            )
            parsed = nid_data_from_vision_fields(fields, raw_text=ocr.full_text, source="lmstudio_vision")
        elif engine == "gemini_vision":
            ocr, selected_rotation, selected_psm, candidates, fields = run_gemini_vision(
                ocr_input,
                mode=label,
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
            "source_path": str(image_path),
            "source_name": Path(image_path).name,
            "input_mode": input_mode,
            "filter_mode": mode,
        },
        "request_latency_ms": (time.perf_counter() - request_started) * 1000,
        "rotation": selected_rotation,
        "psm": selected_psm,
        "rotation_candidates": candidates,
        **extra,
        "ocr": ocr_result_to_json(ocr),
        "parsed": nid_data_to_json(parsed),
    }


def save_as_uploaded(image_path: str, target: Path, mode: str = "original") -> ImageInfo:
    """No crop or resize: EXIF orientation plus the optional filter, saved losslessly with DPI metadata kept."""
    if mode not in FILTER_MODES:
        raise ValueError(f"Unknown filter mode: {mode}")
    with Image.open(image_path) as source:
        dpi = source.info.get("dpi")
        image = ImageOps.exif_transpose(source)
        if mode != "original":
            image = apply_filter(image.convert("RGB"), mode)
        elif image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(target, **({"dpi": dpi} if dpi else {}))
        return ImageInfo(width=image.width, height=image.height, mode="as_uploaded")


def rotated_copy(source: Path, degrees: int) -> Path:
    """Clockwise rotation into a sibling PNG; 0 degrees returns the original file untouched."""
    degrees %= 360
    if degrees == 0:
        return source
    target = source.with_name(f"{source.stem}-rot{degrees}.png")
    if not target.exists():
        with Image.open(source) as image:
            dpi = image.info.get("dpi")
            image.rotate(-degrees, expand=True).save(target, **({"dpi": dpi} if dpi else {}))
    return target


def orientation_score(ocr: OCRResult) -> float:
    words = ocr.metadata.get("words") or []
    confidences = [word["confidence"] for word in words if word.get("confidence") is not None]
    if not confidences:
        return 0.0
    return (sum(confidences) / len(confidences)) * len(confidences)


def run_tesseract(
    image_path: Path,
    language: str,
    mode: str,
    rotation: str,
    variant: str,
    psm: int,
    strategy: str,
) -> dict:
    """Bounded Tesseract flow: one OSD pass (plus a 4-way check only if OSD cannot decide), then one OCR
    call per requested PSM. Results are never concatenated across PSMs."""
    engine = TesseractEngine()
    languages = language.split("+")
    calls = 0
    orientation: dict = {"method": "manual"}
    reusable: dict[int, OCRResult] = {}

    def recognize(path: Path, degrees: int, page_psm: int) -> OCRResult:
        nonlocal calls
        calls += 1
        return engine.recognize(
            path, languages, preprocessing=f"{mode}_rot{degrees}_psm{page_psm}", psm=page_psm, variant=variant
        )

    if rotation == "auto":
        osd = engine.detect_orientation(image_path)
        calls += 1
        if osd and osd.confidence >= OSD_MIN_CONFIDENCE:
            degrees = osd.rotate
            orientation = {"method": "osd", "rotate": osd.rotate, "confidence": osd.confidence, "script": osd.script}
        else:
            scores = {}
            for candidate in (0, 90, 180, 270):
                result = recognize(rotated_copy(image_path, candidate), candidate, psm)
                reusable[candidate] = result
                scores[candidate] = round(orientation_score(result), 3)
            degrees = max(scores, key=scores.get)
            orientation = {
                "method": "rotation-check",
                "osd": None if osd is None else {"rotate": osd.rotate, "confidence": osd.confidence},
                "scores": scores,
                "score": "sum of word confidences",
            }
    else:
        degrees = int(rotation)

    input_path = rotated_copy(image_path, degrees)
    candidates = []
    if strategy == "sweep":
        best = None
        for page_psm in SWEEP_PSMS:
            result = reusable.get(degrees) if page_psm == psm and degrees in reusable else recognize(input_path, degrees, page_psm)
            score = score_ocr_result(result)
            candidates.append(
                {
                    "rotation": degrees,
                    "psm": page_psm,
                    "score": round(score, 4),
                    "score_kind": "heuristic, not accuracy",
                    "blocks": len(result.blocks),
                    "latency_ms": round(result.latency_ms or 0),
                    "text_preview": result.full_text[:160],
                }
            )
            if best is None or score > best[0]:
                best = (score, page_psm, result)
        assert best is not None
        selected_psm, ocr = best[1], best[2]
    else:
        ocr = reusable.get(degrees) or recognize(input_path, degrees, psm)
        selected_psm = psm
    return {
        "ocr": ocr,
        "rotation": degrees,
        "psm": selected_psm,
        "candidates": candidates,
        "calls": calls,
        "orientation": orientation,
        "input_path": input_path,
    }


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
        rotated_path = rotated_copy(image_path, degrees)
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
    rotated_path = rotated_copy(image_path, degrees)
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
    rotated_path = rotated_copy(image_path, degrees)
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
