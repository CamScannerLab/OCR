from __future__ import annotations

import importlib.util
import inspect
import time
from pathlib import Path
from typing import Any

from nid_ocr_lab.engines.base import OCREngine
from nid_ocr_lab.models import BoundingBox, OCRResult, OCRTextBlock


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def recognize(
        self,
        image_path: Path,
        languages: list[str],
        preprocessing: str | None = None,
        psm: int | None = None,
    ) -> OCRResult:
        if not is_available():
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddleocr and paddlepaddle before running this engine."
            )

        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        requested_language = "+".join(languages) if languages else "eng"
        paddle_language = paddle_language_for(languages)
        started = time.perf_counter()
        ocr = create_paddle_ocr(PaddleOCR, paddle_language)
        raw = run_paddle_ocr(ocr, image_path)
        blocks = parse_paddle_result(raw)
        latency_ms = (time.perf_counter() - started) * 1000
        full_text = "\n".join(block.text for block in blocks if block.text)
        return OCRResult(
            blocks=blocks,
            full_text=full_text,
            engine=self.name,
            language=requested_language,
            preprocessing=preprocessing,
            latency_ms=latency_ms,
            metadata={
                "source_image": str(image_path),
                "paddle_language": paddle_language,
                "raw_result_shape": describe_result(raw),
            },
        )


def is_available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


def paddle_language_for(languages: list[str]) -> str:
    normalized = {language.lower() for language in languages}
    if normalized == {"eng"} or "eng" in normalized:
        return "en"
    return "en"


def create_paddle_ocr(paddle_ocr_class: Any, language: str) -> Any:
    signature = inspect.signature(paddle_ocr_class)
    kwargs: dict[str, Any] = {"lang": language}
    if "use_textline_orientation" in signature.parameters:
        kwargs["use_textline_orientation"] = True
    elif "use_angle_cls" in signature.parameters:
        kwargs["use_angle_cls"] = True
    if "show_log" in signature.parameters:
        kwargs["show_log"] = False
    return paddle_ocr_class(**kwargs)


def run_paddle_ocr(ocr: Any, image_path: Path) -> Any:
    if hasattr(ocr, "ocr"):
        signature = inspect.signature(ocr.ocr)
        if "cls" in signature.parameters:
            return ocr.ocr(str(image_path), cls=True)
        return ocr.ocr(str(image_path))
    if hasattr(ocr, "predict"):
        return ocr.predict(str(image_path))
    raise RuntimeError("Installed PaddleOCR object has neither ocr() nor predict().")


def parse_paddle_result(raw: Any) -> list[OCRTextBlock]:
    blocks: list[OCRTextBlock] = []
    for item in iter_result_items(raw):
        block = parse_result_item(item)
        if block:
            blocks.append(block)
    return blocks


def iter_result_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        payload = raw.get("res") if isinstance(raw.get("res"), dict) else raw
        rec_texts = payload.get("rec_texts") or []
        rec_scores = payload.get("rec_scores") or []
        rec_polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        return [
            [rec_polys[index] if index < len(rec_polys) else None, [text, rec_scores[index] if index < len(rec_scores) else None]]
            for index, text in enumerate(rec_texts)
        ]
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        return raw[0]
    if isinstance(raw, list):
        flattened = []
        for item in raw:
            if isinstance(item, dict):
                flattened.extend(iter_result_items(item))
                continue
            if isinstance(item, list) and item and isinstance(item[0], list) and len(item[0]) == 2:
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened
    return []


def parse_result_item(item: Any) -> OCRTextBlock | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    box = item[0]
    text_payload = item[1]
    if not isinstance(text_payload, (list, tuple)) or not text_payload:
        return None
    text = str(text_payload[0]).strip()
    if not text:
        return None
    confidence = parse_confidence(text_payload[1] if len(text_payload) > 1 else None)
    return OCRTextBlock(
        text=text,
        bounding_box=bbox_from_polygon(box),
        confidence=confidence,
    )


def bbox_from_polygon(points: Any) -> BoundingBox | None:
    if not isinstance(points, (list, tuple)):
        return None
    xs = []
    ys = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return None
    left = min(xs)
    top = min(ys)
    return BoundingBox(x=left, y=top, width=max(xs) - left, height=max(ys) - top)


def parse_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(confidence, 1.0))


def describe_result(raw: Any) -> str:
    if raw is None:
        return "none"
    if isinstance(raw, dict):
        return f"dict:{','.join(sorted(raw.keys()))}"
    if isinstance(raw, list):
        return f"list:{len(raw)}"
    return type(raw).__name__
