from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from nid_ocr_lab.engines.base import OCREngine
from nid_ocr_lab.models import BoundingBox, OCRResult, OCRTextBlock


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, binary: str = "tesseract") -> None:
        self.binary = binary

    def recognize(
        self,
        image_path: Path,
        languages: list[str],
        preprocessing: str | None = None,
    ) -> OCRResult:
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                "Tesseract is not installed or is not on PATH. Install it before running OCR."
            )

        language = "+".join(languages) if languages else "eng"
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nid-ocr-tess-") as tmp:
            output_base = Path(tmp) / "out"
            command = [
                self.binary,
                str(image_path),
                str(output_base),
                "-l",
                language,
                "tsv",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            tsv_path = output_base.with_suffix(".tsv")
            blocks = parse_tsv(tsv_path.read_text(encoding="utf-8", errors="replace"))

        latency_ms = (time.perf_counter() - started) * 1000
        full_text = "\n".join(block.text for block in blocks if block.text)
        return OCRResult(
            blocks=blocks,
            full_text=full_text,
            engine=self.name,
            language=language,
            preprocessing=preprocessing,
            latency_ms=latency_ms,
            metadata={"source_image": str(image_path)},
        )


def parse_tsv(tsv: str) -> list[OCRTextBlock]:
    lines = [line for line in tsv.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    headers = lines[0].split("\t")
    blocks = []
    for line in lines[1:]:
        row = dict(zip(headers, line.split("\t")))
        text = (row.get("text") or "").strip()
        if not text:
            continue
        confidence = parse_float(row.get("conf"))
        if confidence is not None and confidence < 0:
            confidence = None
        bbox = BoundingBox(
            x=parse_float(row.get("left")) or 0.0,
            y=parse_float(row.get("top")) or 0.0,
            width=parse_float(row.get("width")) or 0.0,
            height=parse_float(row.get("height")) or 0.0,
        )
        blocks.append(
            OCRTextBlock(
                text=text,
                bounding_box=bbox,
                confidence=confidence / 100.0 if confidence is not None else None,
            )
        )
    return blocks


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ocr_result_to_json(result: OCRResult) -> dict[str, Any]:
    return {
        "engine": result.engine,
        "language": result.language,
        "preprocessing": result.preprocessing,
        "latency_ms": result.latency_ms,
        "full_text": result.full_text,
        "metadata": result.metadata,
        "blocks": [
            {
                "text": block.text,
                "confidence": block.confidence,
                "bounding_box": {
                    "x": block.bounding_box.x,
                    "y": block.bounding_box.y,
                    "width": block.bounding_box.width,
                    "height": block.bounding_box.height,
                }
                if block.bounding_box
                else None,
            }
            for block in result.blocks
        ],
    }

