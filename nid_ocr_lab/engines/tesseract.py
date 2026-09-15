from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from nid_ocr_lab.engines.base import OCREngine
from nid_ocr_lab.engines.tessdata import describe_models, get_variant, system_tessdata_dir
from nid_ocr_lab.models import BoundingBox, OCRResult, OCRTextBlock

DEFAULT_PSM = 6
DEFAULT_OEM = 1  # LSTM only; the fast/best models carry no legacy engine.


@dataclass(frozen=True)
class Orientation:
    rotate: int
    confidence: float
    script: str | None = None
    script_confidence: float | None = None


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, binary: str = "tesseract") -> None:
        self.binary = binary

    def ensure_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                "Tesseract is not installed or is not on PATH. Install it before running OCR."
            )

    def build_command(
        self,
        image_path: Path,
        language: str,
        tessdata_dir: Path,
        psm: int,
        oem: int,
        dpi: int | None,
        config: dict[str, str] | None = None,
    ) -> list[str]:
        command = [
            self.binary,
            str(image_path),
            "stdout",
            "--tessdata-dir",
            str(tessdata_dir),
            "-l",
            language,
            "--oem",
            str(oem),
            "--psm",
            str(psm),
        ]
        if dpi:
            command.extend(["--dpi", str(dpi)])
        # Set the renderer by parameter: the "tsv" config file only exists in the system tessdata/configs.
        params = {"tessedit_create_tsv": "1", **(config or {})}
        for key, value in params.items():
            command.extend(["-c", f"{key}={value}"])
        return command

    def recognize(
        self,
        image_path: Path,
        languages: list[str],
        preprocessing: str | None = None,
        psm: int | None = None,
        *,
        variant: str | None = "system",
        oem: int = DEFAULT_OEM,
        dpi: int | None = None,
        threads: int | None = None,
        config: dict[str, str] | None = None,
    ) -> OCRResult:
        self.ensure_available()
        language = "+".join(languages) if languages else "eng"
        model_variant = get_variant(variant)
        missing = [item for item in language.split("+") if not model_variant.model_path(item).is_file()]
        if missing:
            raise RuntimeError(
                f"Tesseract variant '{model_variant.id}' has no model for {missing} in {model_variant.directory}."
            )
        psm = DEFAULT_PSM if psm is None else psm
        resolved_dpi, dpi_source = resolve_dpi(image_path, dpi)
        command = self.build_command(
            image_path, language, model_variant.directory, psm, oem, resolved_dpi, config
        )
        env = dict(os.environ)
        if threads:
            env["OMP_THREAD_LIMIT"] = str(threads)

        started = time.perf_counter()
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        latency_ms = (time.perf_counter() - started) * 1000
        raw_tsv = completed.stdout
        blocks, words, full_text = parse_tsv_lines(raw_tsv)
        if resolved_dpi is None:
            resolved_dpi = estimated_dpi(completed.stderr)

        return OCRResult(
            blocks=blocks,
            full_text=full_text,
            engine=self.name,
            language=language,
            preprocessing=preprocessing,
            latency_ms=latency_ms,
            metadata={
                "source_image": str(image_path),
                "psm": psm,
                "oem": oem,
                "dpi": resolved_dpi,
                "dpi_source": dpi_source,
                "threads": threads,
                "variant": model_variant.id,
                "tessdata_dir": str(model_variant.directory),
                "models": describe_models(model_variant, language.split("+")),
                "command": command,
                "stderr": completed.stderr.strip() or None,
                "block_granularity": "line",
                "words": words,
                "raw_response": raw_tsv,
            },
        )

    def detect_orientation(self, image_path: Path, dpi: int | None = None) -> Orientation | None:
        """Single OSD pass with the system osd model. Returns None when OSD cannot decide."""
        self.ensure_available()
        resolved_dpi, _source = resolve_dpi(image_path, dpi)
        command = [
            self.binary,
            str(image_path),
            "stdout",
            "--tessdata-dir",
            str(system_tessdata_dir()),
            "--psm",
            "0",
            *(["--dpi", str(resolved_dpi)] if resolved_dpi else []),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            return None
        return parse_osd(completed.stdout)


def resolve_dpi(image_path: Path, dpi: int | None) -> tuple[int | None, str]:
    """Explicit DPI, else trustworthy image metadata, else let Tesseract estimate from text size."""
    if dpi:
        return int(dpi), "explicit"
    try:
        with Image.open(image_path) as image:
            value = image.info.get("dpi")
    except OSError:
        value = None
    if value:
        horizontal = float(value[0] if isinstance(value, tuple) else value)
        if 70 <= horizontal <= 1200:
            return round(horizontal), "image metadata"
    return None, "tesseract estimate"


def estimated_dpi(stderr: str) -> int | None:
    match = re.search(r"Estimating resolution as (\d+)", stderr or "")
    return int(match.group(1)) if match else None


def parse_osd(text: str) -> Orientation | None:
    rotate = re.search(r"Rotate:\s*(\d+)", text)
    confidence = re.search(r"Orientation confidence:\s*([\d.]+)", text)
    if not rotate or not confidence:
        return None
    script = re.search(r"Script:\s*(\S+)", text)
    script_confidence = re.search(r"Script confidence:\s*([\d.]+)", text)
    return Orientation(
        rotate=int(rotate.group(1)) % 360,
        confidence=float(confidence.group(1)),
        script=script.group(1) if script else None,
        script_confidence=float(script_confidence.group(1)) if script_confidence else None,
    )


def parse_tsv_rows(tsv: str) -> list[dict[str, str]]:
    lines = [line for line in tsv.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    return [dict(zip(headers, line.split("\t"))) for line in lines[1:]]


def parse_tsv_lines(tsv: str) -> tuple[list[OCRTextBlock], list[dict[str, Any]], str]:
    """Rebuild Tesseract's own page/block/paragraph/line hierarchy from word rows."""
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    words: list[dict[str, Any]] = []
    for row in parse_tsv_rows(tsv):
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        confidence = parse_float(row.get("conf"))
        word = {
            "text": text,
            "confidence": confidence / 100.0 if confidence is not None and confidence >= 0 else None,
            "bounding_box": {
                "x": parse_float(row.get("left")) or 0.0,
                "y": parse_float(row.get("top")) or 0.0,
                "width": parse_float(row.get("width")) or 0.0,
                "height": parse_float(row.get("height")) or 0.0,
            },
            "line_key": [int(parse_float(row.get(key)) or 0) for key in ("page_num", "block_num", "par_num", "line_num")],
        }
        words.append(word)
        grouped.setdefault(tuple(word["line_key"]), []).append(word)

    blocks: list[OCRTextBlock] = []
    text_lines: list[str] = []
    previous_paragraph = None
    for key in sorted(grouped):
        line_words = grouped[key]
        boxes = [word["bounding_box"] for word in line_words]
        left = min(box["x"] for box in boxes)
        top = min(box["y"] for box in boxes)
        right = max(box["x"] + box["width"] for box in boxes)
        bottom = max(box["y"] + box["height"] for box in boxes)
        confidences = [word["confidence"] for word in line_words if word["confidence"] is not None]
        text = " ".join(word["text"] for word in line_words)
        blocks.append(
            OCRTextBlock(
                text=text,
                bounding_box=BoundingBox(x=left, y=top, width=right - left, height=bottom - top),
                confidence=sum(confidences) / len(confidences) if confidences else None,
            )
        )
        paragraph = key[:3]
        if previous_paragraph is not None and paragraph != previous_paragraph:
            text_lines.append("")
        previous_paragraph = paragraph
        text_lines.append(text)
    return blocks, words, "\n".join(text_lines)


def parse_tsv(tsv: str) -> list[OCRTextBlock]:
    return parse_tsv_lines(tsv)[0]


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
