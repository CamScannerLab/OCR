"""Read one line crop with Tesseract.

The CLI costs ~54 ms of process startup per call, which was 80% of the per-row re-read time.
The C API reader keeps one initialized handle per (model folder, language) for the whole process,
so a re-read costs ~10 ms instead of ~89 ms with the same output.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nid_ocr_lab.engines import tesseract_capi as capi
from nid_ocr_lab.engines.tesseract import TesseractEngine

_HANDLES: dict[tuple[str, str], tuple[capi.TessAPI, threading.Lock]] = {}
_HANDLES_LOCK = threading.Lock()


@dataclass(frozen=True)
class LineRead:
    text: str
    confidence: float | None  # 0..1, or None when Tesseract reported nothing


class CApiLineReader:
    """Reuses initialized Tesseract handles; the dashboard server is threaded, so each handle has a lock."""

    name = "capi"

    def handle(self, datapath: Path, language: str) -> tuple[capi.TessAPI, threading.Lock]:
        key = (str(datapath), language)
        with _HANDLES_LOCK:
            if key not in _HANDLES:
                _HANDLES[key] = (capi.TessAPI(datapath, language), threading.Lock())
            return _HANDLES[key]

    def read(self, image_path: Path, datapath: Path, languages: list[str], psm: int, config: dict[str, str] | None = None) -> LineRead:
        api, lock = self.handle(datapath, "+".join(languages))
        with lock:
            try:
                for key, value in (config or {}).items():
                    api.set_variable(key, value)
                api.set_psm(psm)
                api.set_image(image_path)
                text = api.text()
                confidence = api.mean_confidence()
            finally:
                # The handle outlives this call, so per-call variables must not leak into the next one.
                for key in config or {}:
                    api.set_variable(key, "")
        return LineRead(text=text.strip(), confidence=confidence / 100.0 if confidence and confidence > 0 else None)


class CliLineReader:
    """Fallback when the C API cannot be loaded: one `tesseract` process per line."""

    name = "cli"

    def __init__(self, engine: TesseractEngine | None = None) -> None:
        self.engine = engine or TesseractEngine()

    def read(self, image_path: Path, datapath: Path, languages: list[str], psm: int, config: dict[str, str] | None = None) -> LineRead:
        result = self.engine.recognize(image_path, languages, psm=psm, variant=_variant_for(datapath), dpi=300, config=config)
        confidences = [word["confidence"] for word in result.metadata.get("words") or [] if word.get("confidence") is not None]
        return LineRead(text=result.full_text.strip(), confidence=sum(confidences) / len(confidences) if confidences else None)


def _variant_for(datapath: Path) -> str:
    from nid_ocr_lab.engines.tessdata import list_variants

    for variant in list_variants():
        if variant.directory == datapath:
            return variant.id
    return "system"


def line_reader(engine: TesseractEngine | None = None) -> Any:
    return CApiLineReader() if capi.is_available() else CliLineReader(engine)


def close_handles() -> None:
    """Release cached handles (tests, shutdown)."""
    with _HANDLES_LOCK:
        for api, _lock in _HANDLES.values():
            api.close()
        _HANDLES.clear()
