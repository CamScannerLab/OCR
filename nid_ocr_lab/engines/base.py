from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nid_ocr_lab.models import OCRResult


class OCREngine(Protocol):
    name: str

    def recognize(
        self,
        image_path: Path,
        languages: list[str],
        preprocessing: str | None = None,
    ) -> OCRResult:
        """Recognize text from a SmartScan output image."""

