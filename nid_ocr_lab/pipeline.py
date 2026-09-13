from __future__ import annotations

from nid_ocr_lab.models import NIDData, OCRResult
from nid_ocr_lab.parsers.nid_parser import NIDParser


class OCRPipeline:
    def __init__(self, parser: NIDParser | None = None) -> None:
        self.parser = parser or NIDParser()

    def parse_ocr_result(self, ocr: OCRResult) -> NIDData:
        return self.parser.parse(ocr)

