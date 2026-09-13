from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class OCRTextBlock:
    text: str
    bounding_box: BoundingBox | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class OCRResult:
    blocks: list[OCRTextBlock]
    full_text: str
    engine: str
    language: str | None = None
    preprocessing: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldResult:
    raw_value: str | None
    corrected_value: str | None = None
    confidence: float = 0.0
    needs_review: bool = True
    source: str | None = None

    @property
    def value(self) -> str | None:
        return self.corrected_value or self.raw_value


@dataclass(frozen=True)
class NIDData:
    nid_number: FieldResult
    name_bangla: FieldResult
    name_english: FieldResult
    father_name_bangla: FieldResult
    father_name_english: FieldResult
    mother_name_bangla: FieldResult
    mother_name_english: FieldResult
    date_of_birth: FieldResult
    address_bangla: FieldResult
    address_english: FieldResult
    raw_text: str


FIELD_NAMES = [
    "nid_number",
    "name_bangla",
    "name_english",
    "father_name_bangla",
    "father_name_english",
    "mother_name_bangla",
    "mother_name_english",
    "date_of_birth",
    "address_bangla",
    "address_english",
]

