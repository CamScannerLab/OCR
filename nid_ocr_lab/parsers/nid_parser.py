from __future__ import annotations

import re

from nid_ocr_lab.models import FIELD_NAMES, FieldResult, NIDData, OCRResult


DATE_PATTERNS = [
    re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"),
]


class NIDParser:
    """Initial text parser for R&D.

    This parser intentionally starts simple. The benchmark should expose where
    layout-aware bbox parsing and ROI fallback are needed.
    """

    def parse(self, ocr: OCRResult) -> NIDData:
        text = normalize_space(ocr.full_text)
        fields = {name: empty_field() for name in FIELD_NAMES}
        fields["nid_number"] = find_nid_number(text)
        fields["date_of_birth"] = find_date_of_birth(text)

        lines = [normalize_space(line) for line in ocr.full_text.splitlines()]
        fields["name_english"] = find_labeled_value(lines, ["name", "নাম"])
        fields["father_name_english"] = find_labeled_value(lines, ["father", "পিতা"])
        fields["mother_name_english"] = find_labeled_value(lines, ["mother", "মাতা"])
        fields["address_english"] = find_labeled_value(lines, ["address", "ঠিকানা"])

        return NIDData(raw_text=ocr.full_text, **fields)


def empty_field() -> FieldResult:
    return FieldResult(raw_value=None, confidence=0.0, needs_review=True)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def find_nid_number(text: str) -> FieldResult:
    candidates = re.findall(r"\b\d{10,17}\b", text)
    if not candidates:
        return empty_field()
    value = max(candidates, key=len)
    return FieldResult(raw_value=value, confidence=0.85, needs_review=False, source="regex")


def find_date_of_birth(text: str) -> FieldResult:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = "-".join(part.zfill(2) for part in match.groups())
            return FieldResult(raw_value=value, confidence=0.75, needs_review=False, source="regex")
    return empty_field()


def find_labeled_value(lines: list[str], labels: list[str]) -> FieldResult:
    lowered_labels = [label.lower() for label in labels]
    for index, line in enumerate(lines):
        lower = line.lower()
        if any(label in lower for label in lowered_labels):
            inline = value_after_separator(line)
            if inline:
                return FieldResult(raw_value=inline, confidence=0.55, needs_review=True, source="label-inline")
            if index + 1 < len(lines) and lines[index + 1]:
                return FieldResult(raw_value=lines[index + 1], confidence=0.50, needs_review=True, source="label-next-line")
    return empty_field()


def value_after_separator(line: str) -> str | None:
    parts = re.split(r"[:：]", line, maxsplit=1)
    if len(parts) != 2:
        return None
    value = normalize_space(parts[1])
    return value or None

