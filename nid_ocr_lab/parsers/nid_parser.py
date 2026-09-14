from __future__ import annotations

import re

from nid_ocr_lab.models import FIELD_NAMES, FieldResult, NIDData, OCRResult


DATE_PATTERNS = [
    re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"),
]

MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

MONTH_DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b",
    re.IGNORECASE,
)

NID_CONTEXT_PATTERN = re.compile(
    r"\b(?:ID|NID)\s*(?:NO|NUMBER|নং)?\s*[:：]?\s*([0-9০-৯]{8,17})\b",
    re.IGNORECASE,
)

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


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
    context_match = NID_CONTEXT_PATTERN.search(text)
    if context_match:
        return FieldResult(
            raw_value=normalize_digits(context_match.group(1)),
            confidence=0.90,
            needs_review=False,
            source="id-context-regex",
        )
    ascii_candidates = re.findall(r"\b[0-9]{10,17}\b", text)
    if ascii_candidates:
        value = max(ascii_candidates, key=lambda item: (len(item) == 10, len(item)))
        return FieldResult(raw_value=value, confidence=0.85, needs_review=False, source="regex")
    candidates = re.findall(r"\b[0-9০-৯]{10,17}\b", text)
    if not candidates:
        return empty_field()
    value = normalize_digits(max(candidates, key=len))
    return FieldResult(raw_value=value, confidence=0.85, needs_review=False, source="regex")


def normalize_digits(value: str) -> str:
    return value.translate(BANGLA_DIGITS)


def find_date_of_birth(text: str) -> FieldResult:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = "-".join(part.zfill(2) for part in match.groups())
            return FieldResult(raw_value=value, confidence=0.75, needs_review=False, source="regex")
    match = MONTH_DATE_PATTERN.search(text)
    if match:
        day, month_name, year = match.groups()
        month = MONTHS.get(month_name.lower().rstrip("."))
        if month:
            value = f"{year}-{month}-{day.zfill(2)}"
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
            value = following_value(lines, index + 1)
            if value:
                return FieldResult(raw_value=value, confidence=0.50, needs_review=True, source="label-next-line")
    return empty_field()


def following_value(lines: list[str], start: int) -> str | None:
    values = []
    stop_words = {
        "date",
        "birth",
        "id",
        "no",
        "father",
        "mother",
        "address",
        "নাম",
        "পিতা",
        "মাতা",
        "ঠিকানা",
    }
    for line in lines[start : start + 4]:
        value = normalize_space(line)
        if not value:
            continue
        lower = value.lower().strip(":：")
        if lower in stop_words:
            break
        if re.search(r"\d", value):
            break
        values.append(value)
    return normalize_space(" ".join(values)) if values else None


def value_after_separator(line: str) -> str | None:
    parts = re.split(r"[:：]", line, maxsplit=1)
    if len(parts) != 2:
        return None
    value = normalize_space(parts[1])
    return value or None
