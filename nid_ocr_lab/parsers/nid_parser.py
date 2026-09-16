from __future__ import annotations

import re
from datetime import date

from nid_ocr_lab.models import FIELD_NAMES, FieldResult, NIDData, OCRResult


MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

ISO_DATE_PATTERN = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
DMY_DATE_PATTERN = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b")
MONTH_DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*([A-Za-z]{3,9})\.?,?\s*(\d{4})\b", re.IGNORECASE)

# Cards print the number as one run or in spaced groups ("597 029 5035").
NID_CONTEXT_PATTERN = re.compile(
    r"\b(?:ID|NID)\s*(?:NO|NUMBER|নং)?\.?\s*[:：]?\s*([0-9০-৯]{3,17}(?:[  ][0-9০-৯]{2,6}){0,4})",
    re.IGNORECASE,
)

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BENGALI_RANGE = "ঀ-৿"
LETTER_CLASS = f"A-Za-z{BENGALI_RANGE}'’"

# (field, labels, expected script). English labels match case-insensitively.
LABELED_FIELDS = [
    ("name_bangla", ["নাম", "লাম"], "ben"),
    ("name_english", ["name", "neme", "narne", "nane"], "eng"),
    ("father_name_bangla", ["পিতা"], "ben"),
    ("father_name_english", ["father's name", "father"], "eng"),
    ("mother_name_bangla", ["মাতা"], "ben"),
    ("mother_name_english", ["mother's name", "mother"], "eng"),
    ("address_bangla", ["ঠিকানা"], "ben"),
    ("address_english", ["address"], "eng"),
]

# A bare "Name"/"নাম" label preceded by one of these belongs to another person's field.
RELATION_WORDS = ("father", "mother", "husband", "spouse", "পিতা", "মাতা", "স্বামী", "স্ত্রী")

LABEL_WORDS = {
    "date", "birth", "id", "no", "name", "neme", "narne", "nane", "father", "mother", "address",
    "নাম", "লাম", "পিতা", "মাতা", "ঠিকানা", "জন্ম", "তারিখ",
}


class NIDParser:
    """Text parser for R&D.

    Works on line-level OCR text (one OCR line per text line). Values are routed by
    label script: Bengali labels fill *_bangla fields, English labels fill *_english.
    """

    def parse(self, ocr: OCRResult) -> NIDData:
        text = normalize_space(ocr.full_text)
        fields = {name: empty_field() for name in FIELD_NAMES}
        fields["nid_number"] = find_nid_number(text)
        fields["date_of_birth"] = find_date_of_birth(ocr.full_text)

        lines = [normalize_space(line) for line in ocr.full_text.splitlines()]
        lines = [line for line in lines if line]
        for field, labels, script in LABELED_FIELDS:
            fields[field] = find_labeled_value(lines, labels, script)

        return NIDData(raw_text=ocr.full_text, **fields)


def empty_field(source: str | None = None) -> FieldResult:
    return FieldResult(raw_value=None, confidence=0.0, needs_review=True, source=source)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_digits(value: str) -> str:
    return value.translate(BANGLA_DIGITS)


def find_nid_number(text: str) -> FieldResult:
    context_match = NID_CONTEXT_PATTERN.search(text)
    if context_match and len(re.sub(r"\D", "", normalize_digits(context_match.group(1)))) >= 8:
        return FieldResult(
            raw_value=re.sub(r"\D", "", normalize_digits(context_match.group(1))),
            confidence=0.90,
            needs_review=False,
            source="id-context-regex",
        )
    # Fallback: a long digit run anywhere, in one piece or in spaced groups ("597 029 5035").
    candidates = re.findall(r"\b[0-9০-৯]{3,17}(?:[  ][0-9০-৯]{2,6}){0,4}\b", text)
    normalized = [re.sub(r"\D", "", normalize_digits(item)) for item in candidates]
    normalized = [item for item in normalized if len(item) in (10, 13, 17)]
    if not normalized:
        return empty_field()
    value = max(normalized, key=lambda item: (len(item) in (10, 13, 17), len(item)))
    return FieldResult(raw_value=value, confidence=0.70, needs_review=True, source="regex")


def valid_iso(year: int, month: int, day: int) -> str | None:
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    if not 1900 <= parsed.year <= date.today().year:
        return None
    return parsed.isoformat()


def find_date_of_birth(text: str) -> FieldResult:
    normalized = normalize_digits(text)
    lines = normalized.splitlines()
    # Prefer the line carrying the birth-date label, then anywhere in the text.
    labeled = [line for line in lines if re.search(r"birth|জন্ম", line, re.IGNORECASE)]
    for source, haystack in (("dob-label", "\n".join(labeled)), ("regex", normalized)):
        if not haystack:
            continue
        for match in MONTH_DATE_PATTERN.finditer(haystack):
            day, month_name, year = match.groups()
            month = MONTHS.get(month_name.lower().rstrip("."))
            iso = valid_iso(int(year), month, int(day)) if month else None
            if iso:
                return FieldResult(raw_value=iso, confidence=0.75, needs_review=False, source=source)
        for match in ISO_DATE_PATTERN.finditer(haystack):
            year, month, day = match.groups()
            iso = valid_iso(int(year), int(month), int(day))
            if iso:
                return FieldResult(raw_value=iso, confidence=0.70, needs_review=False, source=source)
        for match in DMY_DATE_PATTERN.finditer(haystack):
            day, month, year = match.groups()
            iso = valid_iso(int(year), int(month), int(day))
            if iso:
                return FieldResult(raw_value=iso, confidence=0.70, needs_review=False, source=source)
    return empty_field()


def label_match(line: str, label: str) -> re.Match | None:
    flags = re.IGNORECASE if label.isascii() else 0
    escaped = re.escape(label).replace("'", "['’]?")
    pattern = re.compile(rf"(?<![{LETTER_CLASS}]){escaped}(?![{LETTER_CLASS}])\s*[:：]?", flags)
    for match in pattern.finditer(line):
        prefix = line[: match.start()].lower()
        followed_by_colon = match.group(0).rstrip().endswith((":", "："))
        at_start = not re.search(rf"[{LETTER_CLASS}0-9]", prefix)
        if not (followed_by_colon or at_start):
            continue
        if label in ("name", "নাম") and any(word in prefix for word in RELATION_WORDS):
            continue
        return match
    return None


def script_share(value: str, script: str) -> float:
    bengali = sum(1 for char in value if "ঀ" <= char <= "৿")
    latin = sum(1 for char in value if char.isascii() and char.isalpha())
    letters = bengali + latin
    if not letters:
        return 0.0
    return (bengali if script == "ben" else latin) / letters


def find_labeled_value(lines: list[str], labels: list[str], script: str) -> FieldResult:
    for index, line in enumerate(lines):
        match = next((found for label in labels if (found := label_match(line, label))), None)
        if not match:
            continue
        inline = normalize_space(trim_at_next_label(line[match.end():])).strip(":： ")
        if inline:
            candidate, source = inline, "label-inline"
        else:
            candidate, source = following_value(lines, index + 1), "label-next-line"
        if not candidate:
            continue
        if script_share(candidate, script) < 0.6:
            return empty_field(source=f"{source}-script-mismatch")
        return FieldResult(raw_value=candidate, confidence=0.55, needs_review=True, source=source)
    return empty_field()


def trim_at_next_label(value: str) -> str:
    next_label = None
    for _field, labels, _script in LABELED_FIELDS:
        for label in labels:
            match = label_match(value, label)
            if match and (next_label is None or match.start() < next_label):
                next_label = match.start()
    for pattern in (r"\bdate\s+of\s+birth\b\s*[:：]?", r"\b(?:id|nid)\s*(?:no|number)?\b\s*[:：]?"):
        match = re.search(pattern, value, re.IGNORECASE)
        if match and (next_label is None or match.start() < next_label):
            next_label = match.start()
    return value[:next_label] if next_label is not None else value


def following_value(lines: list[str], start: int) -> str | None:
    values = []
    for line in lines[start : start + 2]:
        value = normalize_space(line)
        if not value:
            continue
        first_word = re.split(r"[\s:：]+", value.lower(), maxsplit=1)[0]
        if first_word in LABEL_WORDS or re.search(r"\d", value):
            break
        values.append(value)
        break
    return normalize_space(" ".join(values)) if values else None
