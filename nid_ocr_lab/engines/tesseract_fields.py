from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageOps

from nid_ocr_lab.engines.tessdata import get_variant
from nid_ocr_lab.engines.tesseract import TesseractEngine
from nid_ocr_lab.engines.tesseract_reader import line_reader
from nid_ocr_lab.models import BoundingBox, OCRResult, OCRTextBlock
from nid_ocr_lab.parsers.nid_parser import script_share

# (psm, white border px): single line first, then raw line without border when the first read is weak.
READ_ATTEMPTS = ((7, 12), (13, 0))
GOOD_SCORE = 0.75  # confidence x script share that skips the retry
CROP_HEIGHT = 64  # crops are only ever upscaled to this; the LSTM normalizes line height itself
CROP_DPI = 300
MIN_SCRIPT_SHARE = 0.6
DIGIT_WHITELIST = {"tessedit_char_whitelist": "0123456789"}
LABEL_PUNCTUATION = ":;：|[](){}'\"‘’“”.,/\\-_"


@dataclass(frozen=True)
class NIDRow:
    field: str
    label: str
    aliases: tuple[str, ...]  # normalized first label word, including misreads seen in real runs
    script: str  # "ben", "eng" or "digits"
    label_words: int  # label length when no word ends with ":"
    name_row: bool


# Card order. Both NID layouts print these rows top to bottom in this order.
NID_ROWS = (
    NIDRow("name_bangla", "নাম", ("নাম", "লাম"), "ben", 1, True),
    NIDRow("name_english", "Name", ("name", "neme", "narne", "nane"), "eng", 1, True),
    NIDRow("father_name_bangla", "পিতা", ("পিতা",), "ben", 1, True),
    NIDRow("mother_name_bangla", "মাতা", ("মাতা",), "ben", 1, True),
    NIDRow("date_of_birth", "Date of Birth", ("date", "dete", "dale", "pete"), "eng", 3, False),
    NIDRow("nid_number", "ID NO", ("id", "nid", "1d", "10", "lb"), "digits", 2, False),
)


@dataclass
class RowBox:
    row: NIDRow
    source: str  # "page" or "interpolated"
    left: float
    top: float
    right: float
    bottom: float
    label_right: float
    value_left: float | None
    page_value: str
    page_confidence: float | None
    line_key: tuple[int, ...] | None
    value_line_key: tuple[int, ...] | None = None  # set when the value sits on the line below the label


def refine_nid_fields(
    engine: TesseractEngine,
    image_path: Path,
    page: OCRResult,
    variant: str | None,
    workdir: Path,
    languages: dict[str, list[str]] | None = None,
    reader: Any | None = None,
) -> OCRResult:
    """Re-read each NID row with one script at PSM 7 and restore rows the page pass dropped."""
    languages = {"ben": ["ben"], "eng": ["eng"], "digits": ["eng"], **(languages or {})}
    reader = reader or line_reader(engine)
    model = get_variant(variant)
    lines = group_lines(page.metadata.get("words") or [])
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    rows = locate_rows(lines, image.size)

    fields: list[dict[str, Any]] = []
    latency = page.latency_ms or 0.0
    calls = 0
    claimed: set[tuple[int, ...]] = set()
    entries: list[tuple[float, str, OCRTextBlock]] = []
    for index, box in enumerate(rows):
        for key in (box.line_key, box.value_line_key):
            if key is not None:
                claimed.add(key)
        value_left = box.value_left if box.value_left is not None else box.label_right
        region = (value_left, box.top, box.right, box.bottom)
        candidates = []
        if box.page_value:
            candidates.append(candidate("page", box.page_value, box.page_confidence, box.row))
        # Tiny line crops flip between junk and correct with PSM and border, so a weak first read gets one retry.
        for attempt, (psm, border) in enumerate(READ_ATTEMPTS):
            if attempt and max((item["score"] for item in candidates if item["source"] != "page"), default=0) >= GOOD_SCORE:
                break
            crop_path = workdir / f"field-{index}-{box.row.field}-psm{psm}.png"
            save_line_crop(image, region, crop_path, border=border)
            started = time.perf_counter()
            read = reader.read(
                crop_path,
                model.directory,
                languages[box.row.script],
                psm,
                DIGIT_WHITELIST if box.row.script == "digits" else None,
            )
            calls += 1
            latency += (time.perf_counter() - started) * 1000
            candidates.append(candidate(f"reread-psm{psm}", clean_value(read.text), read.confidence, box.row))
        best = max(candidates, key=lambda item: item["score"], default=None)
        if best and best["score"] > 0:
            text, confidence = best["text"], best["confidence"]
            source = "page" if best["source"] == "page" else ("interpolated" if box.source == "interpolated" else "reread")
        elif box.page_value:
            text, source, confidence = box.page_value, "page", box.page_confidence
        else:
            text, source, confidence = "", "missing", None
        fields.append(
            {
                "field": box.row.field,
                "label": box.row.label,
                "languages": languages[box.row.script],
                "source": source,
                "text": text,
                "confidence": confidence,
                "candidates": candidates,
                "box": {"x": region[0], "y": region[1], "width": region[2] - region[0], "height": region[3] - region[1]},
            }
        )
        if text:
            block = OCRTextBlock(
                text=text,
                bounding_box=BoundingBox(x=region[0], y=region[1], width=region[2] - region[0], height=region[3] - region[1]),
                confidence=confidence,
            )
            entries.append((box.top, f"{box.row.label}: {text}", block))

    for key, words in lines.items():
        if key in claimed:
            continue
        block = line_block(words)
        entries.append((block.bounding_box.y, block.text, block))
    entries.sort(key=lambda entry: entry[0])

    return OCRResult(
        blocks=[entry[2] for entry in entries],
        full_text="\n".join(entry[1] for entry in entries),
        engine=page.engine,
        language=page.language,
        preprocessing=f"{page.preprocessing}_fields" if page.preprocessing else "fields",
        latency_ms=latency,
        metadata={
            **page.metadata,
            "page_full_text": page.full_text,
            "fields": fields,
            "field_calls": calls,
            "field_reader": reader.name,
        },
    )


def group_lines(words: list[dict[str, Any]]) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    """Words keep Tesseract's reading order; lines are ordered top to bottom."""
    lines: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for word in words:
        lines.setdefault(tuple(word["line_key"]), []).append(word)
    return dict(sorted(lines.items(), key=lambda item: min(word["bounding_box"]["y"] for word in item[1])))


def normalize_label_word(text: str) -> str:
    return text.strip(LABEL_PUNCTUATION + " ").lower()


def label_length(words: list[dict[str, Any]], start: int, row: NIDRow) -> int:
    for offset, word in enumerate(words[start : start + row.label_words]):
        if word["text"].rstrip().endswith((":", ";", "：")):
            return offset + 1
    return row.label_words


def match_row(words: list[dict[str, Any]], row: NIDRow) -> int | None:
    """Index of the label's first word: word 0, or word 1 after a short junk fragment."""
    for index, word in enumerate(words[:2]):
        if index == 1 and len(normalize_label_word(words[0]["text"])) > 3:
            break
        if normalize_label_word(word["text"]) in row.aliases:
            return index
    return None


def value_below(ordered: list[tuple[tuple[int, ...], list[dict[str, Any]]]], position: int):
    """The next line, when it holds this row's value: some NID cards print the label on its own line."""
    if position + 1 >= len(ordered):
        return None, []
    key, words = ordered[position + 1]
    if any(match_row(words, row) is not None for row in NID_ROWS):
        return None, []  # the next line is another label, so this row has no value
    return key, words


def box_edges(words: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    boxes = [word["bounding_box"] for word in words]
    return (
        min(box["x"] for box in boxes),
        min(box["y"] for box in boxes),
        max(box["x"] + box["width"] for box in boxes),
        max(box["y"] + box["height"] for box in boxes),
    )


def locate_rows(lines: dict[tuple[int, ...], list[dict[str, Any]]], image_size: tuple[int, int]) -> list[RowBox]:
    width, _height = image_size
    ordered = list(lines.items())
    found: dict[int, RowBox] = {}
    claimed: set[tuple[int, ...]] = set()
    last_top = float("-inf")
    for row_index, row in enumerate(NID_ROWS):
        for position, (key, words) in enumerate(ordered):
            left, top, right, bottom = box_edges(words)
            if top <= last_top or key in claimed:
                continue
            start = match_row(words, row)
            if start is None:
                continue
            count = label_length(words, start, row)
            label_words = words[start : start + count]
            value_words = words[start + count :]
            label_right = box_edges(label_words)[2]
            height = bottom - top
            value_key = None
            if value_words:
                value_left = max(label_right + 1, value_words[0]["bounding_box"]["x"] - 0.3 * height)
            else:
                # Layout with the label on its own line: the value is the line below it.
                value_key, value_words = value_below(ordered, position)
                if value_words:
                    left, top, right, bottom = box_edges(value_words)
                    value_left = left - 0.15 * (bottom - top)
                else:
                    value_left = None
            found[row_index] = RowBox(
                row=row,
                source="page",
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                label_right=label_right,
                value_left=value_left,
                page_value=clean_value(" ".join(word["text"] for word in value_words)),
                page_confidence=mean_word_confidence(value_words),
                line_key=key,
                value_line_key=value_key,
            )
            claimed.update({key} | ({value_key} if value_key else set()))
            last_top = top
            break
    if not found:
        return []

    heights = [box.bottom - box.top for box in found.values()]
    line_height = median(heights)
    indexes = sorted(found)
    steps = [
        (found[below].top - found[above].top) / (below - above) for above, below in zip(indexes, indexes[1:])
    ]
    pitch = median(steps) if steps else None
    name_boxes = [box for box in found.values() if box.row.name_row]
    name_right = max((box.right for box in name_boxes), default=None)
    # Name-row values share one column on the card, so found rows place the missing ones.
    name_value_lefts = [box.value_left for box in name_boxes if box.value_left is not None]
    name_label_rights = [box.label_right for box in name_boxes]

    rows: list[RowBox] = []
    for row_index, row in enumerate(NID_ROWS):
        if row_index in found:
            box = found[row_index]
        else:
            above = max((index for index in indexes if index < row_index), default=None)
            below = min((index for index in indexes if index > row_index), default=None)
            if above is None or below is None:
                continue
            fraction = (row_index - above) / (below - above)
            top = found[above].top + (found[below].top - found[above].top) * fraction
            label_right = median(name_label_rights) if name_label_rights else found[above].label_right
            value_left = median(name_value_lefts) if row.name_row and name_value_lefts else label_right + 0.3 * line_height
            box = RowBox(
                row=row,
                source="interpolated",
                left=found[above].left,
                top=top,
                right=max(found[above].right, found[below].right),
                bottom=top + line_height,
                label_right=label_right,
                value_left=value_left,
                page_value="",
                page_confidence=None,
                line_key=None,
            )
        height = box.bottom - box.top
        right = max(box.right, name_right) if row.name_row and name_right is not None else box.right
        # Pad 25% vertically, but never past half the row spacing, so a crop cannot take in the next row.
        center = (box.top + box.bottom) / 2
        half = min(0.75 * height, 0.475 * pitch) if pitch else 0.75 * height
        rows.append(replace(box, right=min(width, right + height), top=center - half, bottom=center + half))
    return rows


def save_line_crop(image: Image.Image, region: tuple[float, float, float, float], target: Path, border: int) -> None:
    left = max(0, int(region[0]))
    top = max(0, int(region[1]))
    right = min(image.width, int(round(region[2])))
    bottom = min(image.height, int(round(region[3])))
    if right <= left or bottom <= top:
        raise ValueError(f"Field region {region} is outside the OCR image")
    crop = image.crop((left, top, right, bottom))
    scale = max(1.0, CROP_HEIGHT / crop.height)
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    ImageOps.expand(crop, border=border, fill=(255, 255, 255)).save(target, dpi=(CROP_DPI, CROP_DPI))


def clean_value(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip(LABEL_PUNCTUATION + " ")


def script_fit(text: str, script: str) -> float:
    """Share of the value in the row's script; 0 when it cannot be that field at all."""
    if not text:
        return 0.0
    if script == "digits":
        return 1.0 if len(re.sub(r"\D", "", text)) >= 8 else 0.0
    letters = sum(1 for char in text if char.isalpha())
    share = script_share(text, script)
    return share if letters >= 2 and share >= MIN_SCRIPT_SHARE else 0.0


def candidate(source: str, text: str, confidence: float | None, row: NIDRow) -> dict[str, Any]:
    if row.name_row:
        # Names carry no digits or lone symbols; those tokens are emblem and border specks.
        text = " ".join(token for token in text.split() if any(char.isalpha() for char in token))
    return {
        "source": source,
        "text": text,
        "confidence": confidence,
        "score": round((confidence or 0.0) * script_fit(text, row.script), 4),
    }


def mean_word_confidence(words: list[dict[str, Any]]) -> float | None:
    confidences = [word["confidence"] for word in words if word.get("confidence") is not None]
    return sum(confidences) / len(confidences) if confidences else None


def line_block(words: list[dict[str, Any]]) -> OCRTextBlock:
    left, top, right, bottom = box_edges(words)
    confidences = [word["confidence"] for word in words if word.get("confidence") is not None]
    return OCRTextBlock(
        text=" ".join(word["text"] for word in words),
        bounding_box=BoundingBox(x=left, y=top, width=right - left, height=bottom - top),
        confidence=sum(confidences) / len(confidences) if confidences else None,
    )
