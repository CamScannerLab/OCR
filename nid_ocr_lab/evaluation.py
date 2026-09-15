from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

from nid_ocr_lab.models import FIELD_NAMES, NIDData


@dataclass(frozen=True)
class FieldMetric:
    field: str
    expected: str | None
    actual: str | None
    exact_match: bool
    evaluated: bool = True
    cer: float | None = None


def evaluate_fields(result: NIDData, ground_truth: dict[str, Any]) -> list[FieldMetric]:
    """Only fields annotated in the ground truth are scored; unlabeled fields are reported, not counted."""
    metrics: list[FieldMetric] = []
    for field in FIELD_NAMES:
        expected = normalize_value(ground_truth.get(field))
        actual = normalize_value(getattr(result, field).value)
        evaluated = field in ground_truth
        metrics.append(
            FieldMetric(
                field=field,
                expected=expected,
                actual=actual,
                exact_match=evaluated and expected == actual,
                evaluated=evaluated,
                cer=character_error_rate(actual or "", expected) if evaluated and expected else None,
            )
        )
    return metrics


def summarize(metrics: list[FieldMetric]) -> dict[str, Any]:
    scored = [metric for metric in metrics if metric.evaluated]
    total = len(scored)
    correct = sum(1 for metric in scored if metric.exact_match)
    per_field: dict[str, dict[str, Any]] = {}
    for metric in scored:
        entry = per_field.setdefault(metric.field, {"total": 0, "correct": 0, "edits_ratio_sum": 0.0, "with_text": 0})
        entry["total"] += 1
        entry["correct"] += int(metric.exact_match)
        if metric.cer is not None:
            entry["edits_ratio_sum"] += metric.cer
            entry["with_text"] += 1
    return {
        "fields_total": total,
        "fields_correct": correct,
        "fields_unlabeled": len(metrics) - total,
        "field_exact_accuracy": correct / total if total else None,
        "per_script": {
            script: script_accuracy(scored, script) for script in ("bangla", "english", "numeric")
        },
        "per_field": {
            field: {
                "total": entry["total"],
                "exact_accuracy": entry["correct"] / entry["total"],
                "mean_cer": entry["edits_ratio_sum"] / entry["with_text"] if entry["with_text"] else None,
            }
            for field, entry in per_field.items()
        },
        "failures": [
            {"field": metric.field, "expected": metric.expected, "actual": metric.actual, "cer": metric.cer}
            for metric in scored
            if not metric.exact_match
        ],
    }


def field_script(field: str) -> str:
    if field.endswith("_bangla"):
        return "bangla"
    if field.endswith("_english"):
        return "english"
    return "numeric"


def script_accuracy(metrics: list[FieldMetric], script: str) -> dict[str, Any]:
    selected = [metric for metric in metrics if field_script(metric.field) == script]
    correct = sum(1 for metric in selected if metric.exact_match)
    return {"total": len(selected), "exact_accuracy": correct / len(selected) if selected else None}


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(unicodedata.normalize("NFC", str(value)).strip().split())
    if not normalized:
        return None
    return normalized.upper() if normalized.isascii() else normalized


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def character_error_rate(predicted: str, expected: str) -> float:
    """Code-point edit distance over NFC text divided by the reference length."""
    predicted = unicodedata.normalize("NFC", predicted)
    expected = unicodedata.normalize("NFC", expected)
    if not expected:
        return 0.0 if not predicted else 1.0
    return levenshtein(predicted, expected) / len(expected)
