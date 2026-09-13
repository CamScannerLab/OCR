from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nid_ocr_lab.models import FIELD_NAMES, NIDData


@dataclass(frozen=True)
class FieldMetric:
    field: str
    expected: str | None
    actual: str | None
    exact_match: bool


def evaluate_fields(result: NIDData, ground_truth: dict[str, Any]) -> list[FieldMetric]:
    metrics: list[FieldMetric] = []
    for field in FIELD_NAMES:
        expected = normalize_value(ground_truth.get(field))
        actual = normalize_value(getattr(result, field).value)
        metrics.append(
            FieldMetric(
                field=field,
                expected=expected,
                actual=actual,
                exact_match=expected == actual if expected is not None else actual is None,
            )
        )
    return metrics


def summarize(metrics: list[FieldMetric]) -> dict[str, Any]:
    total = len(metrics)
    correct = sum(1 for metric in metrics if metric.exact_match)
    return {
        "fields_total": total,
        "fields_correct": correct,
        "field_exact_accuracy": correct / total if total else 0.0,
        "failures": [
            {
                "field": metric.field,
                "expected": metric.expected,
                "actual": metric.actual,
            }
            for metric in metrics
            if not metric.exact_match
        ],
    }


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    return normalized.upper() if normalized.isascii() else normalized

