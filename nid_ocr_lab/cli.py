from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nid_ocr_lab.engines.tesseract import TesseractEngine, ocr_result_to_json
from nid_ocr_lab.evaluation import evaluate_fields, summarize
from nid_ocr_lab.models import BoundingBox, OCRResult, OCRTextBlock
from nid_ocr_lab.pipeline import OCRPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangladesh NID OCR R&D harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse-json", help="Parse one OCR JSON file")
    parse_parser.add_argument("ocr_json", type=Path)

    tess_parser = subparsers.add_parser("ocr-tesseract", help="Run Tesseract and emit OCR JSON")
    tess_parser.add_argument("image", type=Path)
    tess_parser.add_argument("--lang", default="eng", help="Tesseract language string, for example eng, ben, or eng+ben")
    tess_parser.add_argument("--preprocessing", default=None)
    tess_parser.add_argument("--psm", type=int, default=None)
    tess_parser.add_argument("--output", type=Path, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate OCR JSON outputs against a dataset manifest")
    eval_parser.add_argument("ocr_output_dir", type=Path)
    eval_parser.add_argument("manifest", type=Path)

    args = parser.parse_args()
    if args.command == "parse-json":
        result = OCRPipeline().parse_ocr_result(load_ocr_json(args.ocr_json))
        print(json.dumps(nid_data_to_json(result), indent=2, ensure_ascii=False))
    elif args.command == "ocr-tesseract":
        languages = args.lang.split("+")
        result = TesseractEngine().recognize(
            args.image,
            languages,
            preprocessing=args.preprocessing,
            psm=args.psm,
        )
        payload = ocr_result_to_json(result)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.command == "evaluate":
        report = evaluate_dir(args.ocr_output_dir, args.manifest)
        print(json.dumps(report, indent=2, ensure_ascii=False))


def evaluate_dir(ocr_output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = OCRPipeline()
    sample_reports = []
    all_metrics = []

    for sample in manifest.get("samples", []):
        sample_id = sample["id"]
        for ocr_file in sorted(ocr_output_dir.glob(f"{sample_id}*.json")):
            ocr = load_ocr_json(ocr_file)
            parsed = pipeline.parse_ocr_result(ocr)
            metrics = evaluate_fields(parsed, sample.get("ground_truth", {}))
            all_metrics.extend(metrics)
            sample_reports.append(
                {
                    "sample_id": sample_id,
                    "ocr_file": str(ocr_file),
                    "engine": ocr.engine,
                    "language": ocr.language,
                    "preprocessing": ocr.preprocessing,
                    "latency_ms": ocr.latency_ms,
                    "summary": summarize(metrics),
                }
            )

    return {
        "summary": summarize(all_metrics),
        "samples": sample_reports,
    }


def load_ocr_json(path: Path) -> OCRResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = [load_block(block) for block in data.get("blocks", [])]
    full_text = data.get("full_text") or "\n".join(block.text for block in blocks)
    return OCRResult(
        blocks=blocks,
        full_text=full_text,
        engine=data.get("engine", "unknown"),
        language=data.get("language"),
        preprocessing=data.get("preprocessing"),
        latency_ms=data.get("latency_ms"),
        metadata=data.get("metadata", {}),
    )


def load_block(data: dict[str, Any]) -> OCRTextBlock:
    bbox = data.get("bounding_box")
    return OCRTextBlock(
        text=data.get("text", ""),
        bounding_box=BoundingBox(**bbox) if bbox else None,
        confidence=data.get("confidence"),
    )


def nid_data_to_json(result: Any) -> dict[str, Any]:
    output = {"raw_text": result.raw_text}
    for key, value in result.__dict__.items():
        if key == "raw_text":
            continue
        output[key] = {
            "raw_value": value.raw_value,
            "corrected_value": value.corrected_value,
            "value": value.value,
            "confidence": value.confidence,
            "needs_review": value.needs_review,
            "source": value.source,
        }
    return output


if __name__ == "__main__":
    main()
