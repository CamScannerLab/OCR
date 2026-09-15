from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from nid_ocr_lab.training.dataset import (
    PROJECT_ROOT,
    dataset_dir,
    ground_truth_root,
    guess_script,
    normalize_gt_text,
    save_training_lines,
)

DRAFTS_ROOT = PROJECT_ROOT / "benchmark" / "training" / "drafts"
TRAINING_ROOT = PROJECT_ROOT / "benchmark" / "training"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
SCRIPTS = ("ben", "eng", "digits", "all")
REVIEW_FILE = "REVIEW.tsv"


def drafts_root() -> Path:
    return DRAFTS_ROOT


def collect_images(paths: list[Path]) -> list[Path]:
    """Image files given directly or found recursively in folders, never the training data itself."""
    excluded = [TRAINING_ROOT.resolve(), drafts_root().resolve(), ground_truth_root().resolve()]
    found: set[Path] = set()
    for path in paths:
        path = Path(path).resolve()
        if path.is_dir():
            candidates = (item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            candidates = iter([path])
        else:
            raise ValueError(f"Not found: {path}")
        for item in candidates:
            if item.suffix.lower() not in IMAGE_SUFFIXES or any(item.is_relative_to(root) for root in excluded):
                continue
            # A dashboard upload keeps original.<ext> next to the page-NNN.png that is OCR'd; use only the page.
            if item.stem == "original" and (item.parent / "upload.json").is_file():
                continue
            found.add(item)
    return sorted(found)


def card_id_for(image: Path) -> str:
    """One id per physical card so training-split never puts one card on both sides."""
    if image.stem.startswith("page-") and (image.parent / "upload.json").is_file():
        return f"{image.parent.name}-p{image.stem.removeprefix('page-')}"
    return image.stem


def draft_lines(
    paths: list[Path],
    dataset: str = "nid_ben",
    *,
    filter_mode: str = "nid_ink",
    strategy: str = "fields",
    variant: str = "best",
    language: str = "ben+eng",
    rotation: str = "auto",
    psm: int = 6,
    script: str = "ben",
) -> dict:
    """OCR each image, then save every detected line crop with Tesseract's guess as a draft .gt.txt."""
    from nid_ocr_lab.dashboard.server import run_ocr_payload

    if script not in SCRIPTS:
        raise ValueError(f"script must be one of {', '.join(SCRIPTS)}")
    target = dataset_dir(dataset, drafts_root())
    images = collect_images(paths)
    if not images:
        raise ValueError("No images found in the given paths")

    summary = {"dataset": dataset, "images": len(images), "lines": 0, "skipped_script": 0, "errors": [], "drafts_dir": str(target)}
    for image in images:
        card_id = card_id_for(image)
        try:
            result = run_ocr_payload(
                {
                    "engine": "tesseract",
                    "image_path": str(image),
                    "input": "as_is",
                    "mode": filter_mode,
                    "language": language,
                    "rotation": rotation,
                    "tesseract_variant": variant,
                    "psm": psm,
                    "strategy": strategy,
                    "sample_id": card_id,
                }
            )
            lines = []
            for index, block in enumerate(result["ocr"]["blocks"]):
                text = normalize_gt_text(block.get("text") or "")
                if not text or not block.get("bounding_box"):
                    continue
                if script != "all" and guess_script(text) != script:
                    summary["skipped_script"] += 1
                    continue
                lines.append(
                    {
                        "index": index,
                        "text": text,
                        "ocr_text": text,
                        "ocr_confidence": block.get("confidence"),
                        "script": guess_script(text),
                        "bounding_box": block["bounding_box"],
                    }
                )
            if lines:
                save_training_lines(
                    dataset,
                    result["run_id"],
                    card_id,
                    lines,
                    root=drafts_root(),
                    extra={"status": "draft", "source_image": str(image)},
                )
            summary["lines"] += len(lines)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            summary["errors"].append({"image": str(image), "error": str(exc)})
    write_review(target)
    return summary


def promote_drafts(dataset: str = "nid_ben") -> dict:
    """Move corrected drafts into ground truth. Empty .gt.txt = not reviewed yet; delete a draft's files to discard it."""
    source = dataset_dir(dataset, drafts_root())
    target = dataset_dir(dataset)
    if not source.is_dir():
        raise ValueError(f"No drafts for {dataset} in {source}")
    summary = {"dataset": dataset, "promoted": 0, "edited": 0, "skipped_empty": 0, "skipped_existing": 0, "ground_truth_dir": str(target)}
    for gt in sorted(source.glob("*.gt.txt")):
        stem = gt.name.removesuffix(".gt.txt")
        files = {suffix: source / f"{stem}{suffix}" for suffix in (".png", ".gt.txt", ".json")}
        if not all(path.is_file() for path in files.values()):
            continue
        text = normalize_gt_text(gt.read_text(encoding="utf-8"))
        if not text:
            summary["skipped_empty"] += 1
            continue
        if any((target / path.name).exists() for path in files.values()):
            summary["skipped_existing"] += 1
            continue
        sidecar = json.loads(files[".json"].read_text(encoding="utf-8"))
        edited = text != normalize_gt_text(sidecar.get("ocr_text") or "")
        sidecar.update(
            {
                "status": "verified",
                "text": text,
                "script": guess_script(text),
                "edited": edited,
                "promoted": datetime.now().isoformat(timespec="seconds"),
            }
        )
        files[".gt.txt"].write_text(text + "\n", encoding="utf-8")
        files[".json"].write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        target.mkdir(parents=True, exist_ok=True)
        for path in files.values():
            shutil.move(str(path), target / path.name)
        summary["promoted"] += 1
        summary["edited"] += int(edited)
    write_review(source)
    if summary["promoted"]:
        summary["next"] = f"re-run: python -m nid_ocr_lab.cli training-split --dataset {dataset}"
    return summary


def write_review(directory: Path) -> None:
    """Checklist of the drafts still waiting for review, rebuilt from the files on disk."""
    rows = ["stem\timage\tguess\tconfidence\tsource_image"]
    for gt in sorted(directory.glob("*.gt.txt")) if directory.is_dir() else []:
        stem = gt.name.removesuffix(".gt.txt")
        sidecar_path = directory / f"{stem}.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
        confidence = sidecar.get("ocr_confidence")
        rows.append(
            "\t".join(
                [
                    stem,
                    f"{stem}.png",
                    (sidecar.get("ocr_text") or "").replace("\t", " "),
                    "" if confidence is None else f"{confidence:.2f}",
                    sidecar.get("source_image") or "",
                ]
            )
        )
    if directory.is_dir():
        (directory / REVIEW_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
