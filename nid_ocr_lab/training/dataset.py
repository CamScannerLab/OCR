from __future__ import annotations

import json
import random
import re
import shutil
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "benchmark" / "generated" / "runs"
GROUND_TRUTH_ROOT = PROJECT_ROOT / "benchmark" / "training" / "ground-truth"
MAX_RUNS = 50
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
RUN_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")


def runs_root() -> Path:
    return RUNS_ROOT


def ground_truth_root() -> Path:
    return GROUND_TRUTH_ROOT


# --- OCR run artifacts -------------------------------------------------------

def save_run(input_image: Path, ocr_json: dict, context: dict) -> str:
    """Keep the exact image Tesseract read so line boxes can be cropped later."""
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    directory = runs_root() / run_id
    directory.mkdir(parents=True)
    shutil.copyfile(input_image, directory / "input.png")
    (directory / "ocr.json").write_text(json.dumps(ocr_json, ensure_ascii=False), encoding="utf-8")
    (directory / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    prune_runs()
    return run_id


def prune_runs(keep: int = MAX_RUNS) -> None:
    root = runs_root()
    if not root.is_dir():
        return
    runs = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    for stale in runs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)


def run_dir(run_id: str) -> Path:
    if not RUN_ID_PATTERN.match(run_id or ""):
        raise ValueError("Invalid run id")
    directory = runs_root() / run_id
    if not (directory / "input.png").is_file():
        raise ValueError("OCR run not found (runs are pruned after the newest 50)")
    return directory


def padded_box(bbox: dict, image_size: tuple[int, int], horizontal: int = 8, vertical_ratio: float = 0.15) -> tuple[int, int, int, int]:
    width, height = image_size
    x, y = float(bbox["x"]), float(bbox["y"])
    w, h = float(bbox["width"]), float(bbox["height"])
    pad_y = max(2, round(h * vertical_ratio))
    left = max(0, int(x) - horizontal)
    top = max(0, int(y) - pad_y)
    right = min(width, int(round(x + w)) + horizontal)
    bottom = min(height, int(round(y + h)) + pad_y)
    if right <= left or bottom <= top:
        raise ValueError("Line box is outside the OCR image")
    return left, top, right, bottom


def crop_line(run_id: str, bbox: dict) -> Image.Image:
    with Image.open(run_dir(run_id) / "input.png") as image:
        return image.crop(padded_box(bbox, image.size)).copy()


# --- Ground-truth lines ------------------------------------------------------

def normalize_gt_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value or "").strip("_")
    return cleaned[:60] or fallback


def guess_script(text: str) -> str:
    bengali = sum(1 for char in text if "ঀ" <= char <= "৿")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    digits = sum(1 for char in text if char.isdigit())
    if bengali and bengali >= latin:
        return "ben"
    if latin:
        return "eng"
    return "digits" if digits else "eng"


def dataset_dir(dataset: str) -> Path:
    if not NAME_PATTERN.match(dataset or ""):
        raise ValueError("Dataset name may contain only letters, digits, '_' and '-' (max 40)")
    return ground_truth_root() / dataset


def save_training_lines(dataset: str, run_id: str, card_id: str, lines: list[dict]) -> list[dict]:
    target = dataset_dir(dataset)
    target.mkdir(parents=True, exist_ok=True)
    directory = run_dir(run_id)
    context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
    card = safe_name(card_id, "card")
    saved = []
    with Image.open(directory / "input.png") as image:
        for line in lines:
            text = normalize_gt_text(line.get("text", ""))
            if not text:
                raise ValueError(f"Line {line.get('index')} has empty text")
            index = int(line["index"])
            box = padded_box(line["bounding_box"], image.size)
            stem = f"{card}__{run_id}__{index:03d}"
            image.crop(box).save(target / f"{stem}.png")
            (target / f"{stem}.gt.txt").write_text(text + "\n", encoding="utf-8")
            sidecar = {
                "card_id": card_id,
                "run_id": run_id,
                "line_index": index,
                "text": text,
                "ocr_text": line.get("ocr_text"),
                "script": line.get("script") or guess_script(text),
                "bounding_box": line["bounding_box"],
                "crop_box": list(box),
                "source": context,
                "saved": datetime.now().isoformat(timespec="seconds"),
            }
            (target / f"{stem}.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
            saved.append({"stem": stem, "text": text, "script": sidecar["script"]})
    return saved


def line_records(dataset: str) -> list[dict]:
    target = dataset_dir(dataset)
    records = []
    for gt in sorted(target.glob("*.gt.txt")):
        stem = gt.name.removesuffix(".gt.txt")
        image = target / f"{stem}.png"
        if not image.is_file():
            continue
        sidecar = target / f"{stem}.json"
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        records.append(
            {
                "stem": stem,
                "image": image,
                "text": gt.read_text(encoding="utf-8").strip(),
                "card_id": meta.get("card_id") or stem.split("__", 1)[0],
                "script": meta.get("script") or guess_script(gt.read_text(encoding="utf-8")),
            }
        )
    return records


def dataset_stats() -> list[dict]:
    root = ground_truth_root()
    if not root.is_dir():
        return []
    stats = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and NAME_PATTERN.match(path.name)):
        records = line_records(directory.name)
        scripts: dict[str, int] = {}
        for record in records:
            scripts[record["script"]] = scripts.get(record["script"], 0) + 1
        stats.append(
            {
                "dataset": directory.name,
                "lines": len(records),
                "cards": len({record["card_id"] for record in records}),
                "scripts": scripts,
            }
        )
    return stats


def split_by_card(records: list[dict], eval_ratio: float = 0.15, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Every line of a card lands on the same side, so eval never sees a training card."""
    cards = sorted({record["card_id"] for record in records})
    if len(cards) < 2:
        raise ValueError("Need lines from at least 2 different cards to make a train/eval split")
    random.Random(seed).shuffle(cards)
    eval_count = min(len(cards) - 1, max(1, round(len(cards) * eval_ratio)))
    eval_cards = set(cards[:eval_count])
    train = [record for record in records if record["card_id"] not in eval_cards]
    evaluation = [record for record in records if record["card_id"] in eval_cards]
    return train, evaluation


def write_split(dataset: str, eval_ratio: float = 0.15, seed: int = 0) -> dict:
    records = line_records(dataset)
    train, evaluation = split_by_card(records, eval_ratio, seed)
    target = dataset_dir(dataset)
    (target / "split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "eval_ratio": eval_ratio,
                "train": [record["stem"] for record in train],
                "eval": [record["stem"] for record in evaluation],
                "train_cards": sorted({record["card_id"] for record in train}),
                "eval_cards": sorted({record["card_id"] for record in evaluation}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "dataset": dataset,
        "train_lines": len(train),
        "eval_lines": len(evaluation),
        "train_cards": len({record["card_id"] for record in train}),
        "eval_cards": len({record["card_id"] for record in evaluation}),
    }


def load_split(dataset: str) -> dict:
    path = dataset_dir(dataset) / "split.json"
    if not path.is_file():
        raise ValueError(f"No split for {dataset}; run training-split first")
    return json.loads(path.read_text(encoding="utf-8"))
