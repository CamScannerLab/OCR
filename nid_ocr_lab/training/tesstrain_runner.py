from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from nid_ocr_lab.engines.tessdata import BEST_DIR, CUSTOM_DIR, get_variant
from nid_ocr_lab.engines.tesseract import TesseractEngine
from nid_ocr_lab.evaluation import levenshtein
from nid_ocr_lab.training.dataset import (
    NAME_PATTERN,
    PROJECT_ROOT,
    dataset_dir,
    line_records,
    load_split,
    normalize_gt_text,
)

TESSTRAIN_DIR = PROJECT_ROOT / "training" / "tesstrain"
TRAINING_VENV_PYTHON = PROJECT_ROOT / "training" / ".venv" / "bin" / "python"
REPORTS_DIR = PROJECT_ROOT / "benchmark" / "reports"


def gnu_make() -> str:
    """tesstrain's Makefile uses $(file ...), which needs GNU make >= 4.2 (macOS ships 3.81)."""
    for candidate in ("gmake", "make"):
        path = shutil.which(candidate)
        if not path:
            continue
        output = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
        match = re.search(r"GNU Make (\d+)\.(\d+)", output)
        if match and (int(match.group(1)), int(match.group(2))) >= (4, 2):
            return path
    raise RuntimeError("tesstrain needs GNU make >= 4.2. Install it with `brew install make` (provides gmake).")


def tesstrain_path(path: Path) -> str:
    # tesstrain recipes pass paths to bash unquoted; the project lives under "R&D", so hand make
    # only paths relative to the tesstrain checkout, which contain no shell metacharacters.
    return os.path.relpath(path, TESSTRAIN_DIR)


def run_make(make: str, target: str, variables: dict[str, str], log_path: Path) -> None:
    command = [make, target, *(f"{key}={value}" for key, value in variables.items())]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, cwd=TESSTRAIN_DIR, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"`make {target}` failed (exit {completed.returncode}); see {log_path}")


def train(
    dataset: str,
    model: str,
    start_model: str = "ben",
    max_iterations: int = 3000,
    learning_rate: float = 0.0001,
    lang_type: str = "Indic",
    psm: int = 7,
    base_variant: str = "best",
) -> dict:
    if not NAME_PATTERN.match(model):
        raise ValueError("Model name may contain only letters, digits, '_' and '-'")
    if not (TESSTRAIN_DIR / "Makefile").is_file():
        raise RuntimeError("tesstrain is missing; run scripts/fetch_tessdata.sh first")
    if not (TESSTRAIN_DIR / "data" / "langdata" / "radical-stroke.txt").is_file():
        raise RuntimeError("tesstrain langdata is missing; run scripts/fetch_tessdata.sh first")
    make = gnu_make()
    if not TRAINING_VENV_PYTHON.is_file():
        raise RuntimeError("training/.venv is missing; run scripts/fetch_tessdata.sh to create it")
    base = get_variant(base_variant)
    if not base.model_path(start_model).is_file():
        raise RuntimeError(f"Start model {start_model} not found in {base.directory} (fast models cannot be fine-tuned)")
    split = load_split(dataset)

    data_dir = TESSTRAIN_DIR / "data"
    output_dir = data_dir / model
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_link = data_dir / f"{model}-ground-truth"
    if gt_link.is_symlink() or gt_link.exists():
        if not gt_link.is_symlink():
            raise RuntimeError(f"{gt_link} exists and is not a symlink; remove it manually")
        gt_link.unlink()
    gt_link.symlink_to(dataset_dir(dataset), target_is_directory=True)
    # A failed helper leaves zero-byte .box/.lstmf files that make would treat as up to date.
    for pattern in ("*.box", "*.lstmf"):
        for stale in dataset_dir(dataset).glob(pattern):
            if stale.stat().st_size == 0:
                stale.unlink()

    log_path = output_dir / "nid_train.log"
    variables = {
        "MODEL_NAME": model,
        "START_MODEL": start_model,
        "TESSDATA": tesstrain_path(base.directory),
        "GROUND_TRUTH_DIR": f"data/{model}-ground-truth",
        "LANG_TYPE": lang_type,
        "PSM": str(psm),
        "MAX_ITERATIONS": str(max_iterations),
        "LEARNING_RATE": str(learning_rate),
        # tesstrain's box/list helpers need python-bidi and Pillow from the isolated training venv.
        "PY_CMD": tesstrain_path(TRAINING_VENV_PYTHON),
    }
    run_make(make, "lists", variables, log_path)

    # Replace tesstrain's random line split with the card-level split.
    def lstmf_list(stems: list[str]) -> list[str]:
        return [f"data/{model}-ground-truth/{stem}.lstmf" for stem in stems if (gt_link / f"{stem}.lstmf").is_file()]

    train_lines, eval_lines = lstmf_list(split["train"]), lstmf_list(split["eval"])
    if not train_lines or not eval_lines:
        raise RuntimeError("Split produced no .lstmf files for train or eval; re-run training-split")
    (output_dir / "list.train").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "list.eval").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    run_make(make, "training", variables, log_path)

    built = data_dir / f"{model}.traineddata"
    if not built.is_file():
        raise RuntimeError(f"Training finished but {built} was not produced; see {log_path}")
    target = CUSTOM_DIR / model
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, target / f"{model}.traineddata")
    for companion in ("eng", "ben", "osd"):
        link = target / f"{companion}.traineddata"
        source = BEST_DIR / f"{companion}.traineddata"
        if source.exists() and not link.exists():
            link.symlink_to(source.resolve())
    info = {
        "model": model,
        "dataset": dataset,
        "start_model": f"{base.id}:{start_model}",
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "lang_type": lang_type,
        "psm": psm,
        "train_lines": len(train_lines),
        "eval_lines": len(eval_lines),
        "split_seed": split.get("seed"),
        "eval_cards": split.get("eval_cards"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
    }
    (target / "model.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def evaluate_models(dataset: str, model_specs: list[str], split_name: str = "eval", psm: int = 7) -> dict:
    """Compare models on identical held-out line crops. Spec format: '<variant>:<lang>', e.g. 'best:ben'."""
    split = load_split(dataset)
    stems = set(split[split_name])
    records = [record for record in line_records(dataset) if record["stem"] in stems]
    if not records:
        raise ValueError(f"No {split_name} lines in dataset {dataset}")
    engine = TesseractEngine()
    report = {
        "dataset": dataset,
        "split": split_name,
        "lines": len(records),
        "cards": sorted({record["card_id"] for record in records}),
        "psm": psm,
        "created": datetime.now().isoformat(timespec="seconds"),
        "models": [],
    }
    for spec in model_specs:
        variant, _, language = spec.partition(":")
        if not language:
            raise ValueError(f"Model spec must be '<variant>:<lang>', got {spec}")
        edits = chars = exact = 0
        per_script: dict[str, dict[str, int]] = {}
        latency = 0.0
        samples = []
        for record in records:
            ocr = engine.recognize(record["image"], language.split("+"), psm=psm, variant=variant)
            predicted = normalize_gt_text(ocr.full_text)
            expected = record["text"]
            distance = levenshtein(predicted, expected)
            edits += distance
            chars += len(expected)
            exact += int(predicted == expected)
            latency += ocr.latency_ms or 0.0
            bucket = per_script.setdefault(record["script"], {"lines": 0, "exact": 0, "edits": 0, "chars": 0})
            bucket["lines"] += 1
            bucket["exact"] += int(predicted == expected)
            bucket["edits"] += distance
            bucket["chars"] += len(expected)
            if predicted != expected:
                samples.append({"stem": record["stem"], "expected": expected, "predicted": predicted})
        report["models"].append(
            {
                "spec": spec,
                "cer": edits / chars if chars else None,
                "line_exact_accuracy": exact / len(records),
                "mean_latency_ms": latency / len(records),
                "per_script": {
                    script: {
                        "lines": bucket["lines"],
                        "cer": bucket["edits"] / bucket["chars"] if bucket["chars"] else None,
                        "line_exact_accuracy": bucket["exact"] / bucket["lines"],
                    }
                    for script, bucket in per_script.items()
                },
                "errors": samples[:50],
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"tesseract_eval_{dataset}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(path)
    return report
