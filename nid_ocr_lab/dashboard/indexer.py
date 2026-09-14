from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on local environment
    Image = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SMART_SCAN_ROOT = Path("/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan")
DEFAULT_ROOTS = [
    SMART_SCAN_ROOT / "dataset",
    SMART_SCAN_ROOT / "training/data",
]

DASHBOARD_IMAGE_DIRS = [
    SMART_SCAN_ROOT / "dataset/composites/images",
    SMART_SCAN_ROOT / "dataset/incoming/_done",
    SMART_SCAN_ROOT / "dataset/incoming/_preview",
    SMART_SCAN_ROOT / "dataset/incoming/_review2",
    SMART_SCAN_ROOT / "dataset/new_dataset/_done",
    SMART_SCAN_ROOT / "dataset/new_dataset/_preview",
    SMART_SCAN_ROOT / "dataset/overlays",
    SMART_SCAN_ROOT / "dataset/raw",
    SMART_SCAN_ROOT / "training/data/train/images",
    SMART_SCAN_ROOT / "training/data/val/images",
]

DASHBOARD_ANNOTATION_DIRS = [
    SMART_SCAN_ROOT / "dataset/annotations",
    SMART_SCAN_ROOT / "dataset/incoming/_done",
    SMART_SCAN_ROOT / "dataset/new_dataset/_done",
]


@dataclass
class Sample:
    id: str
    source: str
    raw: str | None = None
    annotation: str | None = None
    mask: str | None = None
    overlay: str | None = None
    variants: dict[str, str] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)


def build_index(roots: list[Path] | None = None) -> list[dict]:
    samples: dict[str, Sample] = {}
    for root in roots or DEFAULT_ROOTS:
        if not root.exists():
            continue
        index_root(root, samples)
    return [asdict(sample) for sample in sorted(samples.values(), key=lambda item: item.id)]


def dataset_health(root: Path = SMART_SCAN_ROOT) -> dict:
    checks = {
        "dataset": pair_check(root / "dataset/raw", root / "dataset/masks", ".jpg", ".png"),
        "train": pair_check(root / "training/data/train/images", root / "training/data/train/masks", ".jpg", ".png"),
        "val": pair_check(root / "training/data/val/images", root / "training/data/val/masks", ".jpg", ".png"),
        "composites": pair_check(root / "dataset/composites/images", root / "dataset/composites/masks", ".jpg", ".png"),
    }
    return checks


def pair_check(image_dir: Path, mask_dir: Path, image_suffix: str, mask_suffix: str) -> dict:
    image_stems = file_stems(image_dir, image_suffix)
    mask_stems = file_stems(mask_dir, mask_suffix)
    paired = sorted(image_stems & mask_stems)
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)
    dimension_mismatches = dimension_check(image_dir, mask_dir, paired, image_suffix, mask_suffix)
    return {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "images": len(image_stems),
        "masks": len(mask_stems),
        "paired": len(paired),
        "missing_masks": missing_masks,
        "orphan_masks": orphan_masks,
        "dimension_mismatches": dimension_mismatches,
    }


def file_stems(path: Path, suffix: str) -> set[str]:
    if not path.exists():
        return set()
    return {item.stem for item in path.glob(f"*{suffix}") if item.is_file()}


def dimension_check(
    image_dir: Path,
    mask_dir: Path,
    stems: list[str],
    image_suffix: str,
    mask_suffix: str,
) -> list[dict]:
    if Image is None:
        return []
    mismatches = []
    for stem in stems:
        image_path = image_dir / f"{stem}{image_suffix}"
        mask_path = mask_dir / f"{stem}{mask_suffix}"
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image_size = image.size
                mask_size = mask.size
        except OSError:
            continue
        if image_size != mask_size:
            mismatches.append(
                {
                    "id": stem,
                    "image": image_size,
                    "mask": mask_size,
                }
            )
    return mismatches


def index_root(root: Path, samples: dict[str, Sample]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or not should_include_dashboard_file(path):
            continue
        if path.suffix.lower() == ".json":
            attach_annotation(path, samples)
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            attach_image(path, samples)


def should_include_dashboard_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return any(is_relative_to(path, directory) for directory in DASHBOARD_IMAGE_DIRS)
    if suffix == ".json":
        return any(is_relative_to(path, directory) for directory in DASHBOARD_ANNOTATION_DIRS)
    return False


def is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def attach_annotation(path: Path, samples: dict[str, Sample]) -> None:
    sample_id = path.stem
    sample = samples.setdefault(sample_id, Sample(id=sample_id, source=source_name(path)))
    sample.annotation = str(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    image_path = data.get("imagePath")
    if image_path:
        resolved = (path.parent / image_path).resolve()
        if resolved.exists():
            sample.images.setdefault("annotation image", str(resolved))
            if not sample.raw:
                sample.raw = str(resolved)


def attach_image(path: Path, samples: dict[str, Sample]) -> None:
    sample_id = path.stem
    if sample_id.startswith("sample-"):
        sample_id = sample_id.removeprefix("sample-")
    sample = samples.setdefault(sample_id, Sample(id=sample_id, source=source_name(path)))
    lower_parts = [part.lower() for part in path.parts]
    lower_name = path.name.lower()

    role = image_role(path)
    sample.images[role] = str(path)

    if "masks" in lower_parts:
        sample.mask = str(path)
    elif "overlays" in lower_parts or "overlay" in lower_name:
        sample.overlay = str(path)
    elif "raw" in lower_parts:
        sample.raw = str(path)
    elif "images" in lower_parts and not sample.raw:
        sample.raw = str(path)
    elif "assets" in lower_parts and not sample.raw:
        sample.raw = str(path)
    else:
        variant = path.parent.name.lower()
        sample.variants[variant] = str(path)


def image_role(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()

    if "training" in parts and "train" in parts and "images" in parts:
        return "train image"
    if "training" in parts and "train" in parts and "masks" in parts:
        return "train mask"
    if "training" in parts and "val" in parts and "images" in parts:
        return "validation/test image"
    if "training" in parts and "val" in parts and "masks" in parts:
        return "validation/test mask"
    if "baseline_masks" in parts and "masks" in parts:
        return "baseline edge/mask"
    if "masks" in parts:
        return "mask"
    if "overlays" in parts or "overlay" in name:
        return "overlay"
    if "raw" in parts:
        return "dataset/raw"
    if "incoming" in parts and "_done" in parts:
        return "incoming raw/done"
    if "incoming" in parts and "_preview" in parts:
        return "incoming preview"
    if "incoming" in parts and "_review2" in parts:
        return "incoming review"
    if "new_dataset" in parts and "_done" in parts:
        return "new_dataset raw/done"
    if "new_dataset" in parts and "_preview" in parts:
        return "new_dataset preview"
    if "composites" in parts and "images" in parts:
        return "composite image"
    if "composites" in parts and "masks" in parts:
        return "composite mask"
    if "assets" in parts:
        return "sdk asset"
    if "images" in parts:
        return "image"
    return path.parent.name.lower()


def source_name(path: Path) -> str:
    text = str(path)
    if "/Smart-Scan-SDK/" in text:
        return "Smart-Scan-SDK"
    if "/Smart-Scan/" in text:
        return "Smart-Scan"
    return path.parent.name


def load_annotation(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
