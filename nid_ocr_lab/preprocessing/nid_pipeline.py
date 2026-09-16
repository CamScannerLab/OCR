from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dashboard dependency fallback
    cv2 = None

RECIPES = {
    "off": "Off",
    "nid_ink_v1": "NID ink v1",
    "nid_ink_v2_contrast": "NID ink v2 controlled contrast",
    "nid_ink_v2": "NID ink v2 masked ink",
}


def inspect_preprocessing(image_path: Path, workdir: Path, recipe: str) -> dict[str, Any]:
    if recipe not in RECIPES:
        raise ValueError(f"Unknown preprocessing recipe: {recipe}")
    started = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        dpi = opened.info.get("dpi")
        image = ImageOps.exif_transpose(opened).convert("RGB")

    steps: list[dict[str, Any]] = []
    input_name = save(image, workdir, "00-input.png", dpi=dpi)
    steps.append(
        step(
            "input",
            "Input image",
            "The image after EXIF orientation normalization, before NID-specific preprocessing.",
            {"width": image.width, "height": image.height, "mode": "RGB", "dpi": dpi_value(dpi)},
            {"output": input_name},
        )
    )

    if recipe == "off":
        final = image
        final_name = save(final, workdir, "99-final-ocr.png", dpi=dpi)
        steps.append(
            step(
                "final",
                "Final OCR image",
                "No preprocessing was applied. This preserves the current baseline.",
                image_metrics(final),
                {"output": final_name},
            )
        )
    elif recipe == "nid_ink_v1":
        final, recipe_steps = nid_ink_v1(image, workdir, dpi)
        steps.extend(recipe_steps)
        final_name = save(final, workdir, "99-final-ocr.png", dpi=dpi)
        steps.append(
            step(
                "final",
                "Final OCR image",
                "The current dashboard NID ink filter: green channel plus autocontrast.",
                image_metrics(final),
                {"output": final_name},
            )
        )
    elif recipe in {"nid_ink_v2", "nid_ink_v2_contrast"}:
        final_mode = "controlled_contrast" if recipe == "nid_ink_v2_contrast" else "masked_ink"
        final, recipe_steps = nid_ink_v2(image, workdir, dpi, final_mode=final_mode)
        steps.extend(recipe_steps)
        final_name = save(final, workdir, "99-final-ocr.png", dpi=dpi)
        final_explain = (
            "The controlled-contrast image handed to local Tesseract. This matches the best manual flow: NID ink, v2 preprocessing, then OCR on 04-controlled-contrast.png."
            if final_mode == "controlled_contrast"
            else "The masked text-ink image handed to local Tesseract after measured green-channel cleanup."
        )
        steps.append(
            step(
                "final",
                "Final OCR image",
                final_explain,
                image_metrics(final),
                {"output": final_name},
            )
        )

    final_path = workdir / "99-final-ocr.png"
    return {
        "recipe": {"id": recipe, "label": RECIPES[recipe]},
        "image": {"width": final.width, "height": final.height, "source_name": image_path.name},
        "timings": {"preprocess_ms": round((time.perf_counter() - started) * 1000, 1)},
        "steps": steps,
        "final_image": {"name": "99-final-ocr.png", "path": str(final_path), "width": final.width, "height": final.height},
    }


def nid_ink_v1(image: Image.Image, workdir: Path, dpi: tuple[int, int] | None) -> tuple[Image.Image, list[dict[str, Any]]]:
    green = image.getchannel("G")
    green_name = save(green, workdir, "01-green-channel.png", dpi=dpi)
    final = ImageOps.autocontrast(green, cutoff=1).convert("RGB")
    output_name = save(final, workdir, "02-nid-ink-v1.png", dpi=dpi)
    return final, [
        step(
            "green_channel",
            "Green channel",
            "Keeps the green channel. The yellow/orange emblem moves closer to the card background while dark text remains dark.",
            channel_metrics(green),
            {"output": green_name},
        ),
        step(
            "autocontrast",
            "Autocontrast",
            "Stretches the green-channel tones with 1% clipping, matching the existing NID ink filter.",
            image_metrics(final),
            {"output": output_name},
        ),
    ]


def nid_ink_v2(
    image: Image.Image,
    workdir: Path,
    dpi: tuple[int, int] | None,
    *,
    final_mode: str = "masked_ink",
) -> tuple[Image.Image, list[dict[str, Any]]]:
    green = image.getchannel("G")
    green_name = save(green, workdir, "01-green-channel.png", dpi=dpi)

    background = estimate_background(green)
    background_name = save(background, workdir, "02-background-estimate.png", dpi=dpi)

    normalized = normalize_against_background(green, background)
    normalized_name = save(normalized, workdir, "03-background-normalized.png", dpi=dpi)

    stretched = ImageOps.autocontrast(normalized, cutoff=0.5)
    stretched_name = save(stretched, workdir, "04-controlled-contrast.png", dpi=dpi)

    mask = foreground_mask(stretched)
    mask_name = save(mask, workdir, "05-ink-mask.png", dpi=dpi)

    masked_final = blend_preserving_dark_ink(stretched, mask).convert("RGB")
    final_name = save(masked_final, workdir, "06-nid-ink-v2-masked.png", dpi=dpi)
    if final_mode == "controlled_contrast":
        final = stretched.convert("RGB")
    elif final_mode == "masked_ink":
        final = masked_final
    else:
        raise ValueError(f"Unknown NID ink v2 final mode: {final_mode}")

    return final, [
        step(
            "green_channel",
            "Green channel",
            "Starts with the same colour separation as NID ink v1 because it suppresses the yellow/orange emblem well.",
            channel_metrics(green),
            {"output": green_name},
        ),
        step(
            "background_estimate",
            "Background estimate",
            "A low-frequency estimate of paper, emblem and lighting tone. Text strokes are intentionally blurred away here.",
            channel_metrics(background),
            {"output": background_name},
        ),
        step(
            "background_normalized",
            "Background normalized",
            "Divides the green channel by the background estimate so watermark texture becomes flatter.",
            channel_metrics(normalized),
            {"output": normalized_name},
        ),
        step(
            "controlled_contrast",
            "Controlled contrast",
            "Applies gentler clipping than v1 after background normalization, so thin Bengali marks are less likely to disappear.",
            channel_metrics(stretched),
            {"output": stretched_name},
        ),
        step(
            "ink_mask",
            "Ink mask",
            "A diagnostic mask for likely printed text strokes. It is shown for inspection; it is not the final OCR image.",
            {"ink_share": ink_share(mask), **channel_metrics(mask)},
            {"output": mask_name},
        ),
        step(
            "text_ink_image",
            "Masked text ink image",
            "Uses the mask to keep strong dark strokes while leaving the recognizer a grayscale image rather than a hard binary only. This is visible for comparison, but controlled contrast can now be the OCR final image.",
            image_metrics(masked_final),
            {"output": final_name},
        ),
    ]


def estimate_background(grey: Image.Image) -> Image.Image:
    radius = max(9, min(grey.size) // 28)
    if cv2 is not None:
        array = np.asarray(grey)
        kernel = max(17, radius | 1)
        return Image.fromarray(cv2.GaussianBlur(array, (kernel, kernel), 0), "L")
    return grey.filter(ImageFilter.GaussianBlur(radius=radius))


def normalize_against_background(grey: Image.Image, background: Image.Image) -> Image.Image:
    source = np.asarray(grey).astype(np.float32)
    bg = np.asarray(background).astype(np.float32)
    reference = max(1.0, float(np.percentile(bg, 88)))
    normalized = np.clip((source / np.maximum(bg, 1.0)) * reference, 0, 255)
    return Image.fromarray(normalized.astype(np.uint8), "L")


def foreground_mask(grey: Image.Image) -> Image.Image:
    array = np.asarray(grey).astype(np.uint8)
    threshold = max(20, min(245, int(np.percentile(array, 38))))
    mask = array <= threshold
    if cv2 is not None:
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        mask = cleaned > 0
    return Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), "L")


def blend_preserving_dark_ink(grey: Image.Image, mask: Image.Image) -> Image.Image:
    source = np.asarray(grey).astype(np.float32)
    ink = np.asarray(mask) < 128
    output = np.clip(source * 1.06 + 6, 0, 255)
    output[ink] = np.clip(source[ink] * 0.78, 0, 255)
    return Image.fromarray(output.astype(np.uint8), "L")


def step(step_id: str, title: str, explain: str, values: dict[str, Any], images: dict[str, str]) -> dict[str, Any]:
    return {"id": step_id, "title": title, "explain": explain, "values": values, "images": images}


def save(image: Image.Image, workdir: Path, name: str, dpi: tuple[int, int] | None = None) -> str:
    kwargs = {"dpi": dpi} if dpi else {}
    image.save(workdir / name, **kwargs)
    return name


def dpi_value(dpi: object) -> str | None:
    if isinstance(dpi, tuple) and dpi:
        return "x".join(str(round(float(item))) for item in dpi[:2])
    return None


def image_metrics(image: Image.Image) -> dict[str, Any]:
    grey = image.convert("L")
    return {"width": image.width, "height": image.height, **channel_metrics(grey), "ink_share": ink_share(grey)}


def channel_metrics(image: Image.Image) -> dict[str, Any]:
    array = np.asarray(image.convert("L"))
    return {
        "min": int(array.min()),
        "p05": int(np.percentile(array, 5)),
        "median": int(np.percentile(array, 50)),
        "p95": int(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def ink_share(image: Image.Image) -> float:
    return round(float((np.asarray(image.convert("L")) < 128).mean()), 4)

