"""Run Tesseract once through the C API and keep every intermediate result for the dashboard.

Threshold values are not returned by Tesseract, so they are recomputed with Tesseract's own
formulas (src/ccstruct/otsuthr.cpp, src/ccmain/thresholder.cpp in 5.5) and the recomputed
binary is compared pixel by pixel with the binary Tesseract actually used.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image

from nid_ocr_lab.engines import tesseract_capi as capi
from nid_ocr_lab.engines.tessdata import get_variant
from nid_ocr_lab.engines.tesseract import resolve_dpi
from nid_ocr_lab.engines.tesseract_fields import refine_nid_fields
from nid_ocr_lab.models import OCRResult
from nid_ocr_lab.parsers.nid_parser import NIDParser

MIN_CREDIBLE_DPI = 70  # kMinCredibleResolution
MAX_CREDIBLE_DPI = 2400  # kMaxCredibleResolution
THRESHOLD_METHODS = {"otsu": 0, "adaptive_otsu": 1, "sauvola": 2, "manual": None}
THRESHOLD_DEFAULTS = {
    "method": "otsu",
    "tile_size": 0.33,
    "smooth_kernel_size": 0.0,
    "score_fraction": 0.1,
    "window_size": 0.33,
    "kfactor": 0.34,
    "value": 128,
}
THRESHOLD_LIMITS = {
    "tile_size": (0.01, 10.0),
    "smooth_kernel_size": (0.0, 10.0),
    "score_fraction": (0.0, 1.0),
    "window_size": (0.01, 10.0),
    "kfactor": (0.0, 2.0),
    "value": (0, 255),
}
CHANNEL_NAMES = ("red", "green", "blue", "alpha")
# PSMs that run automatic page layout (column finding), the only ones that dump page segmentation images.
AUTO_LAYOUT_PSMS = {1, 2, 3, 4, 11, 12}
NID_FIELD_NAMES = ("name_bangla", "name_english", "father_name_bangla", "mother_name_bangla", "date_of_birth", "nid_number")
LOG_PATTERNS = re.compile(r"Estimating resolution|diacritic|Warning|tile size|window size|image width|Detected|Invalid|Too few")


def normalize_threshold(raw: dict[str, Any] | None) -> dict[str, Any]:
    settings = {**THRESHOLD_DEFAULTS, **{key: value for key, value in (raw or {}).items() if value not in (None, "")}}
    if settings["method"] not in THRESHOLD_METHODS:
        raise ValueError(f"Threshold method must be one of {', '.join(THRESHOLD_METHODS)}")
    for key, (low, high) in THRESHOLD_LIMITS.items():
        try:
            value = float(settings[key])
        except (TypeError, ValueError):
            raise ValueError(f"Threshold {key} must be a number") from None
        if not low <= value <= high:
            raise ValueError(f"Threshold {key} must be between {low} and {high}")
        settings[key] = int(round(value)) if key == "value" else value
    return settings


def threshold_resolution(image_dpi: int, user_dpi: int | None) -> int:
    """The resolution Tesseract's thresholder sizes windows with (TessBaseAPI::Threshold)."""
    if user_dpi:
        return int(user_dpi)
    return image_dpi if MIN_CREDIBLE_DPI <= image_dpi <= MAX_CREDIBLE_DPI else MIN_CREDIBLE_DPI


def adaptive_otsu_sizes(resolution: int, tile_size: float, smooth_kernel_size: float) -> dict[str, int]:
    tile = max(16, int(tile_size * resolution))
    smooth = int(max(0.0, smooth_kernel_size) * resolution)
    return {"tile": tile, "smooth": smooth, "half_smooth": smooth // 2}


def sauvola_sizes(resolution: int, window_size: float, width: int, height: int) -> dict[str, int]:
    window = max(7, int(window_size * resolution))
    window = min(width - 3 if width < height else height - 3, window)
    half = window // 2
    nx = max(1, (width + 125) // 250)
    ny = max(1, (height + 125) // 250)
    if width // nx < half + 2:
        nx = width // (half + 2)
    if height // ny < half + 2:
        ny = height // (half + 2)
    return {"window": window, "half_window": half, "nx": max(1, nx), "ny": max(1, ny)}


def otsu_stats(histogram: np.ndarray) -> tuple[int, int, int]:
    """(best threshold, pixel count, pixels at or below threshold), as OtsuStats computes them."""
    total = int(histogram.sum())
    mu_total = float((np.arange(256) * histogram).sum())
    best_t, best_omega0, best_sigma = -1, 0, 0.0
    omega0, mu_t = 0, 0.0
    for t in range(255):
        omega0 += int(histogram[t])
        mu_t += t * float(histogram[t])
        if omega0 == 0:
            continue
        omega1 = total - omega0
        if omega1 == 0:
            break
        sigma = ((mu_total - mu_t) / omega1 - mu_t / omega0) ** 2 * omega0 * omega1
        if best_t < 0 or sigma > best_sigma:
            best_sigma, best_t, best_omega0 = sigma, t, omega0
    return best_t, total, best_omega0


def tesseract_otsu(channels: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Tesseract's global Otsu: one threshold per channel and which side is ink. Returns (channels, ink mask)."""
    count = channels.shape[2]
    thresholds, hi_values = [-1] * count, [-1] * count
    any_good, best_hi_dist, best_hi_value, best_hi_index = False, 0.0, 1, 0
    for channel in range(count):
        histogram = np.bincount(channels[..., channel].ravel(), minlength=256)
        best_t, total, omega0 = otsu_stats(histogram)
        if omega0 == 0 or omega0 == total:
            continue
        hi_value = int(omega0 < total * 0.5)
        thresholds[channel] = best_t
        if omega0 > total * 0.75:
            any_good, hi_values[channel] = True, 0
        elif omega0 < total * 0.25:
            any_good, hi_values[channel] = True, 1
        else:
            distance = (total - omega0) if hi_value else omega0
            if distance > best_hi_dist:
                best_hi_dist, best_hi_value, best_hi_index = distance, hi_value, channel
    if not any_good:
        hi_values[best_hi_index] = best_hi_value
    ink = np.zeros(channels.shape[:2], dtype=bool)
    summary = []
    for channel in range(count):
        if hi_values[channel] >= 0:
            ink |= (channels[..., channel] > thresholds[channel]) == (hi_values[channel] == 0)
        summary.append(
            {
                "channel": CHANNEL_NAMES[channel] if count > 1 else "grey",
                "threshold": thresholds[channel],
                "used": hi_values[channel] >= 0,
                "ink": {1: f"≤ {thresholds[channel]} is ink", 0: f"> {thresholds[channel]} is ink"}.get(hi_values[channel], "ignored"),
            }
        )
    return summary, ink


def leptonica_channels(image: Image.Image, depth: int) -> np.ndarray:
    """Pixel bytes in the order Tesseract reads them from a Leptonica Pix (RGBA for 32 bpp, alpha 0 when absent)."""
    if depth == 8:
        return np.array(image.convert("L"))[..., None]
    rgb = np.array(image.convert("RGB"))
    alpha = np.array(image.getchannel("A")) if "A" in image.getbands() else np.zeros(rgb.shape[:2], np.uint8)
    return np.dstack([rgb, alpha])


def ink_mask(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L")) < 128


def mask_image(ink: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(ink, 0, 255).astype(np.uint8), "L")


def heatmap(values: Image.Image) -> Image.Image:
    array = np.array(values.convert("L")).astype(np.float32)
    low, high = float(array.min()), float(array.max())
    scaled = (array - low) / max(high - low, 1.0)
    red = (255 * scaled).astype(np.uint8)
    blue = (255 * (1 - scaled)).astype(np.uint8)
    green = (255 * (1 - np.abs(scaled - 0.5) * 2)).astype(np.uint8)
    return Image.fromarray(np.dstack([red, green, blue]), "RGB")


def difference_image(tesseract_ink: np.ndarray, replica_ink: np.ndarray) -> Image.Image:
    output = np.full(tesseract_ink.shape + (3,), 255, np.uint8)
    output[tesseract_ink & replica_ink] = (180, 180, 180)
    output[tesseract_ink & ~replica_ink] = (220, 40, 40)
    output[~tesseract_ink & replica_ink] = (40, 90, 220)
    return Image.fromarray(output, "RGB")


def box_stats(boxes: list[dict[str, int]]) -> dict[str, Any]:
    if not boxes:
        return {"count": 0}
    heights = [box["height"] for box in boxes]
    widths = [box["width"] for box in boxes]
    return {"count": len(boxes), "median_height": median(heights), "median_width": median(widths), "max_height": max(heights)}


def save(image: Image.Image, workdir: Path, name: str) -> str:
    image.save(workdir / name)
    return name


def inspect_steps(
    image_path: Path,
    languages: list[str],
    variant: str | None,
    psm: int,
    workdir: Path,
    *,
    dpi: int | None = None,
    threshold: dict[str, Any] | None = None,
    recognize: bool = True,
    debug_images: bool = True,
    fields: bool = True,
    orientation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    settings = normalize_threshold(threshold)
    if dpi is not None and not MIN_CREDIBLE_DPI <= int(dpi) <= MAX_CREDIBLE_DPI:
        raise ValueError(f"DPI must be between {MIN_CREDIBLE_DPI} and {MAX_CREDIBLE_DPI}")
    model = get_variant(variant)
    language = "+".join(languages)
    missing = [item for item in languages if not model.model_path(item).is_file()]
    if missing:
        raise RuntimeError(f"Tesseract variant '{model.id}' has no model for {missing} in {model.directory}.")

    steps: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    with Image.open(image_path) as opened:
        source = opened.copy()
    metadata_dpi, dpi_source = resolve_dpi(image_path, dpi)

    steps.append(
        {
            "id": "input",
            "title": "Input image",
            "explain": "The exact image handed to Tesseract, after the dashboard's crop, filter and rotation.",
            "values": {"width": source.width, "height": source.height, "mode": source.mode, "dpi": metadata_dpi, "dpi_source": dpi_source},
            "images": {"input": save(source, workdir, "00-input.png")},
        }
    )
    if orientation:
        steps.append(
            {
                "id": "orientation",
                "title": "Orientation (OSD)",
                "explain": "Tesseract's orientation and script detection (--psm 0) decided how far to rotate the image before the steps below.",
                "values": orientation,
                "images": {},
            }
        )

    tesseract_input = image_path
    with capi.Pix.read(image_path) as pix:
        width, height, depth = pix.size
        pix_dpi = pix.y_resolution
        with pix.grey() as grey_pix:
            grey = grey_pix.to_pil().convert("L")
            resolution = threshold_resolution(pix_dpi, dpi)
            method = settings["method"]
            replica_ink: np.ndarray | None = None
            threshold_values: dict[str, Any] = {"method": method, "resolution_used": resolution}
            threshold_images: dict[str, str] = {}
            if depth == 1:
                # Already binary: Tesseract copies it unchanged whatever the method.
                replica_ink = ink_mask(source)
                threshold_values["note"] = "Input is already 1-bit; Tesseract skips thresholding."
            elif method == "otsu":
                channels, replica_ink = tesseract_otsu(leptonica_channels(source, depth))
                threshold_values["channels"] = channels
                used = [item["threshold"] for item in channels if item["used"]]
                threshold_values["threshold"] = used[0] if len(used) == 1 else used
            elif method == "adaptive_otsu":
                sizes = adaptive_otsu_sizes(resolution, settings["tile_size"], settings["smooth_kernel_size"])
                threshold_map, replica = capi.leptonica_threshold(
                    grey_pix, method, tile=sizes["tile"], half_smooth=sizes["half_smooth"], score_fraction=settings["score_fraction"]
                )
                replica_ink = ink_mask(replica)
                values = np.array(threshold_map.convert("L"))
                threshold_values.update(
                    {**sizes, "score_fraction": settings["score_fraction"], "map_size": list(threshold_map.size),
                     "threshold_min": int(values.min()), "threshold_median": int(np.median(values)), "threshold_max": int(values.max()),
                     "tile_thresholds": values.tolist() if values.size <= 4096 else None}
                )
                threshold_images["threshold_map"] = save(heatmap(threshold_map), workdir, "03-threshold-map.png")
            elif method == "sauvola":
                sizes = sauvola_sizes(resolution, settings["window_size"], width, height)
                threshold_map, replica = capi.leptonica_threshold(
                    grey_pix, method, half_window=sizes["half_window"], kfactor=settings["kfactor"], nx=sizes["nx"], ny=sizes["ny"]
                )
                replica_ink = ink_mask(replica)
                values = np.array(threshold_map.convert("L"))
                threshold_values.update(
                    {**sizes, "kfactor": settings["kfactor"], "threshold_min": int(values.min()),
                     "threshold_median": int(np.median(values)), "threshold_max": int(values.max())}
                )
                threshold_images["threshold_map"] = save(heatmap(threshold_map), workdir, "03-threshold-map.png")
            else:
                replica_ink = np.array(grey) <= settings["value"]
                threshold_values["threshold"] = settings["value"]
                tesseract_input = workdir / "03-manual-binary-input.png"
                binary_input = mask_image(replica_ink).convert("1")
                binary_input.save(tesseract_input, **({"dpi": (pix_dpi, pix_dpi)} if pix_dpi else {}))

    histogram = np.bincount(np.array(grey).ravel(), minlength=256).tolist()
    steps.append(
        {
            "id": "grey",
            "title": "Greyscale",
            "explain": "Colour is reduced to one brightness value per pixel (Leptonica pixConvertTo8). Adaptive methods threshold this image; global Otsu thresholds each colour channel.",
            "values": {"depth_bits": depth, "histogram": histogram},
            "images": {"grey": save(grey, workdir, "02-grey.png")},
        }
    )

    with capi.TessAPI(model.directory, language) as api:
        api.set_psm(psm)
        if dpi:
            api.set_variable("user_defined_dpi", int(dpi))
        if THRESHOLD_METHODS[settings["method"]] is not None:
            api.set_variable("thresholding_method", THRESHOLD_METHODS[settings["method"]])
            for key in ("tile_size", "smooth_kernel_size", "score_fraction", "window_size", "kfactor"):
                api.set_variable(f"thresholding_{key}", settings[key])
        if recognize:
            api.set_variable("lstm_choice_mode", 2)
        api.set_image(tesseract_input)

        started = time.perf_counter()
        binary, scale = api.thresholded_image()
        timings["threshold_ms"] = round((time.perf_counter() - started) * 1000, 1)
        tesseract_ink = ink_mask(binary)
        match = None
        if replica_ink is not None and replica_ink.shape == tesseract_ink.shape:
            match = round(float((replica_ink == tesseract_ink).mean()) * 100, 4)
            if match < 100:
                threshold_images["difference"] = save(difference_image(tesseract_ink, replica_ink), workdir, "03-difference.png")
        threshold_images["binary"] = save(mask_image(tesseract_ink), workdir, "03-binary.png")
        threshold_values.update({"ink_share": round(float(tesseract_ink.mean()), 4), "scale_factor": scale, "replica_match_percent": match})
        manual_note = (
            " Manual: the image is binarized here and handed to Tesseract, so the line recognizer also reads this binary image."
            if settings["method"] == "manual"
            else " This binary drives layout only; the line recognizer later reads the original image."
        )
        steps.append(
            {
                "id": "threshold",
                "title": "Threshold (binarize)",
                "explain": "Every pixel becomes ink or background." + manual_note,
                "values": threshold_values,
                "images": threshold_images,
            }
        )

        started = time.perf_counter()
        blobs = api.connected_components()
        elements = api.layout(recognize=recognize)
        timings["layout_ms" if not recognize else "layout_and_recognition_ms"] = round((time.perf_counter() - started) * 1000, 1)
        full_text = api.text() if recognize else ""
        mean_confidence = api.mean_confidence() if recognize else None

    by_level: dict[str, list[dict[str, Any]]] = {level: [] for level in capi.LEVEL_NAMES.values()}
    for element in elements:
        by_level[element["level"]].append(element)

    steps.append(
        {
            "id": "blobs",
            "title": "Connected components (blobs)",
            "explain": "Each separate patch of touching ink pixels is a blob — usually a character or part of one. Tesseract groups these into lines.",
            "values": box_stats(blobs),
            "images": {},
            "overlay": {"source": "binary", "boxes": {"blobs": blobs}},
        }
    )
    block_types: dict[str, int] = {}
    for block in by_level["block"]:
        block_types[block["block_type"]] = block_types.get(block["block_type"], 0) + 1
    steps.append(
        {
            "id": "blocks",
            "title": "Blocks",
            "explain": "Regions found by page layout analysis. Only text blocks are read; image, line and noise blocks are dropped here.",
            "values": {"count": len(by_level["block"]), "types": block_types,
                       "non_text": sum(count for kind, count in block_types.items() if kind not in capi.TEXT_BLOCK_TYPES)},
            "images": {},
            "overlay": {"source": "binary", "boxes": {"blocks": [strip(item) for item in by_level["block"]]}},
        }
    )
    steps.append(
        {
            "id": "paragraphs",
            "title": "Paragraphs",
            "explain": "Lines grouped by indentation and spacing inside each block.",
            "values": {"count": len(by_level["paragraph"])},
            "images": {},
            "overlay": {"source": "binary", "boxes": {"paragraphs": [strip(item) for item in by_level["paragraph"]]}},
        }
    )
    lines_json = []
    for line in by_level["line"]:
        images = line.pop("_line_images", {})
        index = line["id"]
        saved = {}
        if "original" in images:
            saved["original"] = save(images["original"], workdir, f"line-{index:03d}-original.png")
        if "binary" in images:
            saved["binary"] = save(images["binary"], workdir, f"line-{index:03d}-binary.png")
        lines_json.append({**line, "images": saved})
    line_boxes = [line["box"] for line in lines_json]
    slopes = [
        round(float(np.degrees(np.arctan2(line["baseline"][3] - line["baseline"][1], max(1, line["baseline"][2] - line["baseline"][0])))), 2)
        for line in lines_json
        if line.get("baseline")
    ]
    steps.append(
        {
            "id": "lines",
            "title": "Text lines",
            "explain": "Blobs on a shared baseline are chained into lines. A line box much taller than the others usually means two rows were merged.",
            "values": {**box_stats(line_boxes), "baseline_slope_degrees": slopes},
            "images": {},
            "overlay": {"source": "binary", "boxes": {"lines": [strip(line) for line in lines_json]}},
        }
    )
    steps.append(
        {
            "id": "words",
            "title": "Words and symbols",
            "explain": "Lines split into words by gap size, and words into symbols (characters).",
            "values": {"words": len(by_level["word"]), "symbols": len(by_level["symbol"])},
            "images": {},
            "overlay": {"source": "binary", "boxes": {"words": [strip(item) for item in by_level["word"]], "symbols": [strip(item) for item in by_level["symbol"]]}},
        }
    )

    if debug_images and psm in AUTO_LAYOUT_PSMS:
        debug = run_debug_dump(tesseract_input, model.directory, language, psm, dpi, settings, workdir)
        if debug:
            steps.append(
                {
                    "id": "debug",
                    "title": "Tesseract's own page segmentation images",
                    "explain": "Images Tesseract writes with tessedit_dump_pageseg_images=1 during page layout, captioned inside each image: PageSegInput = binary entering layout, NoLines = after removing ruled lines, NoImages = after removing picture regions (photo, emblem). Plus its log. Separate CLI run with the same settings.",
                    "values": {"log": debug["log"], "estimated_resolution": debug["estimated_resolution"]},
                    "images": {f"page_{index + 1}": name for index, name in enumerate(debug["images"])},
                }
            )
    elif debug_images:
        steps.append(
            {
                "id": "debug",
                "title": "Tesseract's own page segmentation images",
                "explain": f"None for PSM {psm}: Tesseract writes these images (and its resolution estimate) only during automatic layout analysis, PSM {', '.join(map(str, sorted(AUTO_LAYOUT_PSMS)))}. PSM {psm} skips column finding.",
                "values": {},
                "images": {},
            }
        )

    if recognize:
        words_by_line: dict[int, list[dict[str, Any]]] = {}
        symbols_by_word: dict[int, list[dict[str, Any]]] = {}
        for symbol in by_level["symbol"]:
            symbols_by_word.setdefault(symbol["parent"], []).append({key: symbol[key] for key in ("text", "confidence", "choices", "box")})
        for word in by_level["word"]:
            words_by_line.setdefault(word["parent"], []).append(
                {**{key: word.get(key) for key in ("text", "confidence", "language", "from_dictionary", "box")}, "symbols": symbols_by_word.get(word["id"], [])}
            )
        for line in lines_json:
            line["words"] = words_by_line.get(line["id"], [])
        steps.append(
            {
                "id": "recognition",
                "title": "LSTM line recognition",
                "explain": "Each line is cut from the original image (not the binary), scaled to the network's input height, and read left to right. Words show which language model won and alternative characters per symbol.",
                "values": {"lines": len(lines_json), "mean_confidence": mean_confidence},
                "images": {},
                "lines": lines_json,
            }
        )
        parsed = NIDParser().parse(OCRResult(blocks=[], full_text=full_text, engine="tesseract"))
        page_fields = {name: getattr(parsed, name).raw_value for name in NID_FIELD_NAMES}
        steps.append(
            {
                "id": "output",
                "title": "Page output",
                "explain": "What one whole-page Tesseract run produces: text in reading order, and the NID fields the parser finds in it. A row the layout step dropped is missing here.",
                "values": {"text": full_text, "mean_confidence": mean_confidence, "fields": page_fields},
                "images": {},
            }
        )
        if fields:
            steps.append(field_step(tesseract_input, by_level, language, variant, workdir, page_fields))
    else:
        steps.append(
            {
                "id": "recognition",
                "title": "LSTM line recognition",
                "explain": "Skipped: layout only. Untick 'Layout only' to read the lines.",
                "values": {"lines": len(lines_json)},
                "images": {},
                "lines": lines_json,
            }
        )

    return {
        "tesseract_version": capi.version(),
        "settings": {"language": language, "variant": model.id, "psm": psm, "dpi": dpi, "threshold": settings, "recognize": recognize, "fields": fields},
        "image": {"width": width, "height": height},
        "timings": timings,
        "steps": steps,
    }


def field_step(
    image_path: Path,
    by_level: dict[str, list[dict[str, Any]]],
    language: str,
    variant: str | None,
    workdir: Path,
    page_fields: dict[str, str | None],
) -> dict[str, Any]:
    """Run the NID fields strategy on this page result and show what each row gained."""
    words = [
        {
            "text": word["text"],
            "confidence": (word["confidence"] or 0) / 100.0,
            "bounding_box": word["box"],
            "line_key": [1, 1, 1, word["parent"]],
        }
        for word in by_level["word"]
        if word.get("text")
    ]
    page_result = OCRResult(blocks=[], full_text="", engine="tesseract", language=language, metadata={"words": words})
    started = time.perf_counter()
    refined = refine_nid_fields(None, image_path, page_result, variant=variant, workdir=workdir)
    elapsed = (time.perf_counter() - started) * 1000
    parsed = NIDParser().parse(refined)
    rows = []
    for field in refined.metadata["fields"]:
        crop = next((name.name for name in sorted(workdir.glob(f"field-*-{field['field']}-psm*.png"))), None)
        rows.append({**{key: field[key] for key in ("field", "label", "source", "text", "confidence", "languages")}, "crop": crop})
    return {
        "id": "fields",
        "title": "NID fields re-read",
        "explain": "Each NID row is located by its label, rows missing between two found labels are placed by position, and every value is re-read on its own crop with one language (PSM 7). The best of page value and re-reads wins.",
        "values": {
            "calls": refined.metadata["field_calls"],
            "reader": refined.metadata["field_reader"],
            "elapsed_ms": round(elapsed),
            "text": refined.full_text,
            "fields": {name: getattr(parsed, name).raw_value for name in NID_FIELD_NAMES},
            "page_fields": page_fields,
        },
        "images": {},
        "rows": rows,
    }


def strip(element: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in element.items() if key in ("id", "parent", "box", "block_type", "baseline", "text", "confidence")}


def run_debug_dump(
    image_path: Path, datapath: Path, language: str, psm: int, dpi: int | None, settings: dict[str, Any], workdir: Path
) -> dict[str, Any] | None:
    """Tesseract only writes its page segmentation debug PDF from a normal run, so use the CLI once."""
    binary = shutil.which("tesseract")
    if not binary:
        return None
    base = workdir / "debug"
    command = [binary, str(image_path), str(base), "--tessdata-dir", str(datapath), "-l", language, "--oem", "1", "--psm", str(psm)]
    if dpi:
        command += ["--dpi", str(dpi)]
    variables = {"tessedit_dump_pageseg_images": 1, "thresholding_debug": 1}
    if THRESHOLD_METHODS[settings["method"]] is not None:
        variables["thresholding_method"] = THRESHOLD_METHODS[settings["method"]]
        for key in ("tile_size", "smooth_kernel_size", "score_fraction", "window_size", "kfactor"):
            variables[f"thresholding_{key}"] = settings[key]
    for key, value in variables.items():
        command += ["-c", f"{key}={value}"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    log = [line.strip() for line in (completed.stderr or "").splitlines() if LOG_PATTERNS.search(line)]
    estimated = re.search(r"Estimating resolution as (\d+)", completed.stderr or "")
    images = []
    pdf = workdir / "debug_debug.pdf"
    if pdf.is_file():
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf))
        try:
            for index in range(len(document)):
                page = document[index]
                name = f"04-pageseg-{index + 1}.png"
                page.render(scale=2).to_pil().save(workdir / name)
                images.append(name)
                page.close()
        finally:
            document.close()
    return {"log": log, "estimated_resolution": int(estimated.group(1)) if estimated else None, "images": images}
