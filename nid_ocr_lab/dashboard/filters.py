from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import hypot, sqrt
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


FILTER_MODES = ["original", "enhance", "deglare", "matte", "super", "edge", "gray", "threshold", "nid_ink"]


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str


def render_image(
    path: str, mode: str = "original", max_width: int = 1800, rotation: int = 0
) -> tuple[bytes, ImageInfo]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((max_width, max_width * 2), Image.Resampling.LANCZOS)
    output = rotate_clockwise(apply_filter(image, mode), rotation)
    payload = BytesIO()
    output.save(payload, format="JPEG", quality=92, optimize=True)
    return payload.getvalue(), ImageInfo(width=output.width, height=output.height, mode=mode)


def save_filtered_image(
    image_path: str,
    output_path: str | Path,
    mode: str = "original",
    max_width: int | None = 1800,
) -> ImageInfo:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if max_width:
        image.thumbnail((max_width, max_width * 2), Image.Resampling.LANCZOS)
    output = apply_filter(image, mode)
    save_ocr_input(output, output_path)
    return ImageInfo(width=output.width, height=output.height, mode=mode)


def save_ocr_input(image: Image.Image, output_path: str | Path) -> None:
    """OCR inputs are saved losslessly (format follows the suffix, PNG expected)."""
    if Path(output_path).suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, format="JPEG", quality=92, optimize=True)
    else:
        image.save(output_path)


def render_mask_overlay(
    image_path: str,
    mask_path: str,
    max_width: int = 1800,
    alpha: float = 0.35,
) -> tuple[bytes, ImageInfo]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    mask = Image.open(mask_path).convert("L")
    if mask.size != original_size:
        mask = mask.resize(original_size, Image.Resampling.NEAREST)

    image.thumbnail((max_width, max_width * 2), Image.Resampling.LANCZOS)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    base = np.asarray(image).astype(np.float32)
    mask_array = (np.asarray(mask).astype(np.float32) / 255.0)[:, :, None]
    green = np.zeros_like(base)
    green[:, :, 1] = 255
    blended = base * (1 - mask_array * alpha) + green * (mask_array * alpha)

    output = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
    payload = BytesIO()
    output.save(payload, format="JPEG", quality=92, optimize=True)
    return payload.getvalue(), ImageInfo(width=output.width, height=output.height, mode="mask_overlay")


def render_sdk_crop(
    image_path: str,
    points: list[list[float]],
    mode: str = "original",
    target_aspect_ratio: float = 85.6 / 54.0,
    output_max_pixels: int = 2_000_000,
    rotation: int = 0,
) -> tuple[bytes, ImageInfo]:
    image = Image.open(image_path).convert("RGB")
    quad = order_quad(points)
    cropped = perspective_correct(image, quad, target_aspect_ratio=target_aspect_ratio)
    cropped = downscale_max_pixels(cropped, output_max_pixels)
    output = rotate_clockwise(apply_filter(cropped, mode), rotation)
    payload = BytesIO()
    output.save(payload, format="JPEG", quality=92, optimize=True)
    return payload.getvalue(), ImageInfo(width=output.width, height=output.height, mode=f"sdk_crop_{mode}")


def save_sdk_crop(
    image_path: str,
    points: list[list[float]],
    output_path: str | Path,
    mode: str = "original",
    target_aspect_ratio: float = 85.6 / 54.0,
    output_max_pixels: int = 2_000_000,
) -> ImageInfo:
    image = Image.open(image_path).convert("RGB")
    quad = order_quad(points)
    cropped = perspective_correct(image, quad, target_aspect_ratio=target_aspect_ratio)
    cropped = downscale_max_pixels(cropped, output_max_pixels)
    output = apply_filter(cropped, mode)
    save_ocr_input(output, output_path)
    return ImageInfo(width=output.width, height=output.height, mode=f"sdk_crop_{mode}")


def rotate_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    """Same clockwise convention as the OCR request rotation."""
    degrees %= 360
    return image.rotate(-degrees, expand=True) if degrees else image


def apply_filter(image: Image.Image, mode: str) -> Image.Image:
    normalized = mode.lower()
    if normalized == "original":
        return image
    if normalized == "enhance":
        return enhance(image, sharpen=1.20, saturation=1.0875)
    if normalized == "deglare":
        return enhance(flatten_lighting(image), sharpen=1.40, saturation=1.175)
    if normalized == "matte":
        return enhance(suppress_specular(flatten_lighting(image)), sharpen=1.60, saturation=1.2625)
    if normalized == "super":
        return enhance(flatten_lighting(image), sharpen=1.80, saturation=1.35)
    if normalized == "edge":
        return edge_detect(image)
    if normalized == "gray":
        return ImageEnhance.Contrast(image.convert("L")).enhance(1.35).convert("RGB")
    if normalized == "threshold":
        return threshold(image)
    if normalized == "nid_ink":
        return nid_ink(image)
    return image


def order_quad(points: list[list[float]]) -> list[tuple[float, float]]:
    if len(points) != 4:
        raise ValueError("SDK crop requires a 4-point quad")
    pts = [(float(x), float(y)) for x, y in points]
    by_sum = sorted(pts, key=lambda point: point[0] + point[1])
    top_left = by_sum[0]
    bottom_right = by_sum[-1]
    remaining = by_sum[1:3]
    top_right, bottom_left = sorted(remaining, key=lambda point: point[0] - point[1], reverse=True)
    return [top_left, top_right, bottom_right, bottom_left]


def perspective_correct(
    image: Image.Image,
    quad: list[tuple[float, float]],
    target_aspect_ratio: float | None,
) -> Image.Image:
    top_left, top_right, bottom_right, bottom_left = quad
    raw_width = max(distance(top_left, top_right), distance(bottom_left, bottom_right))
    raw_height = max(distance(top_left, bottom_left), distance(top_right, bottom_right))
    width = max(raw_width, 1.0)
    height = max(raw_height, 1.0)

    if target_aspect_ratio and target_aspect_ratio > 0:
        ratio = target_aspect_ratio if width >= height else 1.0 / target_aspect_ratio
        if width / height > ratio:
            height = width / ratio
        else:
            width = height * ratio

    scale = min(1.0, 4096.0 / width, 4096.0 / height)
    out_width = max(1, round(width * scale))
    out_height = max(1, round(height * scale))

    dst = [(0.0, 0.0), (out_width, 0.0), (out_width, out_height), (0.0, out_height)]
    coeffs = perspective_coefficients(dst, quad)
    return image.transform(
        (out_width, out_height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )


def perspective_coefficients(
    dst: list[tuple[float, float]],
    src: list[tuple[float, float]],
) -> list[float]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    return np.linalg.solve(np.asarray(matrix), np.asarray(vector)).tolist()


def downscale_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = sqrt(max_pixels / pixels)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def enhance(image: Image.Image, sharpen: float, saturation: float) -> Image.Image:
    stretched = percentile_stretch(image, clip=0.005, gamma=0.85)
    saturated = ImageEnhance.Color(stretched).enhance(saturation)
    return unsharp_luma(saturated, amount=sharpen)


def percentile_stretch(image: Image.Image, clip: float, gamma: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    low = np.percentile(array, clip * 100, axis=(0, 1))
    high = np.percentile(array, (1 - clip) * 100, axis=(0, 1))
    span = np.maximum(high - low, 1)
    normalized = np.clip((array - low) / span, 0, 1)
    corrected = np.power(normalized, gamma) * 255
    return Image.fromarray(corrected.astype(np.uint8), "RGB")


def flatten_lighting(image: Image.Image) -> Image.Image:
    if cv2 is not None:
        array = np.asarray(image).astype(np.float32)
        luma = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        radius = max(15, int(min(image.size) * 0.20))
        kernel = max(3, radius | 1)
        illumination = cv2.GaussianBlur(luma, (kernel, kernel), 0).astype(np.float32)
        reference = np.percentile(illumination, 90)
        gain = np.clip(reference / np.maximum(illumination, 1), 0.4, 3.0)
        corrected = np.clip(array * gain[:, :, None], 0, 255)
        return Image.fromarray(corrected.astype(np.uint8), "RGB")

    gray = image.convert("L").filter(ImageFilter.GaussianBlur(radius=max(8, min(image.size) // 12)))
    base = np.asarray(image).astype(np.float32)
    light = np.asarray(gray).astype(np.float32)
    reference = np.percentile(light, 90)
    gain = np.clip(reference / np.maximum(light, 1), 0.4, 3.0)
    return Image.fromarray(np.clip(base * gain[:, :, None], 0, 255).astype(np.uint8), "RGB")


def suppress_specular(image: Image.Image) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    luma = np.dot(array, [0.299, 0.587, 0.114])
    onset = np.percentile(luma, 60)
    onset = onset + ((255 - onset) * 0.35)
    hot = np.clip((luma - onset) / max(255 - onset, 1), 0, 1)
    chroma = array.max(axis=2) - array.min(axis=2)
    neutral = np.clip(1 - (chroma / 40), 0, 1)
    weight = (hot * neutral).astype(np.float32)

    if cv2 is not None:
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=max(3, min(image.size) * 0.01))
        fill = cv2.GaussianBlur(array, (0, 0), sigmaX=max(8, min(image.size) * 0.06))
    else:
        weight_img = Image.fromarray(np.clip(weight * 255, 0, 255).astype(np.uint8), "L")
        weight = np.asarray(weight_img.filter(ImageFilter.GaussianBlur(radius=max(2, min(image.size) // 80)))) / 255
        fill = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=max(8, min(image.size) // 20)))).astype(np.float32)

    corrected = array * (1 - weight[:, :, None]) + fill * weight[:, :, None]
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8), "RGB")


def unsharp_luma(image: Image.Image, amount: float) -> Image.Image:
    if cv2 is not None:
        array = np.asarray(image).astype(np.float32)
        luma = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        radius = max(1, int(min(image.size) * 0.005))
        local = cv2.blur(luma, (radius * 2 + 1, radius * 2 + 1))
        target = np.clip(local + (luma - local) * amount, 0, 255)
        scale = target / np.maximum(luma, 1)
        sharpened = np.clip(array * scale[:, :, None], 0, 255)
        return Image.fromarray(sharpened.astype(np.uint8), "RGB")
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(90 * amount), threshold=3))


def edge_detect(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    if cv2 is not None:
        edges = cv2.Canny(gray, 70, 160)
        return Image.fromarray(edges, "L").convert("RGB")
    return image.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")


def nid_ink(image: Image.Image) -> Image.Image:
    """Green channel only: black and red NID ink stay dark, the yellow/orange emblem watermark turns white."""
    return ImageOps.autocontrast(image.convert("RGB").getchannel("G"), cutoff=1).convert("RGB")


def threshold(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    if cv2 is not None:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        out = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        return Image.fromarray(out, "L").convert("RGB")
    return image.convert("L").point(lambda pixel: 255 if pixel > 150 else 0).convert("RGB")
