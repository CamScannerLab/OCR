from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = PROJECT_ROOT / "benchmark" / "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 30
PDF_RENDER_DPI = 300
IMAGE_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "TIFF": "tif", "BMP": "bmp"}
UPLOAD_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")


def upload_root() -> Path:
    return UPLOAD_ROOT


def save_upload(filename: str, data: bytes) -> dict:
    if not data:
        raise ValueError("Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    upload_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    directory = upload_root() / upload_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if data.startswith(b"%PDF"):
            (directory / "original.pdf").write_bytes(data)
            pages = render_pdf_pages(data, directory)
            kind = "pdf"
        else:
            pages = [save_image_page(data, directory)]
            kind = "image"
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    record = {
        "id": upload_id,
        "filename": Path(filename or "upload").name,
        "kind": kind,
        "bytes": len(data),
        "created": datetime.now().isoformat(timespec="seconds"),
        "render_dpi": PDF_RENDER_DPI if kind == "pdf" else None,
        "pages": pages,
    }
    (directory / "upload.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return with_paths(record, directory)


def save_image_page(data: bytes, directory: Path) -> str:
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Unsupported file: upload a JPEG, PNG, WEBP, TIFF, BMP image or a PDF") from exc
    extension = IMAGE_FORMATS.get(image.format or "")
    if extension is None:
        raise ValueError(f"Unsupported image format: {image.format}")
    (directory / f"original.{extension}").write_bytes(data)
    dpi = image.info.get("dpi")
    page = ImageOps.exif_transpose(image)
    if page.mode not in ("RGB", "L"):
        page = page.convert("RGB")
    name = "page-001.png"
    page.save(directory / name, **({"dpi": dpi} if dpi else {}))
    return name


def render_pdf_pages(data: bytes, directory: Path) -> list[str]:
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc
    try:
        if len(document) == 0:
            raise ValueError("PDF has no pages")
        if len(document) > MAX_PDF_PAGES:
            raise ValueError(f"PDF has {len(document)} pages; limit is {MAX_PDF_PAGES}")
        names = []
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=PDF_RENDER_DPI / 72)
            image = bitmap.to_pil().convert("RGB")
            name = f"page-{index + 1:03d}.png"
            image.save(directory / name, dpi=(PDF_RENDER_DPI, PDF_RENDER_DPI))
            names.append(name)
            page.close()
        return names
    finally:
        document.close()


def with_paths(record: dict, directory: Path) -> dict:
    return {**record, "directory": str(directory), "page_paths": [str(directory / name) for name in record["pages"]]}


def list_uploads() -> list[dict]:
    root = upload_root()
    if not root.is_dir():
        return []
    records = []
    for directory in sorted(root.iterdir(), reverse=True):
        meta = directory / "upload.json"
        if not directory.is_dir() or not meta.is_file():
            continue
        try:
            record = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(with_paths(record, directory))
    return records


def delete_upload(upload_id: str) -> None:
    if not UPLOAD_ID_PATTERN.match(upload_id or ""):
        raise ValueError("Invalid upload id")
    root = upload_root().resolve()
    directory = (root / upload_id).resolve()
    if directory.parent != root or not directory.is_dir():
        raise ValueError("Upload not found")
    shutil.rmtree(directory)


def upload_samples() -> list[dict]:
    """Uploads shaped like indexer samples so the dashboard sidebar can list them."""
    samples = []
    for record in list_uploads():
        paths = record["page_paths"]
        images = {
            (f"page {index + 1}" if len(paths) > 1 or record["kind"] == "pdf" else "uploaded image"): path
            for index, path in enumerate(paths)
        }
        samples.append(
            {
                "id": f"upload:{record['id']}",
                "source": "upload",
                "upload_id": record["id"],
                "filename": record["filename"],
                "kind": record["kind"],
                "raw": paths[0] if paths else None,
                "annotation": None,
                "mask": None,
                "overlay": None,
                "variants": {},
                "images": images,
            }
        )
    return samples
