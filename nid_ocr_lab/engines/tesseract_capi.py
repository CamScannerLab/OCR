"""Minimal ctypes binding to the installed libtesseract / libleptonica C API.

The CLI only returns final text; the C API exposes the intermediate results the
dashboard step inspector shows (binary image, blobs, layout, per-line crops, choices).
Same library as the `tesseract` binary, so results match the CLI for the same settings.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
from ctypes import POINTER, byref, c_bool, c_char_p, c_float, c_int, c_size_t, c_void_p
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

RIL_BLOCK, RIL_PARA, RIL_TEXTLINE, RIL_WORD, RIL_SYMBOL = range(5)
LEVEL_NAMES = {RIL_BLOCK: "block", RIL_PARA: "paragraph", RIL_TEXTLINE: "line", RIL_WORD: "word", RIL_SYMBOL: "symbol"}
# PolyBlockType from tesseract/publictypes.h, in enum order.
BLOCK_TYPES = (
    "unknown", "flowing_text", "heading_text", "pullout_text", "equation", "inline_equation", "table",
    "vertical_text", "caption_text", "flowing_image", "heading_image", "pullout_image", "horizontal_line",
    "vertical_line", "noise",
)
TEXT_BLOCK_TYPES = {"flowing_text", "heading_text", "pullout_text", "equation", "inline_equation", "table", "vertical_text", "caption_text"}


def _library_candidates(stem: str) -> list[str]:
    candidates = []
    binary = shutil.which("tesseract")
    if binary:
        lib_dir = Path(binary).resolve().parents[1] / "lib"
        candidates += [str(path) for path in sorted(lib_dir.glob(f"lib{stem}.*dylib"))]
        candidates += [str(path) for path in sorted(lib_dir.glob(f"lib{stem}.so*"))]
    found = ctypes.util.find_library(stem)
    if found:
        candidates.append(found)
    return candidates


def _load(stem: str) -> ctypes.CDLL:
    for candidate in _library_candidates(stem):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError(f"lib{stem} was not found next to the tesseract binary or on the library path")


def _declare(lib: ctypes.CDLL, name: str, restype: Any, *argtypes: Any) -> None:
    function = getattr(lib, name)
    function.restype = restype
    function.argtypes = list(argtypes)


@lru_cache(maxsize=1)
def libraries() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    tess = _load("tesseract")
    lept = _load("leptonica")
    P = c_void_p
    INT = POINTER(c_int)

    _declare(lept, "pixRead", P, c_char_p)
    _declare(lept, "pixReadMem", P, c_void_p, c_size_t)
    _declare(lept, "pixDestroy", None, POINTER(c_void_p))
    _declare(lept, "pixClone", P, P)
    _declare(lept, "pixConvertTo8", P, P, c_int)
    _declare(lept, "pixGetDepth", c_int, P)
    _declare(lept, "pixGetWidth", c_int, P)
    _declare(lept, "pixGetHeight", c_int, P)
    _declare(lept, "pixGetYRes", c_int, P)
    _declare(lept, "pixSetResolution", c_int, P, c_int, c_int)
    _declare(lept, "pixWriteMemPng", c_int, POINTER(c_void_p), POINTER(c_size_t), P, c_float)
    _declare(lept, "pixOtsuAdaptiveThreshold", c_int, P, c_int, c_int, c_int, c_int, c_float, POINTER(c_void_p), POINTER(c_void_p))
    _declare(lept, "pixSauvolaBinarizeTiled", c_int, P, c_int, c_float, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p))
    _declare(lept, "boxaGetCount", c_int, P)
    _declare(lept, "boxaGetBoxGeometry", c_int, P, c_int, INT, INT, INT, INT)
    _declare(lept, "boxaDestroy", None, POINTER(c_void_p))
    _declare(lept, "lept_free", None, c_void_p)

    _declare(tess, "TessVersion", c_char_p)
    _declare(tess, "TessBaseAPICreate", P)
    _declare(tess, "TessBaseAPIDelete", None, P)
    _declare(tess, "TessBaseAPIEnd", None, P)
    _declare(tess, "TessBaseAPIInit4", c_int, P, c_char_p, c_char_p, c_int, P, c_int, P, P, c_size_t, c_int)
    _declare(tess, "TessBaseAPISetVariable", c_int, P, c_char_p, c_char_p)
    _declare(tess, "TessBaseAPISetPageSegMode", None, P, c_int)
    _declare(tess, "TessBaseAPISetImage2", None, P, P)
    _declare(tess, "TessBaseAPISetInputName", None, P, c_char_p)
    _declare(tess, "TessBaseAPIGetSourceYResolution", c_int, P)
    _declare(tess, "TessBaseAPIGetThresholdedImage", P, P)
    _declare(tess, "TessBaseAPIGetThresholdedImageScaleFactor", c_int, P)
    _declare(tess, "TessBaseAPIGetConnectedComponents", P, P, POINTER(c_void_p))
    _declare(tess, "TessBaseAPIAnalyseLayout", P, P)
    _declare(tess, "TessBaseAPIRecognize", c_int, P, P)
    _declare(tess, "TessBaseAPIGetIterator", P, P)
    _declare(tess, "TessBaseAPIGetUTF8Text", c_void_p, P)
    _declare(tess, "TessBaseAPIMeanTextConf", c_int, P)
    _declare(tess, "TessDeleteText", None, c_void_p)
    _declare(tess, "TessPageIteratorDelete", None, P)
    _declare(tess, "TessPageIteratorBegin", None, P)
    _declare(tess, "TessPageIteratorNext", c_int, P, c_int)
    _declare(tess, "TessPageIteratorIsAtBeginningOf", c_int, P, c_int)
    _declare(tess, "TessPageIteratorBoundingBox", c_int, P, c_int, INT, INT, INT, INT)
    _declare(tess, "TessPageIteratorBlockType", c_int, P)
    _declare(tess, "TessPageIteratorBaseline", c_int, P, c_int, INT, INT, INT, INT)
    _declare(tess, "TessPageIteratorGetBinaryImage", P, P, c_int)
    _declare(tess, "TessPageIteratorGetImage", P, P, c_int, c_int, P, INT, INT)
    _declare(tess, "TessResultIteratorDelete", None, P)
    _declare(tess, "TessResultIteratorGetPageIterator", P, P)
    _declare(tess, "TessResultIteratorNext", c_int, P, c_int)
    _declare(tess, "TessResultIteratorGetUTF8Text", c_void_p, P, c_int)
    _declare(tess, "TessResultIteratorConfidence", c_float, P, c_int)
    _declare(tess, "TessResultIteratorWordRecognitionLanguage", c_char_p, P)
    _declare(tess, "TessResultIteratorWordIsFromDictionary", c_int, P)
    _declare(tess, "TessResultIteratorGetChoiceIterator", P, P)
    _declare(tess, "TessChoiceIteratorDelete", None, P)
    _declare(tess, "TessChoiceIteratorNext", c_int, P)
    _declare(tess, "TessChoiceIteratorGetUTF8Text", c_char_p, P)
    _declare(tess, "TessChoiceIteratorConfidence", c_float, P)
    return tess, lept


def is_available() -> bool:
    try:
        libraries()
    except (RuntimeError, AttributeError):
        return False
    return True


def version() -> str:
    tess, _ = libraries()
    return tess.TessVersion().decode()


class Pix:
    """Owned Leptonica image; destroyed on close."""

    def __init__(self, pointer: int | None) -> None:
        if not pointer:
            raise RuntimeError("Leptonica returned no image")
        self.pointer = c_void_p(pointer)

    @classmethod
    def read(cls, path: Path) -> "Pix":
        _, lept = libraries()
        pointer = lept.pixRead(str(path).encode())
        if not pointer:
            raise RuntimeError(f"Leptonica could not read {path}")
        return cls(pointer)

    @property
    def size(self) -> tuple[int, int, int]:
        _, lept = libraries()
        return lept.pixGetWidth(self.pointer), lept.pixGetHeight(self.pointer), lept.pixGetDepth(self.pointer)

    @property
    def y_resolution(self) -> int:
        return libraries()[1].pixGetYRes(self.pointer)

    def to_pil(self) -> Image.Image:
        _, lept = libraries()
        data = c_void_p()
        size = c_size_t()
        if lept.pixWriteMemPng(byref(data), byref(size), self.pointer, 0.0) != 0 or not data.value:
            raise RuntimeError("Leptonica could not encode the image")
        try:
            payload = ctypes.string_at(data.value, size.value)
        finally:
            lept.lept_free(data)
        image = Image.open(BytesIO(payload))
        image.load()
        return image

    def grey(self) -> "Pix":
        """Tesseract's own grey conversion (pixConvertTo8), used before adaptive thresholding."""
        return Pix(libraries()[1].pixConvertTo8(self.pointer, 0))

    def close(self) -> None:
        if self.pointer and self.pointer.value:
            libraries()[1].pixDestroy(byref(self.pointer))
            self.pointer = c_void_p()

    def __enter__(self) -> "Pix":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def leptonica_threshold(grey: Pix, method: str, **kwargs: Any) -> tuple[Image.Image, Image.Image]:
    """Run the same Leptonica call Tesseract makes; returns (threshold map, binary)."""
    _, lept = libraries()
    thresholds = c_void_p()
    binary = c_void_p()
    if method == "adaptive_otsu":
        status = lept.pixOtsuAdaptiveThreshold(
            grey.pointer, kwargs["tile"], kwargs["tile"], kwargs["half_smooth"], kwargs["half_smooth"],
            kwargs["score_fraction"], byref(thresholds), byref(binary),
        )
    elif method == "sauvola":
        status = lept.pixSauvolaBinarizeTiled(
            grey.pointer, kwargs["half_window"], kwargs["kfactor"], kwargs["nx"], kwargs["ny"], byref(thresholds), byref(binary),
        )
    else:
        raise ValueError(f"Unsupported Leptonica threshold method {method}")
    if status != 0:
        raise RuntimeError(f"Leptonica {method} thresholding failed")
    with Pix(thresholds.value) as map_pix, Pix(binary.value) as binary_pix:
        return map_pix.to_pil(), binary_pix.to_pil()


def _boxes(boxa: int | None) -> list[dict[str, int]]:
    _, lept = libraries()
    if not boxa:
        return []
    handle = c_void_p(boxa)
    try:
        boxes = []
        x, y, w, h = c_int(), c_int(), c_int(), c_int()
        for index in range(lept.boxaGetCount(handle)):
            lept.boxaGetBoxGeometry(handle, index, byref(x), byref(y), byref(w), byref(h))
            boxes.append({"x": x.value, "y": y.value, "width": w.value, "height": h.value})
        return boxes
    finally:
        lept.boxaDestroy(byref(handle))


def _take_text(pointer: int | None) -> str:
    if not pointer:
        return ""
    tess, _ = libraries()
    try:
        return ctypes.string_at(pointer).decode("utf-8", errors="replace")
    finally:
        tess.TessDeleteText(pointer)


class TessAPI:
    """One TessBaseAPI handle; create per request and close after use."""

    def __init__(self, datapath: Path, language: str, oem: int = 1) -> None:
        self.tess, self.lept = libraries()
        self.handle = c_void_p(self.tess.TessBaseAPICreate())
        self.image: Pix | None = None
        status = self.tess.TessBaseAPIInit4(self.handle, str(datapath).encode(), language.encode(), oem, None, 0, None, None, 0, 0)
        if status != 0:
            self.close()
            raise RuntimeError(f"Tesseract could not load '{language}' from {datapath}")

    def set_variable(self, name: str, value: Any) -> None:
        if not self.tess.TessBaseAPISetVariable(self.handle, name.encode(), str(value).encode()):
            raise ValueError(f"Tesseract rejected variable {name}={value}")

    def set_psm(self, psm: int) -> None:
        self.tess.TessBaseAPISetPageSegMode(self.handle, int(psm))

    def set_image(self, path: Path) -> Pix:
        """Read with Leptonica like the CLI does, so image DPI metadata is honoured the same way."""
        self.image = Pix.read(path)
        self.tess.TessBaseAPISetInputName(self.handle, str(path).encode())
        self.tess.TessBaseAPISetImage2(self.handle, self.image.pointer)
        return self.image

    def source_resolution(self) -> int:
        return self.tess.TessBaseAPIGetSourceYResolution(self.handle)

    def thresholded_image(self) -> tuple[Image.Image, int]:
        with Pix(self.tess.TessBaseAPIGetThresholdedImage(self.handle)) as pix:
            return pix.to_pil(), self.tess.TessBaseAPIGetThresholdedImageScaleFactor(self.handle)

    def connected_components(self) -> list[dict[str, int]]:
        return _boxes(self.tess.TessBaseAPIGetConnectedComponents(self.handle, None))

    def layout(self, recognize: bool) -> list[dict[str, Any]]:
        """Flat element list (blocks .. symbols) in reading order, with text data when recognized."""
        if recognize:
            if self.tess.TessBaseAPIRecognize(self.handle, None) != 0:
                raise RuntimeError("Tesseract recognition failed")
            result = self.tess.TessBaseAPIGetIterator(self.handle)
            page = self.tess.TessResultIteratorGetPageIterator(result) if result else None
        else:
            result = None
            page = self.tess.TessBaseAPIAnalyseLayout(self.handle)
        if not page:
            return []
        try:
            return list(self._walk(c_void_p(page), c_void_p(result) if result else None))
        finally:
            if result:
                self.tess.TessResultIteratorDelete(c_void_p(result))
            else:
                self.tess.TessPageIteratorDelete(c_void_p(page))

    def _walk(self, page: c_void_p, result: c_void_p | None) -> Iterator[dict[str, Any]]:
        tess = self.tess
        left, top, right, bottom = c_int(), c_int(), c_int(), c_int()
        tess.TessPageIteratorBegin(page)
        ids = {level: -1 for level in LEVEL_NAMES}
        while True:
            # Each element starts one or more levels at once; emit the outermost first.
            for level in LEVEL_NAMES:
                if not tess.TessPageIteratorIsAtBeginningOf(page, level):
                    continue
                if not tess.TessPageIteratorBoundingBox(page, level, byref(left), byref(top), byref(right), byref(bottom)):
                    continue
                ids[level] += 1
                element: dict[str, Any] = {
                    "level": LEVEL_NAMES[level],
                    "id": ids[level],
                    "parent": ids[level - 1] if level > RIL_BLOCK else None,
                    "box": {"x": left.value, "y": top.value, "width": right.value - left.value, "height": bottom.value - top.value},
                }
                if level == RIL_BLOCK:
                    block_type = tess.TessPageIteratorBlockType(page)
                    element["block_type"] = BLOCK_TYPES[block_type] if 0 <= block_type < len(BLOCK_TYPES) else str(block_type)
                if level == RIL_TEXTLINE:
                    x1, y1, x2, y2 = c_int(), c_int(), c_int(), c_int()
                    if tess.TessPageIteratorBaseline(page, level, byref(x1), byref(y1), byref(x2), byref(y2)):
                        element["baseline"] = [x1.value, y1.value, x2.value, y2.value]
                    element["_line_images"] = self._line_images(page)
                if result is not None and level >= RIL_TEXTLINE:
                    element["text"] = _take_text(tess.TessResultIteratorGetUTF8Text(result, level)).strip()
                    element["confidence"] = round(float(tess.TessResultIteratorConfidence(result, level)), 2)
                    if level == RIL_WORD:
                        language = tess.TessResultIteratorWordRecognitionLanguage(result)
                        element["language"] = language.decode() if language else None
                        element["from_dictionary"] = bool(tess.TessResultIteratorWordIsFromDictionary(result))
                    if level == RIL_SYMBOL:
                        element["choices"] = self._choices(result)
                yield element
            if not tess.TessPageIteratorNext(page, RIL_SYMBOL):
                break

    def _line_images(self, page: c_void_p) -> dict[str, Any]:
        tess = self.tess
        images: dict[str, Any] = {}
        left, top = c_int(), c_int()
        if self.image is not None:
            pointer = tess.TessPageIteratorGetImage(page, RIL_TEXTLINE, 2, self.image.pointer, byref(left), byref(top))
            if pointer:
                with Pix(pointer) as pix:
                    images["original"] = pix.to_pil()
                    images["original_origin"] = [left.value, top.value]
        pointer = tess.TessPageIteratorGetBinaryImage(page, RIL_TEXTLINE)
        if pointer:
            with Pix(pointer) as pix:
                images["binary"] = pix.to_pil()
        return images

    def _choices(self, result: c_void_p) -> list[dict[str, Any]]:
        tess = self.tess
        pointer = tess.TessResultIteratorGetChoiceIterator(result)
        if not pointer:
            return []
        handle = c_void_p(pointer)
        choices = []
        try:
            while True:
                text = tess.TessChoiceIteratorGetUTF8Text(handle)
                choices.append(
                    {"text": text.decode("utf-8", errors="replace") if text else "", "confidence": round(float(tess.TessChoiceIteratorConfidence(handle)), 2)}
                )
                if not tess.TessChoiceIteratorNext(handle) or len(choices) >= 5:
                    break
        finally:
            tess.TessChoiceIteratorDelete(handle)
        return choices

    def text(self) -> str:
        return _take_text(self.tess.TessBaseAPIGetUTF8Text(self.handle))

    def mean_confidence(self) -> int:
        return self.tess.TessBaseAPIMeanTextConf(self.handle)

    def close(self) -> None:
        if self.handle and self.handle.value:
            self.tess.TessBaseAPIEnd(self.handle)
            self.tess.TessBaseAPIDelete(self.handle)
            self.handle = c_void_p()
        if self.image is not None:
            self.image.close()
            self.image = None

    def __enter__(self) -> "TessAPI":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
