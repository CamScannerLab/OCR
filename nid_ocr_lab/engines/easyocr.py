from __future__ import annotations

import importlib.util
from importlib.metadata import version
import threading
import time
from pathlib import Path
from typing import Any

from nid_ocr_lab.models import BoundingBox, OCRResult, OCRTextBlock

MODEL_DIR = Path(__file__).resolve().parents[2] / 'benchmark' / 'generated' / 'easyocr'
_READERS: dict[tuple[str, ...], Any] = {}
# The threaded dashboard must not initialize/use a shared reader concurrently.
_READER_LOCK = threading.Lock()


def is_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ('easyocr', 'torch', 'torchvision'))


def language_codes(languages: list[str]) -> tuple[str, ...]:
    aliases = {'eng': 'en', 'ben': 'bn', 'en': 'en', 'bn': 'bn'}
    unknown = set(languages) - aliases.keys()
    if unknown:
        raise ValueError(f'Unsupported EasyOCR languages: {sorted(unknown)}')
    return tuple(sorted({aliases[item] for item in languages} or {'en'}))


def get_reader(codes: tuple[str, ...]) -> tuple[Any, bool]:
    """Caller holds _READER_LOCK through inference."""
    if codes in _READERS:
        return _READERS[codes], True
    from easyocr import Reader

    reader = Reader(
        list(codes), gpu=False, model_storage_directory=str(MODEL_DIR / 'models'),
        user_network_directory=str(MODEL_DIR / 'user_network'), verbose=False,
    )
    _READERS[codes] = reader
    return reader, False


class EasyOCREngine:
    name = 'easyocr'

    def recognize(
        self, image_path: Path, languages: list[str],
        preprocessing: str | None = None, psm: int | None = None,
        *, auto_rotate: bool = False,
    ) -> OCRResult:
        codes = language_codes(languages)
        if not is_available():
            raise RuntimeError('EasyOCR is not installed. Install requirements-easyocr.txt in the dashboard Python environment.')
        started = time.perf_counter()
        with _READER_LOCK:
            acquired = time.perf_counter()
            reader, cached = get_reader(codes)
            initialized = time.perf_counter()
            raw = reader.readtext(
                str(image_path), detail=1, paragraph=False, decoder='greedy',
                batch_size=1, workers=0,
                rotation_info=[90, 180, 270] if auto_rotate else None,
            )
            inferred = time.perf_counter()
        blocks = []
        serializable = []
        for polygon, text, score in raw:
            points = [[float(x), float(y)] for x, y in polygon]
            confidence = float(score)
            serializable.append([points, str(text), confidence])
            if not str(text).strip():
                continue
            xs, ys = zip(*points)
            blocks.append(OCRTextBlock(
                text=str(text).strip(), confidence=confidence,
                bounding_box=BoundingBox(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys)),
            ))
        return OCRResult(
            blocks=blocks, full_text='\n'.join(block.text for block in blocks),
            engine=self.name, language='+'.join(languages) or 'eng',
            preprocessing=preprocessing, latency_ms=(time.perf_counter()-started)*1000,
            metadata={
                'source_image': str(image_path), 'easyocr_languages': list(codes),
                'device': 'cpu', 'reader_cached': cached,
                'version': version('easyocr'),
                'model': 'bengali_g1' if 'bn' in codes else 'english_g2',
                'detector': 'craft',
                'queue_ms': (acquired-started)*1000,
                'initialization_ms': (initialized-acquired)*1000,
                'inference_ms': (inferred-initialized)*1000,
                'rotation_strategy': 'text-box auto rotation' if auto_rotate else 'fixed card rotation',
                'raw_response': serializable,
            },
        )
