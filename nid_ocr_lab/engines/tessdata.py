from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESSDATA_ROOT = PROJECT_ROOT / "benchmark" / "generated" / "tessdata"
BEST_DIR = TESSDATA_ROOT / "best"
CUSTOM_DIR = TESSDATA_ROOT / "custom"

_HASH_CACHE: dict[tuple[str, float], str] = {}
_VERSION_CACHE: dict[tuple[str, float], str | None] = {}


@dataclass(frozen=True)
class TessdataVariant:
    id: str
    label: str
    directory: Path

    def languages(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        names = []
        for path in sorted(self.directory.rglob("*.traineddata")):
            name = path.relative_to(self.directory).with_suffix("").as_posix()
            if name != "osd":
                names.append(name)
        return names

    def model_path(self, language: str) -> Path:
        return self.directory / f"{language}.traineddata"


def system_tessdata_dir() -> Path:
    env = os.environ.get("TESSDATA_PREFIX")
    if env and Path(env).is_dir():
        return Path(env)
    binary = shutil.which("tesseract")
    if binary:
        candidate = Path(binary).resolve().parents[1] / "share" / "tessdata"
        if candidate.is_dir():
            return candidate
    return Path("/opt/homebrew/share/tessdata")


def list_variants() -> list[TessdataVariant]:
    variants = [TessdataVariant("system", "system (tessdata_fast)", system_tessdata_dir())]
    if BEST_DIR.is_dir():
        variants.append(TessdataVariant("best", "tessdata_best", BEST_DIR))
    if CUSTOM_DIR.is_dir():
        for directory in sorted(path for path in CUSTOM_DIR.iterdir() if path.is_dir()):
            variants.append(TessdataVariant(f"custom/{directory.name}", f"custom: {directory.name}", directory))
    return variants


def get_variant(variant_id: str | None) -> TessdataVariant:
    wanted = variant_id or "system"
    for variant in list_variants():
        if variant.id == wanted:
            return variant
    raise ValueError(f"Unknown Tesseract model variant: {wanted}")


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    resolved = path.resolve()
    key = (str(resolved), resolved.stat().st_mtime)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return _HASH_CACHE[key]


def model_version(path: Path) -> str | None:
    """Version string embedded in a traineddata file (e.g. '4.00.00alpha:ben:synth20170629')."""
    if not path.is_file() or shutil.which("combine_tessdata") is None:
        return None
    resolved = path.resolve()
    key = (str(resolved), resolved.stat().st_mtime)
    if key not in _VERSION_CACHE:
        try:
            completed = subprocess.run(
                ["combine_tessdata", "-d", str(resolved)], capture_output=True, text=True, timeout=20
            )
            output = completed.stdout + completed.stderr
        except (OSError, subprocess.TimeoutExpired):
            output = ""
        version = next(
            (line.split(":", 1)[1].strip() for line in output.splitlines() if line.startswith("Version:")),
            None,
        )
        _VERSION_CACHE[key] = version
    return _VERSION_CACHE[key]


def describe_models(variant: TessdataVariant, languages: list[str]) -> list[dict]:
    models = []
    for language in languages:
        path = variant.model_path(language)
        models.append(
            {
                "language": language,
                "path": str(path),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": file_sha256(path),
                "version": model_version(path),
            }
        )
    return models
