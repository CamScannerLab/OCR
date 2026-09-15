#!/usr/bin/env bash
# Download official tessdata_best models (accurate, fine-tunable) and the tesstrain
# Makefile repo. Everything lands in gitignored folders; /opt/homebrew tessdata is untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BEST_DIR="$ROOT/benchmark/generated/tessdata/best"
TESSTRAIN_DIR="$ROOT/training/tesstrain"
BASE_URL="https://github.com/tesseract-ocr/tessdata_best/raw/main"
SYSTEM_TESSDATA="${SYSTEM_TESSDATA:-/opt/homebrew/share/tessdata}"

mkdir -p "$BEST_DIR/script"

fetch() {
  local name="$1"
  local target="$BEST_DIR/$name"
  if [ -s "$target" ]; then
    echo "exists: $name"
    return
  fi
  echo "downloading: $name"
  curl -fL --retry 3 -o "$target.part" "$BASE_URL/$name"
  mv "$target.part" "$target"
}

fetch ben.traineddata
fetch eng.traineddata
fetch script/Bengali.traineddata

# OSD is only published as a single model; reuse the system copy.
if [ ! -e "$BEST_DIR/osd.traineddata" ] && [ -f "$SYSTEM_TESSDATA/osd.traineddata" ]; then
  ln -s "$SYSTEM_TESSDATA/osd.traineddata" "$BEST_DIR/osd.traineddata"
fi

if [ ! -d "$TESSTRAIN_DIR/.git" ]; then
  mkdir -p "$(dirname "$TESSTRAIN_DIR")"
  git clone --depth 1 https://github.com/tesseract-ocr/tesstrain.git "$TESSTRAIN_DIR"
fi

python3 - "$BEST_DIR" "$BASE_URL" "$TESSTRAIN_DIR" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

best, base_url, tesstrain = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
models = []
for path in sorted(best.rglob("*.traineddata")):
    if path.is_symlink():
        continue
    rel = path.relative_to(best).as_posix()
    models.append({
        "name": rel,
        "url": f"{base_url}/{rel}",
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
commit = subprocess.run(["git", "-C", str(tesstrain), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = {"tessdata_best": models, "tesstrain_commit": commit or None}
(best.parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

# tesstrain's proto-model step needs the stock langdata (radical-stroke.txt + script unicharsets).
# Its Makefile fetches them with wget, which macOS lacks, so pre-fetch them with curl.
LANGDATA_DIR="$TESSTRAIN_DIR/data/langdata"
mkdir -p "$LANGDATA_DIR"
SCRIPTS="Arabic Armenian Bengali Bopomofo Canadian_Aboriginal Cherokee Cyrillic Devanagari Ethiopic Georgian Greek Gujarati Gurmukhi Hangul Han Hebrew Hiragana Kannada Katakana Khmer Lao Latin Malayalam Myanmar Ogham Oriya Runic Sinhala Syriac Tamil Telugu Thai"
for name in radical-stroke.txt $(for s in $SCRIPTS; do echo "$s.unicharset"; done); do
  if [ ! -s "$LANGDATA_DIR/$name" ]; then
    curl -fsSL --retry 3 -o "$LANGDATA_DIR/$name.part" "https://github.com/tesseract-ocr/langdata_lstm/raw/main/$name"
    mv "$LANGDATA_DIR/$name.part" "$LANGDATA_DIR/$name"
  fi
done
echo "langdata ready: $(ls "$LANGDATA_DIR" | wc -l | tr -d ' ') files"

# tesstrain's helper scripts (box generation, list split) need python-bidi and Pillow.
# Keep them in an isolated venv instead of the SmartScan dashboard venv.
TRAINING_VENV="$ROOT/training/.venv"
if [ ! -x "$TRAINING_VENV/bin/python" ]; then
  python3 -m venv "$TRAINING_VENV"
fi
"$TRAINING_VENV/bin/pip" install -q "Pillow>=6.2.1" "python-bidi>=0.4"
echo "training venv ready: $TRAINING_VENV"
