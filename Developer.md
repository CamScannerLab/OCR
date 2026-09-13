# Developer Guide

This repository is the Bangladesh NID OCR R&D lab.

It sits after the SmartScan work. SmartScan finds the card, crops/perspective-corrects it, and creates enhancement variants. This repo is for testing OCR engines, inspecting SmartScan output, parsing OCR text into NID fields, and measuring field-level accuracy.

## Current Status

Implemented in this repo:

- Local SmartScan sample dashboard.
- SmartScan dataset indexer.
- Annotation/mask/validation/test image viewer.
- SDK-like crop and perspective correction in Python.
- SDK-like filter preview modes.
- Dashboard OCR controls.
- Tesseract OCR adapter.
- First NID parser stub.
- Field-level evaluation scaffold.
- Dataset manifest example.

Not implemented yet:

- PaddleOCR/RapidOCR adapters.
- Batch OCR benchmark runner from dashboard.
- OCR bounding-box overlay on cropped card.
- Real NID field ground-truth dataset for OCR accuracy.
- Learned correction/confusion model.

## Upstream Projects

The OCR lab depends on the previous SmartScan R&D work:

```text
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK
```

Use them for scanner context:

- `Smart-Scan`: dataset, annotations, masks, training/validation split, model R&D.
- `Smart-Scan-SDK`: Android SDK implementation, crop flow, perspective correction, filter names/behavior.

Important SDK files:

```text
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK/smartscan/src/main/java/com/konasl/smartscan/imaging/PerspectiveCorrector.kt
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK/smartscan/src/main/java/com/konasl/smartscan/imaging/Enhancer.kt
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK/smartscan/src/main/java/com/konasl/smartscan/api/ColorMode.kt
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK/smartscan/src/main/java/com/konasl/smartscan/api/SmartScanActivity.kt
```

## Runtime Setup

### Python

The dashboard launcher uses the existing SmartScan virtual environment first:

```text
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/venv/bin/python
```

That venv currently provides the image-processing packages needed by the dashboard:

```text
Pillow
NumPy
OpenCV / cv2
```

If that venv is missing, `scripts/run_dashboard.sh` falls back to the bundled Codex Python runtime, then to `python3`.

### Tesseract

Tesseract is **not installed inside the Python venv**. It is a separate system binary installed with Homebrew:

```text
/opt/homebrew/bin/tesseract
```

Installed version:

```text
Tesseract 5.5.3
```

Available language data:

```text
eng
ben
osd
```

`eng` and `osd` came with the Homebrew `tesseract` formula. `ben.traineddata` was downloaded separately into:

```text
/opt/homebrew/share/tessdata/ben.traineddata
```

The full Homebrew `tesseract-lang` package was attempted but failed during download because the connection reset. The smaller direct Bengali traineddata download succeeded.

Verify OCR setup:

```bash
tesseract --version
tesseract --list-langs
```

## Run The Dashboard

From this repo:

```bash
bash scripts/run_dashboard.sh
```

Open:

```text
http://127.0.0.1:8765
```

Optional port:

```bash
bash scripts/run_dashboard.sh --port 8766
```

The dashboard is a local Python HTTP server. It does not upload images anywhere.

## Dashboard Behavior

The left panel lists samples discovered from SmartScan paths.

The source panel shows the selected source image, mask, overlay, train image, or validation/test image.

The filter panel is the OCR-ready preview. Normal filter choices always run:

```text
selected real image
  -> annotation quad
  -> SDK-style perspective crop
  -> NID aspect ratio 85.6 / 54.0
  -> downscale to about 2MP max
  -> selected filter
  -> fit inside dashboard panel without scrolling
```

Filter options:

```text
Original
Enhance
Deglare
Matte
Super
Edge
Gray
Threshold
Mask Overlay
```

`Mask Overlay` is different from the normal OCR filters. It is for checking annotation/mask quality against the full selected image.

## Run OCR From Dashboard

The OCR panel supports:

```text
Engine: Tesseract
Language: eng+ben / eng / ben
Run OCR
```

When `Run OCR` is clicked:

```text
selected image
  -> SDK-style crop
  -> selected filter mode
  -> temporary JPEG
  -> Tesseract TSV OCR
  -> OCR JSON
  -> NID parser
  -> parsed field preview
```

If Tesseract is not on `PATH`, the dashboard disables the button and reports that the engine is unavailable.

## Run OCR From CLI

The CLI can run Tesseract directly on an image:

```bash
python3 -m nid_ocr_lab.cli ocr-tesseract path/to/image.jpg --lang eng+ben --output benchmark/ocr_outputs/sample.json
```

Parse OCR JSON into NID fields:

```bash
python3 -m nid_ocr_lab.cli parse-json benchmark/ocr_outputs/sample.json
```

Evaluate OCR JSON files against a manifest:

```bash
python3 -m nid_ocr_lab.cli evaluate benchmark/ocr_outputs configs/dataset_manifest.example.json
```

## Dataset Sources

The dashboard reads these SmartScan paths:

```text
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/dataset
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/baseline_masks
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/training/data
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK/demo/src/main/assets
```

Important roles:

- `dataset/raw`: annotated source dataset images.
- `dataset/annotations`: LabelMe JSON annotation files.
- `dataset/masks`: masks generated from annotations.
- `dataset/overlays`: pre-rendered QA overlays.
- `training/data/train/images`: train split images.
- `training/data/train/masks`: train split masks.
- `training/data/val/images`: validation/test split images.
- `training/data/val/masks`: validation/test split masks.
- `dataset/composites`: synthetic composite images and masks.
- `baseline_masks`: extra mask/edge artifacts.

Current structural health check:

```text
dataset:    284 images, 284 masks, 284 paired
train:      241 images, 241 masks, 241 paired
val/test:    43 images,  43 masks,  43 paired
composites: 1500 images, 1500 masks, 1500 paired
```

Those checks confirm files are paired by name and dimensions. They do not prove each annotation is visually perfect.

## Project Layout

```text
AGENTS.md
Developer.md
README.md

Docs/
  current_study.md
  ocr_pipeline_rnd_plan.md

configs/
  dataset_manifest.example.json

benchmark/
  datasets/
  ocr_outputs/
  reports/

nid_ocr_lab/
  cli.py
  models.py
  pipeline.py
  evaluation.py
  engines/
    base.py
    tesseract.py
  parsers/
    nid_parser.py
  dashboard/
    filters.py
    indexer.py
    server.py
    static/

scripts/
  run_dashboard.sh
```

## Key Code Files

- `nid_ocr_lab/dashboard/server.py`: local HTTP dashboard server, image endpoints, OCR endpoint.
- `nid_ocr_lab/dashboard/indexer.py`: discovers SmartScan samples and dataset health.
- `nid_ocr_lab/dashboard/filters.py`: image filters, SDK-style crop, mask overlay rendering.
- `nid_ocr_lab/dashboard/static/app.js`: dashboard state, source/filter selection, OCR button.
- `nid_ocr_lab/engines/tesseract.py`: Tesseract adapter using TSV output.
- `nid_ocr_lab/parsers/nid_parser.py`: first simple parser for NID number, DOB, and labeled fields.
- `nid_ocr_lab/evaluation.py`: field-level exact-match evaluation.
- `nid_ocr_lab/cli.py`: command-line entry point.

## Tools Used During Setup

### Shell tools

- `find`: discovered SmartScan datasets, images, annotations, masks, and SDK files.
- `rg`: searched SmartScan SDK and R&D docs for filter/crop/perspective code.
- `sed`: inspected source files and docs.
- `git`: initialized this repository and created the initial commit.

### Python tools

- SmartScan venv Python: dashboard runtime.
- Pillow: image loading, resizing, JPEG output, fallback filters.
- NumPy: pixel operations, perspective transform coefficients, filter math.
- OpenCV (`cv2`): available through the SmartScan venv; used by filter implementations when present.
- Python `http.server`: local dashboard web server.

### OCR tools

- Tesseract system binary: OCR engine used by the dashboard and CLI.
- Tesseract TSV output: used to capture text, confidence, and bounding boxes.
- Tesseract language data: `eng`, `ben`, `osd`.

### Package/install tools

- Homebrew: installed the Tesseract system binary.
- `curl`: downloaded `ben.traineddata` directly after full `tesseract-lang` download failed.

### SmartScan/annotation tools referenced

- LabelMe JSON annotations: source of document/card quads.
- SmartScan masks: generated from LabelMe annotations.
- SmartScan train/validation split: used for dashboard inspection and future OCR benchmarking.
- Android SDK source: used as the reference for crop and enhancement behavior.

### Codex workspace tools used

- `apply_patch`: created and updated source/docs files in this repo.
- `exec_command`: inspected files, ran Python checks, initialized git, installed/verified Tesseract, and started the dashboard.
- `write_stdin`: stopped and restarted long-running dashboard server sessions.
- `open_in_codex`: opened the local dashboard URL in the Codex app panel.
- `load_workspace_dependencies`: found the bundled Python fallback runtime before switching to the SmartScan venv.

Verification commands used during setup:

```bash
/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/venv/bin/python -m compileall nid_ocr_lab
tesseract --list-langs
```

## Git

This repo was initialized during the OCR R&D setup.

Initial commit:

```text
77c9246 Initialize NID OCR R&D lab
```

There may be later uncommitted dashboard/OCR changes after that commit. Check with:

```bash
git status --short
```

## Next Development Steps

1. Add a button to save dashboard OCR results into `benchmark/ocr_outputs/`.
2. Add batch OCR runs across validation/test samples and filter modes.
3. Draw Tesseract bounding boxes over the cropped OCR image.
4. Add PaddleOCR or RapidOCR adapter.
5. Build real ground-truth manifests for NID field-level accuracy.
6. Compare `eng`, `ben`, and `eng+ben` across `enhance`, `deglare`, `matte`, and `threshold`.
