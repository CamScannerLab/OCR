# Developer Guide

This repository is the Bangladesh NID OCR R&D lab.

It sits after the SmartScan work. SmartScan finds the card, crops/perspective-corrects it, and creates enhancement variants. This repo is for testing OCR engines, inspecting SmartScan output, parsing OCR text into NID fields, and measuring field-level accuracy.

Production direction: local OCR built on Tesseract, fine-tuned on our own NID lines if needed. Gemini and LM Studio vision engines are comparison baselines only and will not ship.

## Current Status

Implemented in this repo:

- Local SmartScan sample dashboard.
- SmartScan dataset indexer.
- Annotation/mask/validation/test image viewer.
- SDK-like crop and perspective correction in Python.
- SDK-like filter preview modes.
- Dashboard OCR controls.
- Dashboard upload of any image or PDF, OCR'd as uploaded or through the filter preview.
- Tesseract adapter with explicit model folder (fast/best/custom), OEM, PSM, DPI and line-level output.
- Line labeling in the dashboard, producing tesstrain ground truth (`.png` + `.gt.txt`).
- Card-level train/eval split, tesstrain fine-tuning wrapper, and line CER evaluation CLI.
- PaddleOCR and EasyOCR adapters; Gemini/LM Studio vision baselines (test only).
- NID parser with Bengali/English label routing and real-date validation (still heuristic).
- Field-level evaluation: annotated fields only, per-script accuracy, CER.
- Dataset manifest example.

Not implemented yet:

- RapidOCR adapter.
- Batch OCR benchmark runner from dashboard.
- OCR bounding-box overlay on cropped card.
- Drawing a missing line box by hand in the labeling panel (only Tesseract-detected lines can be labeled).
- Synthetic training lines via `text2image`.
- Field-ROI OCR (per-field crops with script-specific profiles).
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

Tesseract 5.5.3 is a Homebrew system binary (`/opt/homebrew/bin/tesseract`), not a venv package. The Homebrew training tools (`lstmtraining`, `lstmeval`, `combine_tessdata`, `unicharset_extractor`, `text2image`) are installed alongside it.

The adapter picks a **model folder** (dashboard "Model" select, CLI `--variant`):

| Variant | Folder | Notes |
|---|---|---|
| `system` | `/opt/homebrew/share/tessdata` | `eng`, `ben`, `osd`. Both `eng` and `ben` are **tessdata_fast** (`4.00.00alpha:*:synth20170629`, 8-bit integer LSTM): quickest, least accurate, **cannot be fine-tuned**. |
| `best` | `benchmark/generated/tessdata/best/` | tessdata_best `ben`, `eng`, `script/Bengali` (float LSTM, trainable). `osd` is a symlink to the system copy. |
| `custom/<name>` | `benchmark/generated/tessdata/custom/<name>/` | Fine-tuned models from `train-tesseract`, with `eng`/`ben`/`osd` symlinked from best. |

Set up or refresh the best models, tesstrain, and its langdata (all gitignored; URLs + sha256 in `benchmark/generated/tessdata/MANIFEST.json`):

```bash
bash scripts/fetch_tessdata.sh
```

How the adapter calls Tesseract:

- `tesseract <png> stdout --tessdata-dir <variant> -l <langs> --oem 1 --psm <psm> [--dpi N] -c tessedit_create_tsv=1`.
  The TSV renderer is set by parameter because the `tsv` config file only exists in the system tessdata folder; with another folder, the `tsv` config name silently produced empty output.
- `--dpi` is passed only when known (explicit value or image metadata, e.g. PDF pages rendered at 300 DPI). Otherwise Tesseract estimates it from text size, which is far more accurate for camera crops than a fixed 300 (a 1780 px card estimates ≈ 570 DPI).
- Output blocks are **lines**, rebuilt from TSV page/block/paragraph/line numbers; words are kept in `metadata.words`. Paragraphs are separated by a blank line in `full_text`.
- Metadata records the command, model paths, sha256, embedded model version, DPI source, and timing.

Language order matters: the first language's params model is used. Compare `ben+eng`, `eng+ben`, and `script/Bengali+eng` on the same card.

Verify setup:

```bash
tesseract --version
tesseract --tessdata-dir benchmark/generated/tessdata/best --list-langs
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

The dashboard is a local Python HTTP server. Tesseract, PaddleOCR, and EasyOCR process images locally; selecting Gemini or a remote LM Studio endpoint sends images to that configured service.

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

## Upload Any Image Or PDF

Use **Choose image / PDF from computer…** at the top of the OCR panel (or **Upload image / PDF** / drag-and-drop in the sidebar). Uploads persist in gitignored `benchmark/uploads/<timestamp>-<id>/` and appear at the top of the sample list with a ✕ delete button.

- Images (JPEG/PNG/WEBP/TIFF/BMP): the original file is kept, and `page-001.png` is written after applying EXIF orientation.
- PDFs: each page is rendered with pypdfium2 at 300 DPI to `page-NNN.png` (max 30 pages, 50 MB). Pick the page in the Source select. Embedded PDF text layers are ignored; pages are OCR'd as images.

Uploads may be real NID cards: they stay local and gitignored. Delete them when done.

## Run OCR From Dashboard

OCR controls:

```text
Engine:     Tesseract / EasyOCR / PaddleOCR / LM Studio Vision (test only) / Gemini Vision (test only)
OCR input:  As uploaded (no crop/filter/resize) | Filter preview (SmartScan crop + filter)
Language:   ben+eng / eng+ben / eng / ben / script/Bengali+eng
Rotation:   auto / 0 / 90 / 180 / 270
Tesseract:  Model (system fast / best / custom), PSM (6, 4, 3, 11, 7), Strategy (single / sweep)
```

The Filter / OCR input preview follows the Rotation select: 0/90/180/270 rotate it clockwise exactly as the OCR request does, and `auto` shows the angle chosen by the last OCR run on that image (heading `auto → N°`). The Source panel stays unrotated so annotation overlays line up.

`As uploaded` is the default for uploads and for samples without an annotation; `Filter preview` is the default for annotated SmartScan samples. OCR inputs are always lossless PNG.

Tesseract request flow:

```text
selected image -> as uploaded | SDK crop + filter -> PNG
  -> rotation auto: one OSD pass (system osd); accept if confidence >= 2
       otherwise OCR at 0/90/180/270 with the chosen PSM and keep the highest sum of word confidences
  -> strategy single: one OCR call at the chosen PSM (reused from the rotation check when possible)
     strategy sweep:  PSM 3/4/6/11, each shown separately with a heuristic score (not accuracy); the parser gets one PSM's text, never a union
  -> line-level OCR JSON -> NID parser -> run saved to benchmark/generated/runs/<run_id> (newest 50 kept)
```

Typical call counts: 2 when OSD is confident, 5 when it is not (OSD fails on sideways cards with "Too few characters"), 1 with manual rotation. The previous implementation always ran 16 calls and concatenated all PSM outputs for parsing.

The run summary shows model versions, OCR call count, DPI source, orientation decision, and total request latency.

## LM Studio Vision (test only)

- Default model: `qwen2.5-vl-7b-instruct` when LM Studio has it loaded; override with `LM_STUDIO_VISION_MODEL` or the Model select. Embedding models are hidden from the list.
- Requests send `reasoning_effort: "none"` (`LM_STUDIO_REASONING_EFFORT`, set `default` to omit it). Thinking models such as `qwen/qwen3.5-9b` otherwise spend the whole `max_tokens` budget (`LM_STUDIO_MAX_TOKENS`, default 900) reasoning, return empty content, and the dashboard showed "LM Studio did not return JSON fields". `/no_think` and `chat_template_kwargs.enable_thinking=false` did not disable thinking in LM Studio; `reasoning_effort` did.

## Label Lines For Tesseract Training

After a Tesseract run, the **Label lines** panel lists every detected line with its crop, OCR text, script, and confidence.

1. Set **Dataset** (default `nid_ben`) and **Card id** (defaults to the upload id or sample id; keep one id per physical card).
2. Fix the text so it matches the crop **exactly**, including label words like `নাম:` if they are inside the crop. Editing ticks the row.
3. Tick only verified lines and press **Save selected**.

Each saved line writes to gitignored `benchmark/training/ground-truth/<dataset>/`:

```text
<card>__<run_id>__<nn>.png      line crop from the exact OCR input (vertical padding ~15% of line height, 8 px horizontal)
<card>__<run_id>__<nn>.gt.txt   NFC-normalized single-line text
<card>__<run_id>__<nn>.json     card id, bbox, script, original OCR text, model/language/source
```

Only lines Tesseract detected can be labeled; if a line was missed entirely, try another PSM or rotation first.

## Train And Evaluate Tesseract

Workflow (run with the dashboard interpreter):

```bash
PY="/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/venv/bin/python"
$PY -m nid_ocr_lab.cli training-split --dataset nid_ben --eval-ratio 0.15 --seed 0
$PY -m nid_ocr_lab.cli eval-tesseract --dataset nid_ben --models system:ben,best:ben
$PY -m nid_ocr_lab.cli train-tesseract --dataset nid_ben --model nid_ben --start-model ben --max-iterations 3000
$PY -m nid_ocr_lab.cli eval-tesseract --dataset nid_ben --models best:ben,custom/nid_ben:nid_ben
```

- `training-split` splits **by card id**, so held-out cards never contribute training lines. It writes `split.json` in the dataset folder.
- `train-tesseract` runs tesstrain (`training/tesstrain`, gitignored) with `TESSDATA=best`, `LANG_TYPE=Indic`, `PSM=7`. It replaces tesstrain's random line split with the card split, then copies the result to `benchmark/generated/tessdata/custom/<model>/` with `model.json`. It appears in the dashboard Model select after a dashboard restart or status refresh.
- Watch the log line `Code range changed from X to Y`. It means the training text contains characters missing from the start model's character set. The output layer is then re-initialized and needs thousands of iterations to recover; a 100-iteration smoke test on mixed Bengali/English lines went from 0.8% to 72% CER. Train `ben` on Bengali-script lines (or start from `script/Bengali` for mixed lines), and give it enough iterations.
- tesstrain's helper scripts need `python-bidi` and Pillow; `scripts/fetch_tessdata.sh` installs them into gitignored `training/.venv`, and the runner passes that interpreter as `PY_CMD`.
- tesstrain requires **GNU make >= 4.2** (macOS ships 3.81): `brew install make` provides `gmake`, which the runner finds automatically. The runner passes only relative paths to make because tesstrain recipes do not quote paths and the project path contains `R&D`.
- `eval-tesseract` OCRs each held-out line crop with PSM 7 per model and writes CER, exact-line accuracy, per-script numbers, latency, and the first 50 errors to `benchmark/reports/tesseract_eval_<dataset>_<ts>.json`.
- Establish the base-model numbers first; only train when the errors are recognition errors on correctly cropped lines. Fine-tuning becomes meaningful with hundreds to thousands of verified lines from many cards; small datasets overfit.

## Run OCR From CLI

Run Tesseract directly on an image (defaults: system variant, OEM 1, PSM 6, DPI from metadata or Tesseract's estimate):

```bash
python3 -m nid_ocr_lab.cli ocr-tesseract path/to/image.png --lang ben+eng --variant best --psm 6 --output benchmark/ocr_outputs/sample.json
```

Parse OCR JSON into NID fields:

```bash
python3 -m nid_ocr_lab.cli parse-json benchmark/ocr_outputs/sample.json
```

Evaluate OCR JSON files against a manifest (only fields present in `ground_truth` are scored):

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
Docs/            study notes and plans
configs/         dataset manifest example
scripts/
  run_dashboard.sh
  fetch_tessdata.sh          tessdata_best + tesstrain + langdata (gitignored targets)
benchmark/
  datasets/ ocr_outputs/ reports/
  uploads/                   dashboard uploads (gitignored)
  training/ground-truth/     labeled line pairs per dataset (gitignored)
  generated/                 tessdata variants, OCR runs, EasyOCR models (gitignored)
training/tesstrain/          tesstrain checkout (gitignored)
nid_ocr_lab/
  cli.py models.py pipeline.py evaluation.py
  engines/     base.py tesseract.py tessdata.py easyocr.py paddleocr.py gemini_vision.py lmstudio_vision.py
  parsers/     nid_parser.py
  dashboard/   server.py uploads.py filters.py indexer.py static/
  training/    dataset.py tesstrain_runner.py
tests/         unittest suite
```

## Key Code Files

- `nid_ocr_lab/dashboard/server.py`: HTTP server; OCR, upload, run-crop, and training-line endpoints; bounded Tesseract rotation/PSM flow.
- `nid_ocr_lab/dashboard/uploads.py`: upload storage, EXIF handling, PDF page rendering.
- `nid_ocr_lab/dashboard/indexer.py`: discovers SmartScan samples and dataset health.
- `nid_ocr_lab/dashboard/filters.py`: image filters, SDK-style crop, mask overlay rendering; lossless OCR inputs.
- `nid_ocr_lab/dashboard/static/app.js`: dashboard state, uploads, OCR controls, line labeling.
- `nid_ocr_lab/engines/tesseract.py`: Tesseract adapter (explicit command, line reconstruction, OSD).
- `nid_ocr_lab/engines/tessdata.py`: model folder registry, sha256 and embedded version lookup.
- `nid_ocr_lab/parsers/nid_parser.py`: script-aware label parser, NID number, validated DOB.
- `nid_ocr_lab/evaluation.py`: field metrics over annotated fields, per-script accuracy, CER.
- `nid_ocr_lab/training/dataset.py`: OCR run artifacts, line crops, ground-truth writing, card-level split.
- `nid_ocr_lab/training/tesstrain_runner.py`: tesstrain wrapper and line-level model comparison.
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

1. Label verified lines from many distinct cards (Bengali names, English names, parents, DOB, NID) and record base-model CER for `system:ben`, `best:ben`, `best:script/Bengali`.
2. Install GNU make (`brew install make`) and run a first fine-tune only after the base numbers show recognition (not cropping/layout) errors.
3. Field-ROI OCR: detect the layout, crop fields, and read each with a script-specific profile (PSM 7 lines, digit whitelist for the NID number).
4. Batch runner over validation/test samples and uploads, writing reports to `benchmark/reports/`.
5. Real field-level ground-truth manifests for card accuracy.
6. Draw OCR line boxes over the image and allow adding a missed line box while labeling.

## EasyOCR quick lookup (2026-09-15)

EasyOCR 1.7.2 is installed in the SmartScan venv used by the launcher. To reproduce:

```bash
"/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/venv/bin/python" -m pip install -r requirements-easyocr.txt
```

Select **EasyOCR**, choose `eng+ben`, `eng`, or `ben`, and click **Run OCR**. The adapter maps these to `en`/`bn`. For an upright card, select rotation **0** for the quickest comparison. Auto detects text once and tests text-box rotations; it does not determine a single global card angle. Manual rotation rotates the card before inference.

The CPU reader is reused per language combination and serialized across requests. First use downloads/loads official models; subsequent requests reuse them. Models are stored in ignored `benchmark/generated/easyocr/`. No card image is uploaded by EasyOCR. Raw text, normalized boxes/confidences, raw engine response, and the existing NID parser appear in the usual panels. The parser is still a research stub; check raw OCR separately from parsed fields.

Dashboard timing now shows total request latency. EasyOCR JSON additionally records initialization, inference, queue time, and reader reuse. Restart the dashboard after installing or changing code. Tests: `python -m unittest discover -s tests` using the dashboard interpreter.

Verified locally: English, Bengali-only, and mixed Bengali-English inference; auto text-box rotation; reader reuse; dashboard output with both scripts; four adapter/integration tests. Official detector and recognition model checksums were verified during setup. These smoke checks do not establish field accuracy.
