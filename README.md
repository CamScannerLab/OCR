# Bangladesh NID OCR R&D Lab

This workspace is for researching the OCR pipeline after SmartScan has produced a corrected or enhanced document image.

The goal is not to choose an OCR engine by reputation. The goal is to measure which OCR pipeline gives the best Bangladesh NID field-level extraction accuracy under realistic capture conditions.

## Upstream SmartScan References

SmartScan is already implemented as a separate R&D effort. Keep these projects as the upstream context for scanner behavior, image normalization, enhancement modes, and scan-quality debugging:

- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK`
- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan`

This OCR lab should consume SmartScan outputs rather than redoing the scanner layer.

## Current Boundary

SmartScan is treated as the upstream image acquisition layer:

```text
camera image
  -> card detection
  -> crop / perspective correction
  -> enhancement variants
  -> OCR R&D pipeline
```

This lab starts after that:

```text
SmartScan output image
  -> OCR engine adapter
  -> OCR blocks
  -> NID layout parser
  -> field normalization / validation
  -> candidate ranking
  -> structured NID data
  -> evaluation
```

## First R&D Milestone

1. Collect 30-50 redacted or synthetic NID samples.
2. For each sample, save SmartScan variants: original, corrected, enhance, deglare, matte, and super if available.
3. Run at least Tesseract and PaddleOCR against the same inputs.
4. Save OCR outputs as JSON in `benchmark/ocr_outputs/`.
5. Evaluate field-level accuracy with:

```bash
python3 -m nid_ocr_lab.cli evaluate benchmark/ocr_outputs configs/dataset_manifest.example.json
```

## OCR Baseline

The first OCR adapter is Tesseract. Once Tesseract and the needed language data are installed, run:

```bash
python3 -m nid_ocr_lab.cli ocr-tesseract path/to/cropped-or-smartscan-output.jpg --lang eng+ben --output benchmark/ocr_outputs/sample_001_tesseract_eng-ben_enhance.json
```

Then parse or evaluate the OCR JSON:

```bash
python3 -m nid_ocr_lab.cli parse-json benchmark/ocr_outputs/sample_001_tesseract_eng-ben_enhance.json
```

## SmartScan Sample Dashboard

Run a local dashboard to inspect SmartScan samples, masks, annotations, and OCR-friendly filter variants:

```bash
bash scripts/run_dashboard.sh
```

Then open:

```text
http://127.0.0.1:8765
```

The dashboard reads from:

- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/dataset`
- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/baseline_masks`
- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK/demo/src/main/assets`

It provides Python approximations of the SDK filter modes: original, enhance, deglare, matte, and super, plus edge/gray/threshold views for OCR experiments.

## Layout

```text
Docs/
  current_study.md
  ocr_pipeline_rnd_plan.md

configs/
  dataset_manifest.example.json

benchmark/
  datasets/
    synthetic/
    redacted/
  ocr_outputs/
  reports/

nid_ocr_lab/
  models.py
  pipeline.py
  evaluation.py
  cli.py
  dashboard/
  engines/
  parsers/
```
