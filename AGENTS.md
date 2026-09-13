# Agent Context

This workspace is the Bangladesh NID OCR R&D lab. It starts after SmartScan has already produced a card image variant.

## Upstream SmartScan R&D

The SmartScan work that feeds this OCR phase lives in:

- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan-SDK`
- `/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan`

Use these as upstream references when OCR work needs scanner context, image variants, perspective correction behavior, enhancement modes, segmentation/corner-detection behavior, or SmartScan output contracts.

## Boundary

Do not re-implement the SmartScan scanner layer inside this OCR workspace unless explicitly asked.

Treat SmartScan as responsible for:

- camera capture
- document/card detection
- crop and perspective correction
- image enhancement variants
- debug outputs related to scan quality

Treat this OCR workspace as responsible for:

- OCR engine benchmarking
- OCR adapter contracts
- text block and bounding-box normalization
- NID layout parsing
- field extraction
- dictionary/correction logic
- validation and candidate ranking
- field-level evaluation

## Preferred R&D Flow

Use SmartScan outputs as benchmark inputs:

```text
SmartScan raw/corrected/enhance/deglare/matte output
  -> OCR engine adapter
  -> OCR blocks with text, bbox, confidence
  -> NID field parser
  -> correction and validation
  -> structured NID data
  -> benchmark report
```

Keep OCR experiments engine-agnostic. Compare Tesseract, PaddleOCR, RapidOCR/ONNX, and other candidates through the same output schema and field-level metrics.

