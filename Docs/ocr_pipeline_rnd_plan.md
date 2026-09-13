# OCR Pipeline R&D Plan

## Objective

Create an experiment-first OCR pipeline for Bangladesh NID cards using SmartScan output images as input.

The R&D pipeline should answer:

1. Which OCR engine performs best on NID fields?
2. Which SmartScan output mode performs best per field?
3. Which fields need ROI fallback?
4. What error patterns should drive dictionary and character-confusion correction?
5. What confidence score is reliable enough to accept on-device results?

## Pipeline

```text
SmartScan image variant
  -> OCR engine
  -> OCR blocks with text, bbox, confidence
  -> template/layout detection
  -> field extraction
  -> normalization
  -> validation
  -> candidate ranking
  -> confidence decision
  -> structured NID result
```

## R&D Inputs

Each sample should have:

- sample id
- NID layout/version: old or smart
- ground-truth fields
- SmartScan output image variants
- capture condition tags: blur, glare, rotation, lighting, perspective

## Engines To Test First

- Tesseract 5: `eng`, `ben`, `eng+ben`; compare `fast`, standard, and `best` data.
- PaddleOCR: detector + recognizer output, especially bbox quality.
- RapidOCR/ONNX: test when mobile/server portability becomes important.

Later baselines:

- docTR
- Surya
- EasyOCR
- Apple Vision for iOS-only comparison

## Evaluation Priority

Rank pipelines primarily by field-level correctness:

1. NID number exact match
2. date of birth exact match
3. English name exact match
4. Bangla name exact/normalized match
5. father/mother fields
6. address quality
7. false correction rate
8. latency
9. memory/model size
10. deployment complexity

## Recommended First Experiment

Run this matrix on 30-50 samples:

```text
engine:       tesseract, paddleocr
language:     eng, ben, eng+ben where applicable
input mode:   corrected, enhance, deglare, matte
metric:       field exact accuracy + latency
```

Do not optimize server throughput yet. First prove the field extraction path.

