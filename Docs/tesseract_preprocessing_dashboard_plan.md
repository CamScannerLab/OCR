# Tesseract Preprocessing Dashboard Plan

## Objective

Improve Bangladesh NID OCR by preparing a cleaner image before local Tesseract runs, while making every preprocessing step visible in the dashboard.

The goal is not to replace Tesseract or call a cloud OCR service. Tesseract remains local. The dashboard will simply expose local preprocessing and local Tesseract inspection in one research flow.

## Clarification: Local Dashboard Routes, Not External APIs

When this plan mentions dashboard endpoints such as `/api/tesseract/steps`, it means routes served by the local Python dashboard at:

```text
http://127.0.0.1:8765
```

These routes run local Python image processing and local Tesseract/libtesseract. They are not external APIs and do not send the NID image to a cloud service.

Current local Tesseract pieces:

```text
Dashboard browser
  -> local Python dashboard route
  -> local image preprocessing
  -> local /opt/homebrew/bin/tesseract or local libtesseract
  -> dashboard result
```

## Current NID Ink Behavior

The existing `NID ink` filter suppresses the yellow/orange emblem and background by keeping only the green channel, then applying autocontrast.

Current implementation:

```python
def nid_ink(image):
    return ImageOps.autocontrast(
        image.convert("RGB").getchannel("G"),
        cutoff=1,
    ).convert("RGB")
```

Why it helps:

- the NID emblem/background is yellow/orange
- in the green channel, that watermark becomes closer to the paper/background
- black text stays dark
- much red/dark printed text remains readable
- Tesseract's layout step sees fewer false background strokes behind rows like `নাম`, `পিতা`, and `মাতা`

Current limitation:

- `NID ink` does not remove the portrait photo
- it does not explicitly detect text regions
- it does not know field layout
- it can suppress the emblem but cannot reliably suppress all decorative/noisy regions

## Main Insight

Tesseract output is acceptable when the input is clean. The hard part is that NID cards include:

- colored background
- emblem/watermark behind text
- portrait photo
- mixed Bengali and English scripts
- red, black, and sometimes low-contrast ink
- layout elements that can confuse Tesseract page segmentation

Therefore the next R&D task is:

```text
NID image
  -> visible preprocessing pipeline
  -> cleaner OCR image
  -> local Tesseract
  -> visible Tesseract internal steps
  -> parsed NID fields
```

## Dashboard Requirement

Every preprocessing step must be visible in the dashboard.

The dashboard should show:

- original input
- each intermediate image
- masks used by the step
- measurements for that step
- the final image handed to Tesseract
- Tesseract's existing grey, threshold, layout, recognition, and field re-read steps

The user should be able to inspect why OCR improved or failed.

## Proposed Dashboard Flow

```text
Source image
  -> Preprocessing steps card
       01 input
       02 EXIF/rotation-normalized input
       03 optional SmartScan crop
       04 background estimate
       05 emblem/background suppression
       06 photo/noise region analysis
       07 text ink image
       08 contrast normalization
       09 binary candidate
       10 final OCR image
  -> Tesseract steps card
       input
       orientation
       grey
       threshold
       blobs
       blocks
       paragraphs
       lines
       words/symbols
       recognition
       page output
       NID fields re-read
```

## Preprocessing Recipes To Add

### 1. Off

No new preprocessing. This keeps the current baseline.

Use it to compare whether new preprocessing actually helps.

### 2. NID Ink V1

The current green-channel filter.

Purpose:

- keep a stable baseline
- prove dashboard wiring works
- compare old and new behavior side by side

### 3. NID Ink V2

Improve the current green-channel idea with measured diagnostics.

Possible steps:

- extract green channel
- estimate background tone
- autocontrast with controlled clipping
- preserve dark strokes
- show before/after ink share

Dashboard output:

- green channel image
- autocontrast image
- histogram
- ink mask
- final OCR image

### 4. Background Neutralize

Flatten colored paper/background while preserving printed text.

Possible steps:

- estimate low-frequency background with blur or morphology
- divide or subtract background illumination
- normalize color/luminance
- keep dark/red text strokes strong

Dashboard output:

- estimated background image
- normalized image
- difference/foreground image
- final OCR image

### 5. Emblem Suppression

Specifically reduce watermark/emblem texture behind text rows.

Possible approaches:

- use color-channel separation
- identify yellow/orange background regions
- whiten low-contrast emblem texture
- preserve high-contrast dark strokes

Dashboard output:

- emblem/background color mask
- suppressed image
- text-preservation comparison

### 6. Photo Region Suppression

Handle the portrait photo separately from emblem removal.

This should not blindly crop the photo away unless we know the layout and the ID number will not be harmed.

Possible approaches:

- detect large photo-like region by size, texture, and color variation
- create a photo mask
- neutralize the photo region for page layout
- optionally keep the original image for field-specific crops if needed

Dashboard output:

- photo candidate mask
- photo-neutralized image
- layout comparison with and without photo suppression

### 7. Clean Binary Candidate

Create an OCR-ready binary candidate only after checking that Bengali marks are not damaged.

Possible steps:

- adaptive threshold
- small speckle removal
- optional tiny gap repair
- avoid aggressive erosion

Dashboard output:

- binary image
- removed-noise mask
- connected component statistics
- final OCR image

## Implementation Plan

### Phase 1: Preprocessing Step Model

Add a small backend structure for preprocessing steps.

Suggested module:

```text
nid_ocr_lab/preprocessing/nid_pipeline.py
```

Each step should return:

```python
{
    "id": "emblem_suppression",
    "title": "Emblem suppression",
    "explain": "Suppress yellow/orange watermark texture while preserving dark text strokes.",
    "values": {
        "ink_share": 0.054,
        "background_level": 218
    },
    "images": {
        "output": "05-emblem-suppressed.png",
        "mask": "05-emblem-mask.png"
    }
}
```

### Phase 2: Local Dashboard Endpoint

Add a local route for preprocessing inspection.

Suggested route:

```text
POST /api/preprocess/steps
```

It should:

1. prepare the selected source image using the same source/input settings as OCR
2. run the selected preprocessing recipe
3. save all step images under `benchmark/generated/preprocess/<run_id>/`
4. return step metadata and image names to the dashboard

This is local dashboard plumbing only.

### Phase 3: Dashboard UI

Add a `Pre-processing steps` card.

Controls:

```text
Recipe: Off / NID ink v1 / NID ink v2 / Background neutralize / Emblem suppress / Photo suppress / Clean binary / Compare all
Run preprocessing
Use final image for Tesseract steps
```

The card should render:

- timeline of step images
- values table per step
- masks and outputs
- final OCR image preview

### Phase 4: Pipe Final Image Into Tesseract Steps

When `Use final image for Tesseract steps` is enabled:

```text
preprocessing final image
  -> existing Tesseract step inspector
```

This lets us compare:

```text
raw image -> Tesseract layout
preprocessed image -> Tesseract layout
```

### Phase 5: Compare All Mode

Add a research mode that runs several preprocessing recipes on the same image.

For each candidate, show:

- final image
- preprocessing time
- ink share
- blob count
- Tesseract field count
- mean confidence
- tall/merged line count

This avoids choosing by visual impression only.

## First Implementation Target

Start with:

```text
Pre-processing steps card
  + NID ink v1 visible timeline
  + NID ink v2 visible timeline
  + final image piped into existing Tesseract steps
```

This gives the research loop immediately:

```text
try preprocessing
  -> see each image step
  -> see Tesseract layout/recognition impact
  -> tune based on evidence
```

After that, add background neutralization, photo suppression, and compare-all metrics.

## Tests

Add focused tests for:

- preprocessing pipeline returns stable step JSON
- step images are written
- step image route rejects path traversal
- `NID ink v1` remains equivalent to the current filter
- final preprocessing output can be passed to Tesseract steps
- dashboard route returns recipe, steps, timings, and final image

## Success Criteria

The work is successful when the dashboard can show:

```text
Original image
  -> each preprocessing image/mask
  -> final OCR image
  -> Tesseract internal steps on that final image
  -> parsed NID fields
```

And when we can answer, for any sample:

1. Did preprocessing reduce emblem/background confusion?
2. Did it reduce photo/layout confusion?
3. Did it preserve Bengali and English strokes?
4. Did Tesseract detect better text lines?
5. Did parsed NID fields improve?

