# Bangladesh NID OCR — Complete Technical Context & Research Brief

## 1. Project Goal

I am building an OCR/document-scanning system specifically for **Bangladesh National ID (NID) cards**.

The eventual system should support:

* Android native app
* iOS native app
* Offline/on-device OCR where practical
* Potential server-side OCR for high-accuracy or difficult cases
* Bengali + English + Roman/Latin text
* Numbers, dates, addresses, names, geographic names, labels, etc.
* The two major Bangladesh NID layouts/versions
* Perspective distortion
* Camera rotation
* Blur
* Glare
* Different lighting conditions
* Different phone cameras
* Potentially very high server throughput, possibly **500 OCR requests/second**

The system should ultimately extract structured NID information rather than simply return raw OCR text.

Example structured output:

```swift
struct NIDData {
    let nidNumber: String?
    let nameBangla: String?
    let nameEnglish: String?
    let fatherNameBangla: String?
    let fatherNameEnglish: String?
    let motherNameBangla: String?
    let motherNameEnglish: String?
    let dateOfBirth: Date?
    let addressBangla: String?
    let addressEnglish: String?
    let rawText: String
}
```

---

# 2. Important Constraint: This Is NOT Generic OCR

I am not trying to build a generic "OCR any document" application.

The target is a **known document family with approximately two major NID layouts**.

That is an important advantage.

The system can exploit:

* known card dimensions/aspect ratio
* known field positions
* known labels
* known relationships between labels and values
* known NID number format
* known date format
* Bangladesh geographic vocabulary
* Bengali/English language expectations
* known document structure

Therefore I want to build a **document-specific OCR pipeline**, not simply call an OCR engine and trust its output.

---

# 3. Core OCR Question

The main engines I am considering are:

1. Tesseract 5
2. PaddleOCR / PP-OCR
3. RapidOCR
4. docTR
5. Surya OCR
6. EasyOCR
7. Apple Vision
8. ONNX Runtime with suitable OCR models
9. OpenCV for preprocessing

The initial debate was especially:

> Tesseract vs PaddleOCR

I initially thought Tesseract might be better because of its history and Google's involvement.

Important clarification:

* Tesseract was originally developed at HP.
* It was open-sourced in 2005.
* Google sponsored/developed it from approximately 2006–2017.
* Tesseract is now an open-source community-maintained project.
* Tesseract 4 introduced the LSTM neural OCR engine.
* Tesseract 5 is the current major generation.

Google's historical involvement does NOT automatically mean Tesseract is better than modern OCR systems.

The correct way to choose is to benchmark the engines on actual/synthetic/redacted Bangladesh NID samples.

---

# 4. Current Thinking About Tesseract

Tesseract is still extremely interesting for this project.

Advantages:

* mature
* open source
* offline
* relatively lightweight
* CPU-friendly
* Android possible
* iOS possible
* supports Bengali
* supports English
* supports many other languages/scripts
* easy to deploy server-side
* potentially very attractive for high-throughput CPU scaling
* easy to customize around known NID fields

Important language data to test:

```text
eng
ben
eng+ben
```

Also compare:

```text
tessdata_fast
tessdata
tessdata_best
```

`fast` should be investigated for high-throughput use; `best` for maximum recognition quality.

For the NID project, do NOT simply do:

```text
image -> Tesseract -> final result
```

Instead:

```text
NID image
    ↓
perspective correction
    ↓
normalization
    ↓
field detection / text detection
    ↓
field-specific preprocessing
    ↓
Tesseract
    ↓
post-processing
    ↓
dictionary
    ↓
validation
    ↓
candidate ranking
    ↓
confidence
    ↓
final structured NID result
```

A well-engineered Tesseract-based NID pipeline could potentially outperform a generic neural OCR system for this constrained document.

---

# 5. Current Thinking About PaddleOCR

PaddleOCR is probably the strongest modern alternative to Tesseract.

Its biggest advantage for this project is that it provides a more modern:

```text
Text Detection
        +
Text Recognition
```

pipeline.

This is very useful because I need OCR bounding boxes.

For example:

```text
"নাম / Name"
bbox + confidence

"DONIEL TRIPURA"
bbox + confidence

"পিতা / Father"
bbox + confidence

"JOHN TRIPURA"
bbox + confidence
```

These bounding boxes make field extraction much easier.

PaddleOCR should therefore be tested as both:

* text detector
* text recognizer

rather than blindly using its entire document pipeline.

However, I should NOT assume that a generic PaddleOCR model will automatically give excellent Bengali NID accuracy.

The actual Bengali fonts, card background, security graphics, blur, perspective, and camera conditions must be benchmarked.

---

# 6. Other Engines Worth Testing

## RapidOCR

Very interesting for this project.

It is based around lightweight OCR deployment and can work with Paddle-derived models and runtimes such as ONNX Runtime.

Potential advantage:

```text
Paddle model
      ↓
ONNX
      ↓
ONNX Runtime
      ↓
Android/iOS/server
```

This may provide a better deployment story than embedding the entire Paddle ecosystem.

Especially worth investigating for:

* mobile
* CPU inference
* offline use
* high throughput

---

## docTR

Modern deep-learning OCR pipeline.

Useful for benchmarking:

* text detection
* text recognition

I would treat it mainly as a research/benchmark candidate initially.

---

## Surya OCR

Modern document OCR/layout-oriented system.

Worth testing against real Bengali NID images.

Do NOT assume it is suitable for mobile or production just because it performs well on generic document benchmarks.

---

## EasyOCR

Worth running as a baseline.

It is easy to experiment with multilingual OCR.

But I would not spend significant engineering time on it unless it performs surprisingly well on the actual NID dataset.

---

## Apple Vision

Useful as an independent iOS benchmark.

Potentially useful for Latin/English portions.

However, Bengali support/quality should be explicitly benchmarked rather than assumed.

I would not make Vision the core cross-platform OCR engine.

---

## OpenCV

OpenCV is NOT an OCR engine, but it may be one of the most important components.

Preprocessing may affect final accuracy more than changing OCR engines.

Test:

* grayscale
* contrast normalization
* CLAHE
* illumination correction
* denoising
* sharpening
* adaptive threshold
* Otsu threshold
* morphology
* upscale
* perspective correction

For example:

```text
Original
   ↓
Grayscale
   ↓
CLAHE
   ↓
Upscale 2x
   ↓
Sharpen
   ↓
Tesseract
```

Different fields may require different preprocessing.

---

# 7. ONNX Runtime

ONNX Runtime is not itself an OCR engine.

It is potentially very useful as the **deployment/runtime layer** for neural OCR models.

Possible architecture:

```text
OCR model
   ↓
ONNX
   ↓
ONNX Runtime
   ├── Android
   └── iOS
```

This could provide a clean cross-platform native deployment strategy.

If PaddleOCR produces the best recognition model, investigate whether the required model can be exported/deployed appropriately through ONNX or another mobile runtime.

---

# 8. Mobile vs Server

Both Tesseract and PaddleOCR can be used in two broad ways.

## Option A — On-device

```text
Android/iPhone
      │
    Camera
      │
NID detection
      │
Perspective correction
      │
OCR
      │
NID parser
      │
Structured result
```

Advantages:

* offline
* privacy
* no server cost
* no network dependency
* fast local response
* good for an SDK

Disadvantages:

* limited CPU/GPU
* memory constraints
* app size/model size
* different device performance
* model updates more complicated

---

## Option B — Server

```text
Mobile
   │
   │ image
   ▼
OCR Server
   │
   ├── PaddleOCR
   ├── Tesseract
   ├── preprocessing
   └── NID post-processing
   │
   ▼
JSON
   │
   ▼
Mobile
```

Advantages:

* powerful CPUs/GPUs
* larger OCR models
* easier model updates
* multiple OCR engines
* centralized dictionaries
* easier high-throughput scaling

Disadvantages:

* server cost
* network dependency
* privacy/security concerns
* latency
* NID images are sensitive identity documents

---

# 9. Hybrid Architecture

This is potentially the best product architecture.

Example:

```text
                  Mobile
                    │
                  Camera
                    │
                    ▼
              NID Detection
                    │
                    ▼
           Perspective Correction
                    │
                    ▼
                Tesseract
                    │
              confidence?
               /         \
             HIGH         LOW
              │             │
              ▼             ▼
           ACCEPT        OCR Server
                            │
                         PaddleOCR
                            │
                       post-processing
                            │
                            ▼
                         result
```

Most scans can be processed locally.

Only uncertain scans go to the server.

This reduces server load while retaining a high-accuracy fallback.

---

# 10. Important Privacy Consideration

Bangladesh NID is an identity document.

If server-side OCR is used, carefully consider:

* encryption
* data retention
* access control
* logging
* deletion
* whether the original image needs to leave the device
* whether only cropped fields can be sent
* whether clients require fully offline processing

For development/testing, prefer:

* synthetic NID images
* redacted NID images
* appropriately anonymized datasets

Do not build a benchmark around unnecessarily exposed real identity documents.

---

# 11. Whole-Card OCR vs ROI OCR

An important design decision:

**Whole-card OCR should be the primary pass. ROI OCR should be the fallback/refinement mechanism.**

Do NOT make fixed ROI coordinates the foundation.

Problem with naïve ROI:

```text
Camera image
    ↓
fixed ROI
    ↓
OCR
```

Perspective distortion can cause the actual field to move outside the ROI.

Instead:

```text
Camera image
    ↓
NID corner detection
    ↓
Perspective correction
    ↓
Canonical NID coordinate system
    ↓
Whole-card OCR
    ↓
text blocks + bounding boxes
    ↓
field extraction
    ↓
uncertain field?
    ↓
ROI refinement
```

The key is:

> Camera coordinates → NID coordinates → approximate ROI → local OCR

not:

> Camera coordinates → fixed crop → OCR

---

# 12. Canonical NID Coordinate System

After detecting the four corners, rectify the card into a fixed canvas.

For example:

```text
2000 × 1260
```

The exact canonical dimensions should be benchmarked.

Define ROIs in normalized coordinates:

```swift
struct NormalizedROI {
    let x: CGFloat
    let y: CGFloat
    let width: CGFloat
    let height: CGFloat
}
```

This makes the system robust to different input resolutions.

Use generous ROI margins rather than extremely tight boxes.

For difficult fields, try multiple ROI hypotheses:

```text
normal ROI
large ROI
shifted left
shifted right
slightly higher
slightly lower
```

But ideally use whole-card OCR bounding boxes to dynamically refine the ROI.

---

# 13. Proposed OCR Pipeline

```text
Camera/photo
     ↓
NID Detection
     ↓
Quality Check
     ↓
Corner Detection
     ↓
Perspective Correction
     ↓
Canonical NID Image
     ↓
Whole-card OCR
     ↓
OCR blocks
(text + bbox + confidence)
     ↓
Field Identification
     ↓
OCR Normalization
     ↓
Dictionary / Correction
     ↓
Field Validators
     ↓
Candidate Generation
     ↓
Candidate Ranking
     ↓
Confidence Engine
     │
     ├── high confidence → accept
     │
     └── low confidence → ROI fallback
                              ↓
                         field preprocessing
                              ↓
                         OCR again
                              ↓
                         re-rank candidates
                              ↓
                         final result
```

---

# 14. OCR Engine Abstraction

Do not couple the application directly to one OCR engine.

Use:

```swift
protocol OCREngine {
    func recognize(
        image: CGImage,
        languages: [OCRLanguage]
    ) async throws -> OCRResult
}
```

Possible implementations:

```text
OCREngine
│
├── TesseractEngine
├── PaddleOCREngine
├── ONNXOCREngine
└── VisionOCREngine
```

Then the NID parser never needs to know which engine produced the OCR.

---

# 15. Suggested Data Structures

OCR result:

```swift
struct OCRResult {
    let blocks: [OCRTextBlock]
    let fullText: String
}
```

OCR block:

```swift
struct OCRTextBlock {
    let text: String
    let boundingBox: CGRect
    let confidence: Float
}
```

NID version:

```swift
enum NIDVersion {
    case old
    case smart
}
```

NID schema:

```swift
struct NIDSchema {
    let version: NIDVersion
    let fields: [NIDField]
}
```

Field result:

```swift
struct OCRFieldResult {
    let rawValue: String
    let correctedValue: String?
    let confidence: Float
    let source: OCRSource
    let candidates: [Candidate]
    let needsReview: Bool
}
```

---

# 16. Dictionary / OCR Correction System

I specifically want a post-OCR correction system.

It should NOT replace the OCR engine.

Architecture:

```text
Raw OCR
   ↓
Character normalization
   ↓
Character confusion correction
   ↓
Dictionary lookup
   ↓
Field-specific dictionary
   ↓
Geographic dictionary
   ↓
Format validation
   ↓
Candidate generation
   ↓
Candidate ranking
```

Potential dictionaries:

### Bengali

* common Bengali words
* NID labels
* common names if available appropriately
* address vocabulary

### English

* NID labels
* common names
* common address words
* common abbreviations

### Bangladesh geography

* divisions
* districts
* upazilas
* unions
* municipalities
* cities
* known geographic names

### NID terminology

Examples:

```text
Name
Father
Mother
Date of Birth
Address
National ID
NID
```

---

# 17. Character Confusion

Do NOT globally perform substitutions such as:

```text
O → 0
I → 1
S → 5
B → 8
```

These should be field/context-dependent.

For example:

```text
NID number:
O ↔ 0
I ↔ 1
S ↔ 5
```

may be useful.

But:

```text
Person name:
O → 0
```

could be harmful.

For Bengali character confusion, do not guess the confusion table manually.

Instead, build it empirically:

```text
OCR output
     ↓
ground truth
     ↓
observed errors
     ↓
confusion matrix
     ↓
learned correction probabilities
```

Store:

```text
raw OCR
ground truth
correction
confidence
engine
preprocessing
```

Then use real errors to improve the correction engine.

---

# 18. Candidate Ranking

An initial candidate score could be:

```text
score =
    0.40 * OCR confidence
  + 0.20 * dictionary similarity
  + 0.15 * character similarity
  + 0.15 * field validity
  + 0.10 * layout/context score
```

These weights are only an initial heuristic.

They should eventually be tuned/learned using real labeled test data.

---

# 19. Be Conservative With Names

This is critical.

If OCR returns:

```text
DONIEL TRIPURA
```

and the dictionary does not contain the name, that does NOT mean it is incorrect.

Names can be unique.

Therefore:

### Strong correction

Use stronger correction for:

* NID numbers
* dates
* divisions
* districts
* upazilas
* known labels
* known document terminology

### Conservative correction

Use weaker correction for:

* person's name
* father's name
* mother's name
* village
* house name

Never silently replace identity information merely because it is not in a dictionary.

---

# 20. Field-Specific OCR

Different fields should have different preprocessing and validation.

## NID number

```text
grayscale
→ upscale
→ sharpen
→ threshold
→ OCR
→ numeric normalization
→ format validation
```

## Bengali name

```text
contrast normalization
→ upscale
→ Bengali OCR
→ Bengali normalization
→ candidate ranking
```

## English name

```text
upscale
→ English OCR
→ name validation
```

## Date

Use OCR + regex/date validation.

## Address

Use OCR + geographic/contextual dictionary.

---

# 21. Multiple OCR Engines Should NOT Always Run

Do not do this on every mobile scan:

```text
             NID
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
 Tesseract Paddle Vision
       │      │      │
       └──────┼──────┘
              ▼
           Ranking
```

That wastes CPU, memory and battery.

Instead:

```text
NID
 ↓
primary OCR
 ↓
confidence?
 ├── high → accept
 └── low → secondary OCR
```

For Android:

```text
Tesseract
    ↓
low confidence
    ↓
neural OCR / Paddle-derived model
```

For iOS:

```text
Tesseract
    ↓
low confidence
    ↓
Vision or neural OCR
```

---

# 22. 500 TPS Requirement

Potential server target:

**500 NID images/second**

That equals:

* 30,000 images/minute
* 1.8 million images/hour
* 43.2 million images/day

This is a very high throughput requirement.

Do NOT design around one server.

Use horizontal scaling:

```text
                    Load Balancer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       OCR Worker      OCR Worker     OCR Worker
          │              │              │
       Paddle          Paddle          Paddle
          │              │              │
        GPU/CPU        GPU/CPU        GPU/CPU
```

---

# 23. Rough Hardware Planning

These are only initial planning estimates and must be benchmarked with actual NID images.

## Tesseract

A practical starting server:

```text
8 vCPU
16 GB RAM
```

Potentially around 10–30+ such servers could be needed for 500 TPS depending heavily on:

* image resolution
* preprocessing
* language models
* OCR configuration
* concurrency
* latency target

Tesseract is CPU-oriented, so horizontal CPU scaling is natural.

---

## PaddleOCR

A practical starting server:

```text
8–16 CPU cores
16–32 GB RAM
1 modern NVIDIA GPU
```

A rough initial planning estimate could be:

```text
2–6 GPU servers
```

for 500 TPS, but this is NOT a guarantee.

Actual throughput must be benchmarked.

Factors include:

* detector model
* recognizer model
* image resolution
* batch size
* preprocessing
* GPU
* concurrency
* latency SLA

---

# 24. Important: 500 TPS Does NOT Mean Mobile OCR Needs 500 TPS

If 500 TPS means server requests, that's a backend capacity requirement.

The mobile app itself only processes one/few scans at a time.

Therefore mobile architecture and server architecture should be optimized separately.

---

# 25. Recommended Production Architecture

For a serious NID SDK/service, I would consider:

```text
                     CLIENT
                       │
             ┌─────────┴─────────┐
             │                   │
         Android               iOS
             │                   │
         CameraX          AVCaptureSession
             │                   │
             └─────────┬─────────┘
                       │
                NID Detection
                       │
                Perspective
                 Correction
                       │
                 On-device OCR
                    Tesseract
                       │
                  Confidence
                  /         \
               HIGH          LOW
                │             │
                ▼             ▼
             Result       OCR Server
                              │
                         Paddle/ONNX
                              │
                         Tesseract
                              │
                       NID correction
                              │
                         final result
```

This gives:

* offline-first capability
* privacy
* reduced server load
* high accuracy fallback
* scalable server architecture

---

# 26. Recommended Mobile Architecture

```text
NIDScannerSDK
│
├── Camera
├── NIDDetector
├── NIDQualityChecker
├── PerspectiveCorrector
├── NIDTemplateDetector
│
├── OCR
│   ├── TesseractEngine
│   ├── NeuralOCREngine
│   └── VisionEngine (iOS)
│
├── OCRTextDetector
├── FieldExtractor
├── OCRNormalizer
├── DictionaryManager
├── CandidateGenerator
├── CandidateRanker
├── FieldValidators
├── ConfidenceEngine
├── ROIFallbackManager
│
└── NIDResultBuilder
```

---

# 27. Android

Potential stack:

```text
CameraX
+
Tesseract
+
OpenCV/custom image processing
+
NID parser
```

If neural OCR is needed:

```text
ONNX Runtime Mobile
```

or an appropriate mobile deployment of a Paddle-derived model.

---

# 28. iOS

Potential stack:

```text
AVCaptureSession
+
Tesseract
+
Core Image / OpenCV
+
NID parser
```

Optional:

```text
Apple Vision
```

and/or:

```text
ONNX Runtime
```

for a neural OCR model.

---

# 29. Don't Assume Full PaddleOCR Is the Right Mobile Package

This is an important architectural point.

PaddleOCR is excellent for research/server use.

But embedding an entire PaddleOCR stack into native Android/iOS can introduce:

* native dependency complexity
* model size
* memory usage
* ABI packaging
* ARM64 concerns
* GPU/CPU acceleration issues
* framework integration work
* model updates
* app binary size

Therefore, investigate:

```text
PaddleOCR
   ↓
identify best model
   ↓
export/deploy suitable model
   ↓
ONNX/mobile runtime
```

rather than automatically embedding everything.

---

# 30. Proposed Benchmark

Build an actual OCR benchmark instead of choosing based on reputation.

Directory idea:

```text
ocr-benchmark/
│
├── datasets/
│   ├── synthetic/
│   ├── redacted/
│   └── generated/
│
├── engines/
│   ├── tesseract/
│   ├── paddle/
│   ├── rapidocr/
│   ├── doctr/
│   ├── surya/
│   └── vision/
│
├── preprocessing/
│   ├── original/
│   ├── grayscale/
│   ├── clahe/
│   ├── sharpen/
│   ├── threshold/
│   └── upscale/
│
├── evaluation/
│   ├── character_accuracy/
│   ├── word_accuracy/
│   ├── field_accuracy/
│   ├── exact_match/
│   └── latency/
│
└── reports/
```

Every engine must receive the same images.

---

# 31. Test Matrix

Test at least:

### NID version

* old
* smart/new

### Rotation

* 0°
* 3°
* 5°
* 10°
* 15°

### Perspective

* flat
* mild
* moderate

### Lighting

* normal
* dark
* bright
* glare

### Sharpness

* sharp
* medium
* blurred

### Language composition

* Bengali-heavy
* English-heavy
* mixed
* Romanized names
* numeric-heavy

### Hardware

* low-end Android
* mid-range Android
* high-end Android
* older iPhone
* newer iPhone

---

# 32. Metrics

Do NOT only measure generic OCR character accuracy.

Measure:

```text
NID number exact accuracy
DOB exact accuracy
Name exact accuracy
Father name exact accuracy
Mother name exact accuracy
Address accuracy
Bengali character accuracy
English character accuracy
Field detection accuracy
False correction rate
Latency
Memory
CPU
Model size
Battery impact
```

The most important metric is:

> **Field-level exact accuracy on actual target NID fields.**

---

# 33. Candidate Comparison

Initial candidates:

| Engine       | Android | iOS | Offline | Bengali          | Detection        | Mobile | Server | Priority |
| ------------ | ------- | --- | ------- | ---------------- | ---------------- | ------ | ------ | -------- |
| Tesseract 5  | ✅       | ✅   | ✅       | ✅                | Basic            | ⭐⭐⭐⭐   | ✅      | ⭐⭐⭐⭐⭐    |
| PaddleOCR    | ✅       | ⚠️  | ✅       | model-dependent  | ⭐⭐⭐⭐⭐            | ⭐⭐⭐    | ✅      | ⭐⭐⭐⭐⭐    |
| RapidOCR     | ✅       | ⚠️  | ✅       | model-dependent  | ⭐⭐⭐⭐             | ⭐⭐⭐⭐   | ✅      | ⭐⭐⭐⭐     |
| docTR        | ⚠️      | ⚠️  | ✅       | model-dependent  | ⭐⭐⭐⭐             | ⭐⭐     | ✅      | ⭐⭐⭐      |
| Surya        | ⚠️      | ⚠️  | ✅       | benchmark needed | ⭐⭐⭐⭐             | ⭐      | ✅      | ⭐⭐⭐      |
| EasyOCR      | ⚠️      | ⚠️  | ✅       | model-dependent  | ⭐⭐⭐              | ⭐⭐     | ✅      | ⭐⭐       |
| Vision       | ❌       | ✅   | ✅       | benchmark needed | ⭐⭐⭐⭐             | ⭐⭐⭐⭐⭐  | ❌      | ⭐⭐⭐      |
| ONNX Runtime | ✅       | ✅   | ✅       | model-dependent  | depends on model | ⭐⭐⭐⭐⭐  | ✅      | ⭐⭐⭐⭐⭐    |

---

# 34. Initial Recommendation

Do NOT decide immediately that PaddleOCR is better.

Do NOT decide immediately that Tesseract is better because Google worked on it.

Instead:

## Phase 1

Build:

```text
Tesseract 5
+
OpenCV
+
NID perspective correction
+
NID field extraction
```

Test:

```text
eng
ben
eng+ben
```

and:

```text
fast
standard
best
```

---

## Phase 2

Build:

```text
PaddleOCR
+
same NID images
+
same preprocessing
```

Compare.

---

## Phase 3

Test:

```text
RapidOCR / ONNX
```

especially for mobile deployment and CPU throughput.

---

## Phase 4

Benchmark:

```text
docTR
Surya
EasyOCR
Vision
```

Only keep them if they demonstrate a meaningful advantage.

---

# 35. Most Likely End-State

There is no requirement that one OCR engine do everything.

A strong final architecture could be:

```text
                    NID
                     │
                     ▼
              NID Detector
                     │
                     ▼
           Perspective Correction
                     │
                     ▼
             Whole-card detection
                     │
                     ▼
              Tesseract / OCR
                     │
                confidence
                /        \
             high         low
              │             │
              │        neural OCR
              │             │
              └──────┬──────┘
                     ▼
              Candidate Results
                     │
                     ▼
              NID Field Parser
                     │
                     ▼
           Dictionary / Correction
                     │
                     ▼
                 Validators
                     │
                     ▼
               Candidate Ranking
                     │
                     ▼
                 Confidence
                     │
                     ▼
                 NIDData
```

The **NID-specific detection, preprocessing, field extraction, dictionary, validation, and confidence system** may ultimately contribute more to the final accuracy than the difference between Tesseract and PaddleOCR.

---

# 36. My Current Preference

If I had to start development today:

### Mobile

**Tesseract 5 + OpenCV + NID-specific pipeline**

because it is:

* cross-platform
* offline
* relatively lightweight
* mature
* Bengali-capable
* CPU-friendly
* easier to integrate

### Server

**PaddleOCR / Paddle-derived model + GPU**

for difficult/high-accuracy cases.

### Future mobile neural OCR

**ONNX Runtime + the best-performing model from the benchmark**

if Paddle/neural OCR proves substantially better.

### Correction

Custom:

**Dictionary + character confusion model + validators + candidate ranking**

### Architecture

**OCR engine abstraction**, so the implementation is not locked to Tesseract.

---

# 37. The Most Important Experiment

Before spending significant time on mobile integration, create a benchmark with approximately:

**100–300 synthetic/redacted NID samples**

covering:

* both layouts
* Bengali
* English
* mixed text
* different fonts
* rotation
* perspective
* blur
* glare
* low light
* different resolutions
* different preprocessing

Run:

```text
Tesseract
PaddleOCR
RapidOCR
docTR
Surya
Vision
```

where technically possible.

Then compare:

```text
Accuracy
vs
Latency
vs
Memory
vs
Model size
vs
Deployment complexity
vs
500-TPS server cost
```

Do not choose the winner based on generic OCR benchmarks.

Choose the engine/pipeline that gives the best **Bangladesh NID field-level accuracy + deployment characteristics**.

---

# 38. Final Decision Framework

The decision should ultimately be:

```text
                Does Tesseract meet
                required NID accuracy?
                       │
                  ┌────┴────┐
                 YES        NO
                  │          │
                  ▼          ▼
             Use it     Benchmark Paddle
             on-device       │
                             ▼
                       Is Paddle better?
                         │        │
                        YES       NO
                         │         │
                         ▼         ▼
                    Optimize    Investigate
                    Paddle/ONNX   hybrid
```

If Tesseract reaches the required accuracy after NID-specific preprocessing and correction, **there is little reason to use a much more complicated OCR stack just because PaddleOCR is newer**.

The goal is not to have the fanciest OCR engine.

The goal is:

> **Reliable, fast, offline-capable Bangladesh NID field extraction on Android + iOS, with a scalable server fallback when needed.**

---

# 39. Current Android Scanner Context: NexusPay SmartScan

There is now an important related Android project that should be treated as part of this OCR effort:

```text
/Users/admin/Desktop/KSL_Projects/NexusPay-Android/kona-pay-android-wallet/sdk/common/smartscan
```

Current branch:

```text
feature/nexuspay-app/smart-scan-implementation-v2
```

This module is not an OCR engine yet. It is a reusable Android document-scanning SDK that can provide the capture, document detection, crop, perspective correction, and enhancement stage before OCR.

Important implication:

```text
SmartScan solves or partially solves:
camera capture
live document boundary detection
NID aspect-ratio aware validation
auto/manual capture
crop review
perspective correction
image enhancement
glare/lighting mitigation
debug inspection of model masks

SmartScan does NOT yet solve:
text detection
OCR recognition
Bengali/English field extraction
NID layout parsing
dictionary correction
structured NIDData output
server-side OCR benchmarking
```

## SmartScan Module Stack

The module is an Android library:

```text
com.android.library
Kotlin + Java 8 target
View Binding
CameraX 1.4.2
LiteRT / TensorFlow Lite 1.4.1
OpenCV 4.12.0
AppCompat / Fragment KTX / ConstraintLayout
```

It bundles a segmentation model:

```text
src/main/assets/fairscan-segmentation-model.tflite
```

The public API is centered around:

```text
SmartScanActivity
SmartScanConfig
SmartScanResult
SmartScanHost
ColorMode
DocumentType
Step
```

`SmartScanConfig` already defaults to an NID-oriented flow:

```text
DocumentType.NID aspect ratio: 85.6 / 54.0
default steps: CAMERA -> CROP -> ENHANCE
default quad rules: QuadRules.idCard()
default outputMaxPixels: 2,000,000
default jpegQuality: 92
default minQuadScore: 0.60
default useFairScanPipeline: true
default landscapeInference: true
```

Document types currently include:

```text
NID
PASSPORT
FREE_FORM
```

## Detection Pipeline

The core detection path is:

```text
CameraX ImageAnalysis
    ↓
FrameConverter to upright bitmap
    ↓
SegmentationDetector
    ↓
fairscan-segmentation-model.tflite mask
    ↓
FairScanPipeline / MaskToQuad
    ↓
QuadValidator + QuadRules
    ↓
QuadStabilizer
    ↓
overlay / auto-capture / manual capture
```

`CameraSession` uses:

```text
ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
single-thread analysis executor
AtomicBoolean analysisBusy frame skipping
frame-difference skipping when the quad is already stable
QuadStabilizer promotion before auto-capture
ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
```

The segmentation detector supports:

```text
LiteRT/TFLite inference
API fallback below Android O
optional landscape inference
optional FairScan/OpenCV post-processing
optional edge snapping on capture
debug mask retention for diagnostics
```

The FairScan path wraps the model probability mask as an OpenCV `Mat` and calls:

```text
org.fairscan.imageprocessing.detectDocumentQuad(...)
```

If FairScan/OpenCV is unavailable, detection can fall back to the local `MaskToQuad` path.

## Crop, Perspective, And Enhancement

Perspective correction is already implemented in:

```text
PerspectiveCorrector
```

It uses Android `Matrix.setPolyToPoly()` to map the detected quad to a rectified rectangle, and can honor the target document aspect ratio. Output edge size is capped to avoid huge bitmaps.

The enhancement stage is already implemented in:

```text
Enhancer
ColorMode
```

Supported color modes include:

```text
ORIGINAL
ENHANCE
DEGLARE
MATTE
SUPER
```

The enhancement logic includes:

```text
percentile color stretching
luma sharpening
saturation boost
illumination flattening
specular glare suppression
```

This means future OCR benchmarking should use both:

```text
raw captured image
SmartScan corrected/enhanced output
```

and compare which input gives better field-level OCR accuracy.

## UI And Debugging

The module has a full native XML/ViewBinding flow:

```text
SmartScanCameraFragment
SmartScanCropFragment
SmartScanEnhanceFragment
DocumentCameraView
QuadOverlayView
MaskDebugView
CropOverlayView
EnhanceFilterStrip
```

Useful operational details:

```text
torch button is configurable
rotate button is configurable
auto-capture is configurable
live preview detection is configurable
model debug view can be enabled by config
debug mask/diagnostics are toggled with a triple tap gesture
Bangla strings exist under values-bn
```

This debug mode is especially useful for NID work because it allows inspection of whether failures are caused by:

```text
camera quality
segmentation mask
quad selection
quad validation rules
stabilization
glare/blur
OCR engine weakness
```

## Current Test Coverage

The module already has focused unit tests around the scanner primitives:

```text
EdgeSnapTest
EnhancerTest
GeometryTest
MaskToQuadTest
PerspectiveCorrectorTest
QuadRulesTest
QuadStabilizerTest
QuadValidatorTest
RealMaskTest
SessionBridgeTest
SmartScanConfigApplyQuadRuleTest
ViewportMapperTest
```

So, when extending this project toward OCR, the safest direction is to add OCR-specific tests beside the existing scanner tests rather than mixing OCR logic into camera UI code.

## How This Changes The OCR Plan

The earlier OCR research should now assume:

```text
Android capture/preprocessing baseline = SmartScan
```

The next Android OCR milestone should be:

```text
SmartScan result image
    ↓
OCR engine adapter
    ↓
text boxes + recognized text
    ↓
NID layout parser
    ↓
field-level post-processing
    ↓
structured NIDData
```

Recommended separation:

```text
smartscan module:
document capture, quad detection, crop, enhancement, scanner UI

future OCR/NID module:
OCR engine adapters, text detection/recognition, layout parsing, field validation
```

Do not bury OCR directly inside `DocumentCameraView`, `CameraSession`, or scanner fragments. Keep SmartScan usable as a general capture SDK, and let the Bangladesh NID OCR pipeline consume its output.

## Immediate Integration Questions

Before implementing OCR on Android, answer these with the current SmartScan branch:

```text
1. Which SmartScan output should OCR consume: original, corrected, enhanced, deglare, matte, or multiple candidates?
2. Should OCR run on-device immediately after enhance, or should SmartScan only return an image to a separate OCR module?
3. Should the Android OCR adapter return bounding boxes in SmartScan image coordinates or original camera coordinates?
4. Should NID front/back capture be modeled as SmartScan steps, or as a higher-level flow outside SmartScan?
5. How should OCR confidence combine with SmartScan quad score and enhancement mode?
```

The practical project direction is now:

```text
Use SmartScan as the Android image acquisition and normalization layer.
Benchmark OCR engines on SmartScan outputs.
Keep OCR, NID parsing, and structured field extraction as a separate layer.
```

