# Bengali–English NID OCR: accuracy, speed, and cost plan

Date: 2026-09-14  
Status: proposed implementation plan; no new accuracy or latency benchmark was run for this document.

## Immediate step: EasyOCR dashboard trial (2026-09-15)

User priority: try EasyOCR first; resume the improvement work below after reviewing its results. The dashboard now includes a CPU EasyOCR adapter with `eng`, `ben`, and `eng+ben`, cached readers, normalized boxes/confidence, raw output, and initialization/inference timings. The existing parser remains unchanged.

Use the same SmartScan image/filter for comparisons, choose rotation 0 for upright cards, and compare cold versus warm runs. EasyOCR auto mode rotates detected text boxes, unlike the existing full-card rotation search in Tesseract/PaddleOCR; keep rotation policy explicit in reports. Inspect Bengali names, English names, NID, and DOB independently. A quick dashboard trial is not a labeled accuracy benchmark.

Add EasyOCR to every baseline/field-level comparison below. Test it as both a complete engine and a Bengali-field challenger. Later experiments: greedy versus beam search, detection scale, crop recognition, and CPU versus supported acceleration. Promote it only if held-out accuracy and total latency/cost justify doing so. It has local compute cost but no per-request API fee. See the [official API](https://www.jaided.ai/easyocr/documentation/) and [language/model configuration](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py).

## Recommendation

Build a **field-aware local OCR pipeline**, starting with Tesseract. Read Bengali names with a Bengali recognizer, English names with an English recognizer, and numbers with a constrained recognizer. Retry only uncertain fields. Use Gemini selectively for unresolved cards if cloud processing is acceptable.

This is the most promising near-term route to the three goals. It is a hypothesis to validate on this NID dataset, not a promise that Tesseract will match Gemini. If Bengali accuracy remains inadequate after a bounded tuning effort, replace the Bengali recognition component with a specialized model rather than making every card pass through a large vision-language model.

Keep SmartScan responsible for capture, card detection, perspective correction, and enhancement variants. This plan starts from its card outputs. OCR field cropping and layout parsing belong here; rebuilding the scanner does not.

## 1. Fix the comparison before judging the engines

The current source contains issues that can explain part of the reported behavior:

| Observed in this workspace | Consequence | First action |
|---|---|---|
| `paddle_language_for()` always returns `en`, including for `ben` and `eng+ben`. | The current Paddle path is not a valid Bengali recognition benchmark. | Declare actual supported languages per model; reject unsupported requests instead of silently using English. |
| `PaddleOCREngine.recognize()` constructs `PaddleOCR` every time. | Repeated model initialization is included in each recognition call. Auto rotation invokes it four times. | Load a pinned model once per worker and reuse it. Measure initialization separately. |
| Tesseract dashboard auto mode tries four rotations × four PSMs. | One request executes **16 full-image OCR calls**. Manual rotation still executes four. | Keep exhaustive search as an explicit research mode; introduce a bounded production path. |
| The dashboard returns the selected candidate's `latency_ms`. | Displayed OCR time omits the other candidates and preprocessing; it is not total request latency. | Add an outer request timer and per-stage/per-attempt timings. |
| Tesseract starts a subprocess for every call. | Language loading and temporary-file overhead repeat. | Profile startup, then evaluate persistent native API workers. |
| TSV words are joined with newlines; the parser uses line proximity. | Word boundaries can be mistaken for text lines, splitting labels and names. | Preserve TSV page/block/paragraph/line IDs and reconstruct lines geometrically. |
| OCR from all PSMs at the chosen rotation is concatenated for parsing. | Conflicting/duplicate candidates can contaminate extracted fields. | Keep candidates separate and choose per field with provenance. |
| The parser routes Bengali labels into English-named fields and assigns fixed confidence values. | Script confusion and apparently confident incorrect fields can hide recognition quality. | Separate Bengali/English fields and calibrate acceptance using labeled outcomes. |
| Working card images are repeatedly JPEG-encoded. | Encoding overhead and compression may affect small marks. | Benchmark lossless PNG or in-memory images after SmartScan output. |

Evidence: `nid_ocr_lab/engines/{tesseract,paddleocr}.py`, `nid_ocr_lab/dashboard/server.py`, `nid_ocr_lab/parsers/nid_parser.py`, and `nid_ocr_lab/evaluation.py`.

`Developer.md` is partly stale: PaddleOCR is described as unimplemented, but an adapter now exists. The probe report in `benchmark/tesseract_probe/summary.json` contains confidence/text heuristics, not labeled field accuracy. Neither proves which engine is best. The user's observed 10× slowdown needs remeasurement after the above corrections.

## 2. Establish a trustworthy benchmark

### Dataset

Start with 50–100 manually verified cards for diagnosis, then grow toward 300–500 distinct cards where available. These are planning sizes, not existing annotated counts. Cover supported card layouts and sides, Bengali/English names, long names, small print, glare, blur, low contrast, and imperfect SmartScan outputs.

- Label exact visible field text, field boxes, script, layout, and whether a field is absent or unreadable. Never use Gemini output as unverified ground truth.
- Split by identity/card **before** generating variants. All captures, filters, crops, and synthetic derivatives of one card stay in one split.
- Use development data for tuning; keep a final test set untouched. Report actual distinct-card counts. A small test set cannot establish very high reliability.
- Keep annotation-assisted ideal crops as a diagnostic track. Evaluate deployment performance using actual SmartScan-produced crops as a separate track.

### Metrics and protocol

| Goal | Measurement |
|---|---|
| Accuracy | Exact match per field and per script; all-required-fields-correct per card; Bengali character error rate (CER), plus grapheme-level errors if practical |
| Safe acceptance | Accuracy among automatically accepted cards **and** percentage automatically accepted; count wrong accepted NID/DOB values separately |
| Speed | End-to-end p50/p95, cold start, warm engine latency, attempts/card, throughput at fixed concurrency, peak memory |
| Cost | Compute/card, Gemini fallback rate, actual billed API usage, review/recapture rate, and cost per correctly completed card |

Apply documented Unicode NFC and whitespace normalization; retain raw text. Normalize Bengali digits for numeric comparisons, not by deleting meaningful Bengali marks. Evaluate only annotated applicable fields; do not inflate accuracy by counting missing labels as correct empty predictions. Fix date parsing to canonical ISO dates and validate real calendar dates.

Pin engine/package versions, model hashes, language data, hardware, input size, thread count, and preprocessing. Download models before timing. Run one cold-start test and at least three randomized warm repetitions over the same cards, with caching disabled for engine comparisons. Include preprocessing, all retries, parsing, and network time in request latency.

Initial engineering targets, to revise after baseline: ≥3× faster p95 than the current exhaustive dashboard path, reduced Bengali CER with no critical-field regression, and ≤10% Gemini fallback while preserving agreed acceptance quality. These are targets, not measured results. Set an absolute latency budget on the intended deployment hardware before selecting the final configuration.

## 3. Improve Tesseract in this order

### A. Remove redundant work

1. Use trustworthy upstream orientation metadata when available. Otherwise determine orientation once using a small orientation/layout check, with bounded fallback for uncertain cases. Perspective correction alone does not guarantee upright text.
2. Select one validated full-card configuration or the field pipeline below. Remove the 16-call search from ordinary requests; retain it offline to discover useful profiles.
3. Preserve inputs in memory or lossless files. Avoid re-encoding a zero-degree rotation.
4. Compare `OMP_THREAD_LIMIT=1`, `2`, and `4` on the actual CPU. Benchmark single-card latency separately from concurrent throughput. More threads are not automatically faster. Tesseract documents thread control in its [FAQ](https://github.com/tesseract-ocr/tessdoc/blob/main/FAQ.md).
5. If initialization is material, introduce a persistent `TessBaseAPI` binding with one instance per worker/profile. Do not share a mutable instance across concurrent requests. Reset request/adaptive state and verify repeatability. A long-lived Python worker that still shells out does not eliminate Tesseract startup.

### B. Read fields instead of the whole mixed-language card

Detect the supported NID layout from anchors and geometry; use normalized field regions with padding. Check alignment and text containment. Unknown or badly aligned layouts fall back to text detection/full-card OCR rather than blindly applying fixed rectangles.

| Field region | Initial profile to test | Constraint |
|---|---|---|
| Bengali name / parent names | `ben`, OEM 1, PSM 7 | Preserve vowels, conjuncts, and upper/lower marks |
| English name | `eng`, OEM 1, PSM 7 | Keep spelling and punctuation supported by the field |
| NID number | Script-appropriate recognizer, OEM 1, PSM 7 | Digit whitelist only on this numeric ROI; normalize Bengali digits afterward |
| DOB | `eng` or script-appropriate model, OEM 1, PSM 7 | Retain letters when months are written as words |
| Multiline address | Script-specific model, OEM 1, PSM 6 | Preserve line order and complete field boundaries |

PSM 7 treats input as one line; PSM 6 treats it as one block. Try PSM 13 as a line-level challenger. Language order (`ben+eng` versus `eng+ben`) remains an offline experiment for genuinely mixed regions. Tesseract's [quality guide](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html) documents segmentation modes, borders, and field constraints.

Field OCR can create several calls, so benchmark **total card latency** and use persistent workers where useful. Do not assume smaller crops always beat one full-card pass. Restore each crop's coordinates to the shared card coordinate system.

### C. Benchmark model families and image variants

Use separate, versioned model directories; first identify the provenance/hash of the installed `ben.traineddata`. Compare `tessdata_fast`, installed models, and `tessdata_best` with OEM 1. Fast uses a smaller integer model; best trades speed for accuracy and supplies trainable base models. The published ranking is not a Bengali NID guarantee. See [official model comparison](https://github.com/tesseract-ocr/tessdoc/blob/main/Data-Files.md).

Suggested policy: choose the fastest profile meeting field accuracy requirements; use best only for uncertain Bengali regions if its measured gain justifies the cost.

Benchmark SmartScan corrected/original, gray, and enhance variants first. Add threshold/deglare/matte only for failure classes they improve. Compare a few real pixel scales; changing DPI metadata alone creates no missing detail. Inspect internal thresholded images on failures. Avoid aggressive erosion/sharpening that destroys Bengali marks, and avoid chopping lines through connected script components. Keep the production variant set small.

### D. Correct conservatively and retry selectively

Reconstruct lines from boxes, then associate labels and values spatially. Keep raw and normalized text, source boxes, model profile, alternatives, and correction reason.

Use dictionaries as soft ranking evidence for names; never replace an unusual identity name just because another spelling is common. Do not generate an English name by translating/transliterating Bengali when the card contains an English field. Numeric length/format checks and valid dates are necessary but do not prove the reading is correct. Use only independently verified NID format rules; do not invent a checksum.

Gate retries on field completeness, script, geometry, calibrated confidence, and ambiguity. Raw engine confidence is not a probability of correctness and is not comparable across engines. Retry each failed region at most once with a selected alternate model/variant initially; also impose a total card deadline. Escalate unresolved fields instead of silently filling plausible values.

### E. Fine-tune only after diagnosing residual recognition errors

If clean, correctly segmented Bengali crops still fail consistently, run a time-boxed training experiment. Start with roughly 2,000–5,000 corrected real line crops if obtainable; add synthetic lines with relevant fonts, names, conjuncts, and realistic degradations. This is a starting data budget, not a sufficiency claim. Split by card/person and reserve unseen names/fonts for evaluation.

Use a trainable `tessdata_best` Bengali base and the maintained [tesstrain workflow](https://github.com/tesseract-ocr/tesstrain). Compare the trained model with its unmodified base on identical held-out crops, then the whole pipeline. Quantize/export only after accuracy is established and remeasure afterward. Stop if gains do not justify training and maintenance costs.

## 4. Give PaddleOCR a fair, bounded evaluation

Correct the English-only mapping first. Verify the **specific recognition model's** Bengali character dictionary and documented support. “Multilingual” and “100+ languages” do not establish Bengali support. The official general OCR page checked for this plan does not list Bengali; a [Bengali support discussion](https://github.com/PaddlePaddle/PaddleOCR/discussions/17776) also records a request and a future-support response. Verify again against the exact installed release; do not guess a `bn` configuration. PaddleOCR-VL is a separate model family from lightweight PP-OCR.

Then reuse a model per worker, explicitly pin model/version/device, disable redundant unwarping/orientation modules when input guarantees permit, and test mobile models against server models. Batch recognition of field crops where supported. The [official pipeline guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html) exposes these controls and currently shows PP-OCRv6 defaults, making version pinning especially important.

Validate result conversion for the installed API: consume iterators, handle result objects/NumPy arrays, and preserve boxes and scores. The current list/dict parser can miss other result representations or encounter NumPy truth-value issues. Verify engine output versus adapter output before scoring accuracy.

Benchmark CPU runtime/thread settings on this Mac and separately on the intended production hardware. Do not assume an NVIDIA optimization works on Apple Silicon. ONNX/RapidOCR can be deployment experiments, but a faster runtime does not add Bengali recognition to an incompatible checkpoint.

## 5. Better alternatives and when to use them

| Option | Accuracy prospect | Speed/cost tradeoff | Decision |
|---|---|---|---|
| EasyOCR Bengali + English | Now available for the initial dashboard trial; measure field accuracy | Cached local CPU inference; first load and auto rotation add time | Evaluate first, then choose its role |
| Tuned field-aware Tesseract | Plausible improvement for structured printed fields; Bengali must be measured | Local inference, modest infrastructure | First implementation |
| Tesseract + selective Gemini | Uses the user's observed Gemini quality on difficult cases | Adds cloud latency and fees only on fallback; verify actual billing | Preferred near-term hybrid candidate |
| Specialized Bengali recognizer + English/digit Tesseract | Targets the weakest script directly | Training/integration cost; potentially compact inference | Next step if Bengali remains below target |
| Correctly configured PaddleOCR | Useful English/detection challenger; Bengali depends on checkpoint | Warm-up and model choice matter | Retest after adapter fixes |
| Qwen or PaddleOCR-VL for every card | Requires this dataset's evidence | Larger model runtime can undermine speed/cost goals | Lower priority given the user's Qwen results |

For a Bengali challenger, investigate the authors' [bbOCR project/paper](https://arxiv.org/abs/2308.10647), which targets Bengali document recognition. Benchmark its recognition component on NID field crops and inspect model licensing/dependencies before adoption; no claim is made that it beats the local baseline. A small custom line recognizer is another option if sufficient labeled data and training capacity exist.

For Gemini fallback, send unresolved crops with enough label context and request a compact fixed schema; benchmark crops against full-card context. Smaller crops do not necessarily reduce billing because image accounting can use minimums/tiles. Require explicit unreadable/unknown output and validate results. Keep uncertain disagreements for review. This document authorizes no upload or paid API run.

Cost model:

```text
average cost/card = local compute + fallback_rate × observed API cost/fallback
                  + review_rate × review cost + amortized training/maintenance

average latency = local latency + retry_rate × retry latency
                + fallback_rate × fallback latency
```

These latency terms assume a sequential pipeline and averages; measure p95 directly. At 10% fallback, the API-call component is about 90% lower than calling Gemini on every card **only if per-call charges are comparable**. Total savings will differ. Optimize cost per correctly completed card, not raw calls or nominally free inference.

## 6. Implementation schedule and decision gates

Indicative effort for one engineer, excluding annotation delays and training compute:

| Phase | Effort | Deliverable / exit condition |
|---|---|---|
| 1. Baseline and adapter correctness | 1–2 days plus labeling | Correct language reporting; actual end-to-end timing; pinned environment; verified field manifest |
| 2. Bounded fast path | 2–3 days | No exhaustive default search; reusable Paddle models; thread/startup profiling; cold/warm comparison |
| 3. Layout and field OCR | 3–5 days | Script-specific field profiles, reconstructed lines, safe unknown-layout fallback, corrected parser/schema |
| 4. Quality tuning and retry gates | 2–3 days | Small ablation report; calibrated acceptance and fallback; accuracy/latency/cost comparison |
| 5. Optional specialist/training | 1–2 weeks initially | Held-out gain justifies integration; otherwise stop and retain the measured hybrid |

Run experiments incrementally: baseline → call reduction → field/script routing → model family → image variant → retry → training. Change one major factor at a time; select on development data, then run the final locked configuration once on held-out cards. Do not run the full parameter Cartesian product in production.

Planned file changes:

- `nid_ocr_lab/engines/tesseract.py`: explicit model/OEM/config profiles, line metadata, detailed timings, optional persistent backend.
- `nid_ocr_lab/engines/paddleocr.py`: honest language capabilities, pinned reusable models, robust result normalization.
- `nid_ocr_lab/dashboard/server.py`: research versus production modes, bounded retries, total timing and attempt counts.
- `nid_ocr_lab/models.py` and `parsers/nid_parser.py`: script-specific fields, spatial parsing, date normalization, evidence-based acceptance.
- `nid_ocr_lab/evaluation.py`: annotated-field denominators, per-script/field metrics, CER, accepted accuracy and coverage.
- New benchmark runner/configs: reproducible model/variant sweeps and latency/cost reports; update `Developer.md` to match implemented behavior.

First milestone: publish a report answering **how much of the current failure comes from recognition, language selection, layout/parsing, or repeated work**. Select the fastest, lowest-cost pipeline that meets the measured accuracy requirement; do not commit to Tesseract alone if Bengali remains the limiting factor.
