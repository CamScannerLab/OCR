# How Tesseract Works

This doc explains what happens between clicking **Run OCR** in the dashboard and seeing
line crops with text and a confidence score. It covers two layers:

1. **Our pipeline**: what `nid_ocr_lab` does before and after calling Tesseract.
2. **Tesseract itself**: how the engine turns pixels into text.

Versions here: Tesseract **5.5.3** with Leptonica 1.87, run with `--oem 1` (LSTM only).
The default is `--psm 6`.

---

## 1. The short answer

You're right: **Tesseract does not read the whole image in one go.** It works in two
phases:

1. **Layout analysis** (no reading yet). Tesseract finds where the text is and cuts the
   image into **blocks → paragraphs → lines → words**.
2. **Recognition**. A neural network (an LSTM) reads **one text line at a time**, left to
   right, and outputs characters.

The "parts" you see in the dashboard are the **text lines** Tesseract found in phase 1.
Each one has the text the LSTM read from it in phase 2.

```mermaid
flowchart LR
    A["Card image"] --> B["Phase 1<br/>Layout analysis<br/><i>where is the text?</i>"]
    B --> C["Line 1 image"]
    B --> D["Line 2 image"]
    B --> E["Line N image"]
    C --> F["Phase 2<br/>LSTM reads line"]
    D --> G["Phase 2<br/>LSTM reads line"]
    E --> H["Phase 2<br/>LSTM reads line"]
    F --> I["Text + boxes + confidence"]
    G --> I
    H --> I
```

---

## 2. The full journey (dashboard → Tesseract → dashboard)

```mermaid
flowchart TD
    subgraph OURS_BEFORE["Our code: before Tesseract (dashboard/server.py, filters.py)"]
        U["Uploaded NID photo"] --> P{"OCR input?"}
        P -- "as_is" --> P1["EXIF fix only"]
        P -- "annotated quad" --> P2["Perspective crop + filter<br/>(enhance / deglare / matte ...)"]
        P -- "filter" --> P3["Filter whole image"]
        P1 --> R
        P2 --> R
        P3 --> R
        R{"Rotation = auto?"}
        R -- yes --> OSD["tesseract --psm 0 (OSD)"]
        OSD -- "confidence ≥ 2.0" --> ROT["Rotate by OSD answer"]
        OSD -- "unsure" --> CHK["OCR at 0/90/180/270<br/>pick highest confidence sum"]
        R -- "no (0/90/180/270)" --> ROT
        CHK --> ROT
    end

    subgraph TESS["Tesseract binary (engines/tesseract.py)"]
        ROT --> T1["1. Load image + DPI"]
        T1 --> T2["2. Thresholding (binarize)"]
        T2 --> T3["3. Layout analysis<br/>blobs → lines → words"]
        T3 --> T4["4. LSTM line recognition"]
        T4 --> T5["5. Decode with dictionary<br/>+ confidence"]
        T5 --> T6["6. TSV output"]
    end

    subgraph OURS_AFTER["Our code: after Tesseract"]
        T6 --> A1["parse_tsv_lines()<br/>group words into lines"]
        A1 --> A2["NIDParser<br/>name / NID no / DOB fields"]
        A1 --> A3["Label panel<br/>line crops + editable text"]
        A3 --> A4["Training dataset<br/>(tesstrain fine-tuning)"]
    end
```

The rest of this doc walks through the Tesseract box, steps 1–6.

---

## 3. Step 1: Load the image and work out scale (DPI)

Tesseract has to know **how big the text is in pixels**, because its models were trained
on text of a certain size. Around **300 DPI** works best, where a capital letter is roughly
30 px tall.

We resolve DPI in `resolve_dpi()` ([tesseract.py](../nid_ocr_lab/engines/tesseract.py)), in this order:

| Order | Source | Result |
|---|---|---|
| 1 | `--dpi` passed explicitly | used as-is |
| 2 | DPI saved in the image (only if 70–1200) | passed as `--dpi` |
| 3 | Nothing | Tesseract estimates it from blob sizes (`Estimating resolution as N` in stderr) |

> **Why it matters:** if text is too small (for example a phone photo shrunk to 800 px
> wide), characters blur together and accuracy drops sharply. More pixels help up to a
> point, then only add time.

---

## 4. Step 2: Thresholding (turn the image black and white)

Layout analysis needs a clean **binary image**: every pixel is either ink or background.
Tesseract does this with **Otsu thresholding** by default, using Leptonica. Otsu picks the
grey level that best separates the dark and light pixels.

```mermaid
flowchart LR
    A["Colour photo"] --> B["Greyscale"] --> C["Otsu threshold"] --> D["Binary image<br/>ink = 1, paper = 0"]
```

```
 Greyscale pixel values          After threshold (T = 128)
 ┌────────────────────┐          ┌────────────────────┐
 │ 230 220  40  35 210│          │  ·   ·   █   █   · │
 │ 225  50  45 215 220│   ──►    │  ·   █   █   ·   · │
 │ 218  42 205 212 228│          │  ·   █   ·   ·   · │
 └────────────────────┘          └────────────────────┘
```

**Weak spot for NID cards:** a single global threshold struggles with **glare, shadows and
the guilloche background pattern**. A bright glare patch can wipe out the text under it.
This is why our `deglare` / `matte` filters run *before* Tesseract. Tesseract 5 also
supports `-c thresholding_method=1` (adaptive Otsu) and `=2` (Sauvola), which set the
threshold locally instead of once for the whole image.

> The binary image drives **layout** (finding the lines). The LSTM recogniser reads from the
> **greyscale** version of each line, which keeps more detail than pure black and white.

---

## 5. Step 3: Layout analysis (dividing the image into parts)

This is the "divide into parts" step you noticed. No characters are recognised yet.

### 5.1 How the parts are found

```mermaid
flowchart TD
    A["Binary image"] --> B["Connected components<br/>each touching group of ink pixels = a 'blob'"]
    B --> C["Filter blobs<br/>drop noise, huge photo areas, lines"]
    C --> D["Estimate typical text height"]
    D --> E["Find columns & text regions<br/>(tab-stop / column finding)"]
    E --> F["Group blobs into text lines<br/>fit a baseline per line (handles slight skew)"]
    F --> G["Split lines into words<br/>using gap sizes between blobs"]
    G --> H["Result: page → block → paragraph → line → word"]
```

The core idea is **connected components**: every separate spot of ink becomes a *blob*,
usually one character or part of one. Blobs of similar height that sit on a shared baseline
get chained into a **text line**.

```
  Blobs found on the image           Blobs chained into lines along a baseline
  ┌───────────────────────────┐      ┌───────────────────────────┐
  │ ▪ ▪▪ ▪ ▪   ▪▪ ▪           │      │[▪ ▪▪ ▪ ▪   ▪▪ ▪]  ← line 1│
  │                           │      │                           │
  │ ▪▪ ▪ ▪▪▪ ▪ ▪▪             │  ──► │[▪▪ ▪ ▪▪▪ ▪ ▪▪]    ← line 2│
  │                           │      │                           │
  │ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪       │      │[▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪]← line 3│
  └───────────────────────────┘      └───────────────────────────┘
```

> **Bengali note:** the *matra* (the horizontal headline, ━) joins the letters of a word into
> one big blob. Tesseract handles this at line level, which is one reason the LSTM reads
> whole lines instead of single characters.

### 5.2 The hierarchy (what the TSV "level" column means)

```mermaid
graph TD
    L1["Level 1: Page"] --> L2a["Level 2: Block"]
    L1 --> L2b["Level 2: Block"]
    L2a --> L3a["Level 3: Paragraph"]
    L3a --> L4a["Level 4: Line<br/>'Name: MD. RAHIM'"]
    L3a --> L4b["Level 4: Line<br/>'Date of Birth 01 Jan 1990'"]
    L4a --> W1["Level 5: Word 'Name:'"]
    L4a --> W2["Level 5: Word 'MD.'"]
    L4a --> W3["Level 5: Word 'RAHIM'"]
```

Every row of Tesseract's TSV output carries `page_num, block_num, par_num, line_num,
word_num`, a bounding box and a confidence. Our `parse_tsv_lines()` keeps only **level 5
(word)** rows, groups them by `(page, block, paragraph, line)`, and joins each line's word
boxes into one line box. **These line boxes are the crops in the label panel.**

### 5.3 PSM decides how much layout analysis runs

**PSM (Page Segmentation Mode)** tells Tesseract what kind of page to expect, so it can skip
parts of the layout analysis.

```mermaid
flowchart TD
    Q{"What does the image contain?"}
    Q -- "Only need rotation/script" --> P0["psm 0: OSD only"]
    Q -- "Full page, unknown layout" --> P3["psm 3: fully automatic<br/>(finds columns + blocks)"]
    Q -- "One column, mixed sizes" --> P4["psm 4"]
    Q -- "One uniform block of text" --> P6["psm 6: our default<br/>(skips column finding)"]
    Q -- "A single line" --> P7["psm 7"]
    Q -- "A single word" --> P8["psm 8"]
    Q -- "Scattered text, no order" --> P11["psm 11: sparse text"]
```

| PSM | Meaning | NID card fit |
|---|---|---|
| 0 | Orientation & script detection only | used by our `auto` rotation |
| 3 | Fully automatic layout (Tesseract's default) | can split the photo and text into odd blocks |
| 4 | Single column of variable-size text | often good for card fields |
| **6** | **Single uniform block** | **our default**: whole card = 1 block, lines still found |
| 7 | Single text line | good for **pre-cropped field** images |
| 8 | Single word | NID number crop |
| 11 | Sparse text, find as much as possible | cluttered backgrounds |

The dashboard's **sweep** strategy runs PSM `3, 4, 6, 11` and keeps the highest-scoring
result. That score is a heuristic, not measured accuracy.

---

## 6. Step 4: LSTM line recognition (the actual reading)

With `--oem 1`, every line image goes into an **LSTM neural network**. The pre-LSTM
"legacy" engine (`--oem 0`) matched character shapes one at a time; the `tessdata_fast` and
`tessdata_best` models we use don't include it.

### 6.1 What happens to one line

```mermaid
flowchart LR
    A["Line crop<br/>(greyscale)"] --> B["Normalise height<br/>(scaled to a fixed<br/>height, e.g. 36 px)"]
    B --> C["Slide left → right<br/>one thin column<br/>per time step"]
    C --> D["Conv layers<br/>local stroke features"]
    D --> E["Bidirectional LSTM<br/>reads context<br/>forwards + backwards"]
    E --> F["Softmax per step<br/>probability for every<br/>character in unicharset"]
    F --> G["CTC beam search<br/>+ dictionary"]
    G --> H["'RAHIM'  conf 91"]
```

### 6.2 The sliding view, visually

The network doesn't cut the line into characters first. It sweeps across the line, and at
every small step it outputs a probability for each possible character, or for *blank*
(nothing new here).

```
 Line image:   ┃ R ┃ A ┃ H ┃ I ┃ M ┃
               ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓      (one arrow = one time step)

 Best char:    R R _ A A _ H _ I I M
               └┬┘   └┬┘   │   └┬┘ │
 CTC collapse:  R     A    H    I  M   → "RAHIM"
               (merge repeats, drop blanks "_")
```

This is **CTC (Connectionist Temporal Classification)** decoding:
- merge repeated characters that sit next to each other,
- remove the blank symbol.

This is why the LSTM copes with joined or touching letters, like Bengali conjuncts and
matra-connected words, better than a character-by-character engine.

### 6.3 Why "bidirectional" matters

A character is easier to read when you know its neighbours. The forward LSTM carries
context from the left and the backward LSTM from the right. For example, a smudged glyph
between `1 9` and `0` is much more likely to be a digit than the letter `O`.

### 6.4 Bengali specifics

- The model's **unicharset** lists every character or grapheme unit it can output.
- Bengali uses **recoding**: complex graphemes (consonant + vowel sign + hasanta
  combinations) are broken into smaller codes, so the network predicts short code
  sequences rather than thousands of separate conjunct classes.
- With `-l eng+ben` **both models run**, and Tesseract keeps whichever reading scores better
  for each word or line. Language order and choice affect both speed and accuracy.

---

## 7. Step 5: Dictionary, decoding and confidence

The beam search keeps several candidate readings at once (a "beam"). It scores them using:

- the **LSTM probabilities** for each character,
- **dictionary / word lists** (DAWGs packed inside `ben.traineddata` / `eng.traineddata`),
  which push candidates toward real words,
- character-class rules (digits next to digits, and so on).

```mermaid
flowchart LR
    A["LSTM outputs"] --> B["Beam search<br/>keeps top-N candidates"]
    D["Dictionary<br/>(word DAWG)"] --> B
    B --> C1["'RAHIM'  score 0.91 ✅"]
    B --> C2["'RAH1M'  score 0.42"]
    B --> C3["'PAHIM'  score 0.18"]
```

> **NID caveat:** dictionaries help with ordinary words but can **hurt names and ID numbers**.
> A rare name may get "corrected" toward a common word. For number-only crops, a
> whitelist such as `-c tessedit_char_whitelist=0123456789` together with `--psm 7` is safer.

The **confidence** (0–100) in the TSV is derived from the winning path's probabilities. We
divide it by 100 per word. A line's confidence in the dashboard is the **average of its word
confidences**. Treat it as a rough signal, not a guarantee: a word can be wrong with 90%
confidence.

---

## 8. Step 6: Output

We ask for **TSV** (`-c tessedit_create_tsv=1`). One row per page, block, paragraph, line
and word:

```
level page_num block_num par_num line_num word_num left top width height conf  text
1     1        0         0       0        0        0    0   1200  760    -1
2     1        1         0       0        0        40   60  900   420    -1
4     1        1         1       1        0        40   60  620   38     -1
5     1        1         1       1        1        40   60  110   38     93.1  Name:
5     1        1         1       1        2        165  60  95    38     90.4  MD.
5     1        1         1       1        3        275  61  140   37     91.2  RAHIM
```

Our code then does the following:

```mermaid
sequenceDiagram
    participant D as Dashboard (app.js)
    participant S as server.py
    participant T as tesseract binary
    participant E as TesseractEngine
    participant N as NIDParser

    D->>S: POST /api/ocr/run (filter, rotation, psm, variant)
    S->>S: crop / filter / EXIF fix
    opt rotation = auto
        S->>T: tesseract --psm 0 (OSD)
        T-->>S: Rotate: 90, confidence 5.3
    end
    S->>E: recognize(rotated image, eng+ben, psm 6)
    E->>T: tesseract img stdout --oem 1 --psm 6 -c tessedit_create_tsv=1
    T-->>E: TSV rows (levels 1–5)
    E->>E: parse_tsv_lines() → line blocks + words + full_text
    E-->>S: OCRResult
    S->>N: parse(OCRResult)
    N-->>S: NIDData (name, NID no, DOB, ...)
    S->>S: save_run() for label panel crops
    S-->>D: ocr + parsed + run_id
    D->>D: render line crops, text, confidence %
```

---

## 9. What each knob changes

| Knob | Where | Affects step | Typical effect |
|---|---|---|---|
| Filter (`enhance`, `deglare`, …) | dashboard | 2 Thresholding | removes glare and background so blobs are clean |
| Perspective crop | dashboard | 3 Layout | straight, card-only image means fewer false blocks |
| Rotation (`auto` / OSD) | dashboard | 3 Layout | upside-down text produces garbage lines |
| `--dpi` / image size | engine | 1, 3, 4 | wrong scale: merged or split blobs, bad line height |
| `--psm` | dashboard | 3 Layout | how the image is divided into parts |
| `--oem 1` | engine | 4 | LSTM only |
| Variant `system` (fast) vs `best` | dashboard | 4 | `best` = bigger float model: slower, usually more accurate |
| `custom/<name>` | training | 4, 5 | model fine-tuned on our NID line crops |
| `-l eng+ben` | dashboard | 4, 5 | which models and dictionaries run |
| `OMP_THREAD_LIMIT` | engine | all | CPU threads per call (throughput tuning) |

---

## 10. How this connects to training

The line crops from Step 3 plus your corrected text become the training data:

```mermaid
flowchart LR
    A["Run OCR"] --> B["Line crops<br/>(from layout step)"]
    B --> C["You correct the text<br/>in label panel"]
    C --> D["Line image + .gt.txt pairs<br/>(training/dataset.py)"]
    D --> E["tesstrain fine-tunes the LSTM<br/>(training/tesstrain_runner.py)"]
    E --> F["custom/&lt;name&gt;.traineddata"]
    F --> G["Pick 'custom' variant<br/>in dashboard"]
    G --> A
```

Fine-tuning changes **only Step 4–5** (the LSTM weights and unicharset). It **does not**
change layout analysis. If lines are being split wrongly, fix that with preprocessing,
crop, rotation or PSM; training won't help.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Blob** | A connected group of ink pixels; roughly one character or piece of one |
| **Baseline** | The imaginary line text sits on; used to group blobs into lines |
| **OSD** | Orientation & Script Detection (`--psm 0`) |
| **PSM** | Page Segmentation Mode: how the page is divided into parts |
| **OEM** | OCR Engine Mode: `0` legacy, `1` LSTM, `2` both, `3` default |
| **LSTM** | Long Short-Term Memory, a neural network that reads sequences with memory of context |
| **CTC** | Decoding that turns per-step predictions into a string (merge repeats, drop blanks) |
| **Unicharset** | The set of characters a model can output |
| **DAWG** | Compressed word list (dictionary) inside `.traineddata` |
| **traineddata** | Model file: LSTM weights, unicharset, dictionaries |
| **tessdata_fast / best** | Official model sets: fast = 8-bit integer, small; best = float, more accurate |
