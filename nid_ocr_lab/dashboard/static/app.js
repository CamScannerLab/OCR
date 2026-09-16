const state = {
  samples: [],
  filtered: [],
  selected: null,
  annotation: null,
  health: null,
  ocrStatus: null,
  overlay: true,
  fit: true,
  autoRotation: null,
};

const sampleList = document.querySelector("#sampleList");
const sampleCount = document.querySelector("#sampleCount");
const visibleCount = document.querySelector("#visibleCount");
const search = document.querySelector("#search");
const sampleTitle = document.querySelector("#sampleTitle");
const sampleMeta = document.querySelector("#sampleMeta");
const sourceSelect = document.querySelector("#sourceSelect");
const filterSelect = document.querySelector("#filterSelect");
const sourceImage = document.querySelector("#sourceImage");
const filterImage = document.querySelector("#filterImage");
const overlayCanvas = document.querySelector("#overlayCanvas");
const healthDetails = document.querySelector("#healthDetails");
const fileDetails = document.querySelector("#fileDetails");
const annotationDetails = document.querySelector("#annotationDetails");
const ocrEngineSelect = document.querySelector("#ocrEngineSelect");
const ocrLanguageSelect = document.querySelector("#ocrLanguageSelect");
const ocrRotationSelect = document.querySelector("#ocrRotationSelect");
const ocrModelLabel = document.querySelector("#ocrModelLabel");
const ocrModelSelect = document.querySelector("#ocrModelSelect");
const runOcrButton = document.querySelector("#runOcrButton");
const ocrStatus = document.querySelector("#ocrStatus");
const ocrSummaryOutput = document.querySelector("#ocrSummaryOutput");
const parsedFieldsOutput = document.querySelector("#parsedFieldsOutput");
const rawTextOutput = document.querySelector("#rawTextOutput");
const ocrJsonOutput = document.querySelector("#ocrJsonOutput");
const engineRawOutput = document.querySelector("#engineRawOutput");
const toggleOverlay = document.querySelector("#toggleOverlay");
const fitMode = document.querySelector("#fitMode");
const ocrInputSelect = document.querySelector("#ocrInputSelect");
const ocrPsmLabel = document.querySelector("#ocrPsmLabel");
const ocrPsmSelect = document.querySelector("#ocrPsmSelect");
const ocrStrategyLabel = document.querySelector("#ocrStrategyLabel");
const ocrStrategySelect = document.querySelector("#ocrStrategySelect");
const uploadZone = document.querySelector("#uploadZone");
const uploadInput = document.querySelector("#uploadInput");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const labelPanel = document.querySelector("#labelPanel");
const labelRows = document.querySelector("#labelRows");
const labelDataset = document.querySelector("#labelDataset");
const labelCardId = document.querySelector("#labelCardId");
const labelStatus = document.querySelector("#labelStatus");
const labelSave = document.querySelector("#labelSave");
const labelSelectAll = document.querySelector("#labelSelectAll");
const sidebar = document.querySelector(".sidebar");
const ocrPickFileButton = document.querySelector("#ocrPickFileButton");
const ocrFileInfo = document.querySelector("#ocrFileInfo");
const filterHeading = document.querySelector("#filterHeading");

async function boot() {
  const healthResponse = await fetch("/api/health");
  state.health = await healthResponse.json();
  const ocrStatusResponse = await fetch("/api/ocr/status");
  state.ocrStatus = await ocrStatusResponse.json();
  healthDetails.textContent = JSON.stringify(state.health, null, 2);
  renderOcrStatus();
  await loadSamples();
  if (state.samples.length) {
    selectSample(state.samples[0].id);
  }
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  state.samples = await response.json();
  sampleCount.textContent = state.samples.length;
  renderList();
}

function isUpload(sample) {
  return sample?.source === "upload";
}

function renderOcrStatus() {
  const engine = selectedOcrEngine();
  renderModelOptions(engine);
  if (!engine) {
    ocrStatus.textContent = "No OCR engine selected.";
    runOcrButton.disabled = true;
    return;
  }
  if (!engine.available) {
    ocrStatus.textContent = engine.note || `${engine.label} is not installed or not on PATH.`;
    runOcrButton.disabled = true;
    return;
  }
  const modelText = isVisionEngine(engine.id) && ocrModelSelect.value ? ` · model ${ocrModelSelect.value}` : "";
  ocrStatus.textContent = `${engine.label} ready · ${engine.languages.length || 0} language options found${modelText}`;
  if (engine.id === "tesseract") {
    if (!Array.isArray(engine.variants)) {
      ocrStatus.textContent = "The dashboard server is running older code than this page. Restart it (Ctrl+C, then bash scripts/run_dashboard.sh).";
      runOcrButton.disabled = true;
      return;
    }
    const variant = selectedTesseractVariant();
    const versions = Object.entries(variant?.versions || {}).map(([lang, version]) => `${lang} ${version || "?"}`).join(" · ");
    ocrStatus.textContent = `Tesseract · ${variant?.label || "system"} · ${versions || "no model info"} · auto rotation = OSD, 4-way check only if OSD is unsure`;
    const missing = ocrLanguageSelect.value.split("+").filter((lang) => !(variant?.languages || []).includes(lang));
    if (missing.length) {
      ocrStatus.textContent = `Model folder "${variant?.label}" has no ${missing.join(", ")} model. Pick another language or run scripts/fetch_tessdata.sh.`;
      runOcrButton.disabled = true;
      return;
    }
  }
  if (engine.id === "easyocr") {
    ocrStatus.textContent = "EasyOCR · Bengali + English · CPU · first run loads models. Auto rotates text boxes; choose 0 for fastest upright-card lookup.";
  }
  runOcrButton.disabled = false;
}

function selectedOcrEngine() {
  return state.ocrStatus?.engines?.find((item) => item.id === ocrEngineSelect.value);
}

function isVisionEngine(engineId) {
  return engineId === "gemini_vision" || engineId === "lmstudio_vision";
}

function selectedTesseractVariant() {
  const engine = state.ocrStatus?.engines?.find((item) => item.id === "tesseract");
  return (engine?.variants || []).find((variant) => variant.id === ocrModelSelect.value) || engine?.variants?.[0];
}

function renderModelOptions(engine) {
  const isTesseract = engine?.id === "tesseract";
  ocrPsmLabel.hidden = !isTesseract;
  ocrStrategyLabel.hidden = !isTesseract;
  if (isTesseract) {
    ocrModelLabel.hidden = false;
    ocrModelSelect.disabled = false;
    const previous = ocrModelSelect.value;
    const variants = engine.variants || [];
    if (ocrModelSelect.dataset.engine !== "tesseract" || ocrModelSelect.options.length !== variants.length) {
      ocrModelSelect.innerHTML = "";
      for (const variant of variants) {
        const option = document.createElement("option");
        option.value = variant.id;
        option.textContent = variant.label;
        ocrModelSelect.appendChild(option);
      }
      ocrModelSelect.dataset.engine = "tesseract";
      const preferred = variants.some((variant) => variant.id === "best") ? "best" : variants[0]?.id;
      ocrModelSelect.value = variants.some((variant) => variant.id === previous) ? previous : (preferred || "");
    }
    return;
  }
  ocrModelSelect.dataset.engine = "";
  const show = Boolean(engine && isVisionEngine(engine.id));
  ocrModelLabel.hidden = !show;
  ocrModelSelect.disabled = !show;
  if (!show) {
    ocrModelSelect.innerHTML = "";
    return;
  }
  const previous = ocrModelSelect.value;
  const models = [...new Set([...(engine.models || []), engine.model].filter(Boolean))];
  ocrModelSelect.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model === engine.model ? `${model} (default)` : model;
    ocrModelSelect.appendChild(option);
  }
  ocrModelSelect.value = models.includes(previous) ? previous : (engine.model || models[0] || "");
}

function renderList() {
  const term = search.value.trim().toLowerCase();
  state.filtered = state.samples.filter((sample) => {
    return !term || sample.id.toLowerCase().includes(term) || sample.source.toLowerCase().includes(term);
  });
  visibleCount.textContent = state.filtered.length;
  sampleList.innerHTML = "";
  for (const sample of state.filtered) {
    const row = document.createElement("div");
    row.className = "sample-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample-item ${state.selected?.id === sample.id ? "active" : ""}`;
    if (isUpload(sample)) {
      button.innerHTML = `<strong>${escapeHtml(sample.filename)}</strong><span>upload · ${sample.kind} · ${Object.keys(sample.images).length} page(s)</span>`;
    } else {
      button.innerHTML = `<strong>${sample.id}</strong><span>${sample.source} · ${badges(sample)}</span>`;
    }
    button.addEventListener("click", () => selectSample(sample.id));
    row.appendChild(button);
    if (isUpload(sample)) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "sample-delete";
      remove.title = "Delete this upload from benchmark/uploads";
      remove.textContent = "✕";
      remove.addEventListener("click", () => deleteUpload(sample));
      row.appendChild(remove);
    }
    sampleList.appendChild(row);
  }
}

function badges(sample) {
  const parts = Object.keys(sample.images || {});
  if (sample.raw) parts.push("raw");
  if (sample.annotation) parts.push("annotation");
  if (sample.mask) parts.push("mask");
  if (sample.overlay) parts.push("overlay");
  parts.push(...Object.keys(sample.variants || {}));
  return parts.join(", ") || "file";
}

async function selectSample(id) {
  state.selected = state.samples.find((sample) => sample.id === id);
  state.annotation = null;
  ocrInputSelect.value = "filter";
  labelCardId.value = isUpload(state.selected) ? state.selected.upload_id : (state.selected?.id || "");
  renderList();
  await loadAnnotation();
  renderSample();
}

async function loadAnnotation() {
  if (!state.selected?.annotation) return;
  const response = await fetch(`/api/annotation?path=${encodeURIComponent(state.selected.annotation)}`);
  state.annotation = await response.json();
}

function renderSample() {
  const sample = state.selected;
  if (!sample) return;
  sampleTitle.textContent = isUpload(sample) ? sample.filename : `Sample ${sample.id}`;
  ocrFileInfo.classList.toggle("current", isUpload(sample));
  ocrFileInfo.textContent = isUpload(sample)
    ? `Selected upload: ${sample.filename} · ${sample.kind} · ${Object.keys(sample.images).length} page(s) — pick the page in Source, then Run OCR.`
    : `Current input: SmartScan sample ${sample.id}. Choose a file to OCR your own image or PDF instead.`;
  sampleMeta.textContent = `${sample.source} · ${badges(sample)}`;
  renderSourceOptions(sample);
  fileDetails.textContent = JSON.stringify(sample, null, 2);
  annotationDetails.textContent = JSON.stringify(state.annotation || {}, null, 2);
  updateImages();
}

function renderSourceOptions(sample) {
  const previous = sourceSelect.value;
  sourceSelect.innerHTML = "";
  const options = [];
  for (const [name, path] of Object.entries(sample.images || {})) {
    options.push([name, path]);
  }
  if (sample.raw && !options.some(([, path]) => path === sample.raw)) options.push(["raw", sample.raw]);
  if (sample.mask && !options.some(([, path]) => path === sample.mask)) options.push(["mask", sample.mask]);
  if (sample.overlay && !options.some(([, path]) => path === sample.overlay)) options.push(["overlay", sample.overlay]);
  for (const [name, path] of Object.entries(sample.variants || {})) {
    if (!options.some(([, existingPath]) => existingPath === path)) options.push([name, path]);
  }
  options.sort((a, b) => sourcePriority(a[0]) - sourcePriority(b[0]) || a[0].localeCompare(b[0]));
  for (const [name, path] of options) {
    const option = document.createElement("option");
    option.value = path;
    option.textContent = name;
    option.dataset.role = name;
    sourceSelect.appendChild(option);
  }
  if (previous && options.some(([, path]) => path === previous)) {
    sourceSelect.value = previous;
  }
}

function sourcePriority(role) {
  const order = {
    "dataset/raw": 0,
    "annotation image": 1,
    "validation/test image": 2,
    "train image": 3,
    "incoming raw/done": 4,
    "new_dataset raw/done": 5,
    "sdk asset": 6,
    "mask": 20,
    "validation/test mask": 21,
    "train mask": 22,
    "baseline edge/mask": 23,
    "overlay": 40,
    "incoming preview": 41,
    "new_dataset preview": 42,
  };
  return order[role] ?? 30;
}

function updateImages() {
  const path = sourceSelect.value;
  if (!path) return;
  sourceImage.src = imageUrl(path, "original");
  const asUploaded = ocrInputSelect.value === "as_is";
  const rotation = previewRotation();
  const maskOverlay = filterSelect.value === "mask_overlay";
  if (maskOverlay) {
    const mask = pairedMaskForSelectedSource();
    filterImage.src = mask ? maskOverlayUrl(path, mask) : imageUrl(path, "original");
  } else if (!asUploaded && state.selected?.annotation) {
    filterImage.src = sdkCropUrl(cropBaseImagePath(), state.selected.annotation, filterSelect.value, rotation.degrees);
  } else {
    filterImage.src = imageUrl(path, filterSelect.value, rotation.degrees);
  }
  const base = "Preview filter";
  filterHeading.textContent = maskOverlay ? base : `${base} · ${rotation.label}`;
}

function previewKey() {
  return [state.selected?.id, sourceSelect.value, ocrInputSelect.value, filterSelect.value].join("|");
}

function previewRotation() {
  const value = ocrRotationSelect.value;
  if (value !== "auto") {
    return { degrees: Number(value), label: `rotated ${value}°` };
  }
  if (state.autoRotation?.key === previewKey()) {
    return { degrees: state.autoRotation.degrees, label: `auto → ${state.autoRotation.degrees}° (last OCR)` };
  }
  return { degrees: 0, label: "auto pending" };
}

function imageUrl(path, mode, rotation = 0) {
  return `/image?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(mode)}&rotate=${rotation}&t=${Date.now()}`;
}

function maskOverlayUrl(imagePath, maskPath) {
  const query = new URLSearchParams({ image: imagePath, mask: maskPath, t: String(Date.now()) });
  return `/mask-overlay?${query.toString()}`;
}

function sdkCropUrl(imagePath, annotationPath, mode, rotation = 0) {
  const query = new URLSearchParams({
    image: imagePath,
    annotation: annotationPath,
    mode,
    rotate: String(rotation),
    t: String(Date.now()),
  });
  return `/sdk-crop?${query.toString()}`;
}

function selectedSourceRole() {
  return sourceSelect.selectedOptions[0]?.dataset.role || "";
}

function cropBaseImagePath() {
  const selectedRole = selectedSourceRole();
  if (canUseAsCropBase(selectedRole)) return sourceSelect.value;
  const images = state.selected?.images || {};
  return images["dataset/raw"]
    || images["validation/test image"]
    || images["train image"]
    || images["annotation image"]
    || state.selected?.raw
    || sourceSelect.value;
}

function canUseAsCropBase(role) {
  return [
    "annotation image",
    "dataset/raw",
    "train image",
    "validation/test image",
    "incoming raw/done",
    "new_dataset raw/done",
    "sdk asset",
  ].includes(role);
}

function pairedMaskForSelectedSource() {
  const images = state.selected?.images || {};
  const role = selectedSourceRole();
  if (role === "validation/test image") return images["validation/test mask"] || images.mask;
  if (role === "train image") return images["train mask"] || images.mask;
  if (role === "composite image") return images["composite mask"];
  if (role === "dataset/raw" || role === "annotation image") return images.mask || state.selected?.mask;
  return null;
}

function drawOverlay() {
  const ctx = overlayCanvas.getContext("2d");
  const width = sourceImage.clientWidth;
  const height = sourceImage.clientHeight;
  overlayCanvas.width = Math.max(1, Math.round(width));
  overlayCanvas.height = Math.max(1, Math.round(height));
  overlayCanvas.style.width = `${overlayCanvas.width}px`;
  overlayCanvas.style.height = `${overlayCanvas.height}px`;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!state.overlay || !state.annotation?.shapes?.length || !canDrawAnnotationOnSelectedSource()) return;

  const naturalWidth = state.annotation.imageWidth || sourceImage.naturalWidth;
  const naturalHeight = state.annotation.imageHeight || sourceImage.naturalHeight;
  const scaleX = overlayCanvas.width / naturalWidth;
  const scaleY = overlayCanvas.height / naturalHeight;

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#10b981";
  ctx.fillStyle = "rgba(16, 185, 129, 0.16)";
  for (const shape of state.annotation.shapes) {
    if (!shape.points?.length) continue;
    ctx.beginPath();
    shape.points.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x * scaleX, y * scaleY);
      else ctx.lineTo(x * scaleX, y * scaleY);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

function canDrawAnnotationOnSelectedSource() {
  const role = selectedSourceRole();
  return canUseAsCropBase(role);
}

sourceImage.addEventListener("load", drawOverlay);
sourceSelect.addEventListener("change", updateImages);
filterSelect.addEventListener("change", updateImages);
search.addEventListener("input", renderList);
window.addEventListener("resize", drawOverlay);
ocrEngineSelect.addEventListener("change", renderOcrStatus);
ocrModelSelect.addEventListener("change", renderOcrStatus);
ocrLanguageSelect.addEventListener("change", renderOcrStatus);
ocrInputSelect.addEventListener("change", updateImages);
ocrRotationSelect.addEventListener("change", updateImages);
runOcrButton.addEventListener("click", runOcr);
uploadButton.addEventListener("click", () => uploadInput.click());
ocrPickFileButton.addEventListener("click", () => uploadInput.click());
uploadInput.addEventListener("change", () => uploadFiles([...uploadInput.files]));
labelSave.addEventListener("click", saveLabeledLines);
labelSelectAll.addEventListener("click", () => {
  const boxes = [...labelRows.querySelectorAll("input[type=checkbox]")];
  const next = !boxes.every((box) => box.checked);
  boxes.forEach((box) => { box.checked = next; });
});
for (const eventName of ["dragenter", "dragover"]) {
  sidebar.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  sidebar.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragging");
  });
}
sidebar.addEventListener("drop", (event) => uploadFiles([...(event.dataTransfer?.files || [])]));

toggleOverlay.addEventListener("click", () => {
  state.overlay = !state.overlay;
  toggleOverlay.setAttribute("aria-pressed", String(state.overlay));
  drawOverlay();
});

fitMode.addEventListener("click", () => {
  state.fit = !state.fit;
  fitMode.setAttribute("aria-pressed", String(state.fit));
  for (const stage of document.querySelectorAll(".image-stage")) {
    stage.classList.toggle("original-size", !state.fit);
  }
  drawOverlay();
});

boot();

async function uploadFiles(files) {
  if (!files.length) return;
  let lastSampleId = null;
  for (const [index, file] of files.entries()) {
    uploadStatus.textContent = `Uploading ${index + 1}/${files.length}: ${file.name}…`;
    ocrFileInfo.textContent = uploadStatus.textContent;
    try {
      const response = await fetch("/api/uploads", {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
        body: file,
      });
      const result = await response.json();
      if (!result.ok) {
        uploadStatus.textContent = `${file.name}: ${result.error}`;
        continue;
      }
      lastSampleId = result.sample_id;
      uploadStatus.textContent = `Uploaded ${file.name} · ${result.upload.pages.length} page(s)`;
    } catch (error) {
      uploadStatus.textContent = `${file.name}: upload failed (${error})`;
    }
  }
  uploadInput.value = "";
  search.value = "";
  await loadSamples();
  if (lastSampleId) selectSample(lastSampleId);
}

async function deleteUpload(sample) {
  if (!window.confirm(`Delete upload "${sample.filename}" from benchmark/uploads?`)) return;
  const response = await fetch(`/api/uploads/${encodeURIComponent(sample.upload_id)}`, { method: "DELETE" });
  const result = await response.json();
  if (!result.ok) {
    uploadStatus.textContent = result.error || "Delete failed";
    return;
  }
  uploadStatus.textContent = `Deleted ${sample.filename}`;
  const wasSelected = state.selected?.id === sample.id;
  await loadSamples();
  if (wasSelected && state.samples.length) selectSample(state.samples[0].id);
}

async function runOcr() {
  const asUploaded = ocrInputSelect.value === "as_is";
  const preprocessed = window.currentPreprocessedInput?.() || null;
  const imagePath = preprocessed?.path || (!asUploaded && state.selected?.annotation ? cropBaseImagePath() : sourceSelect.value);
  if (!imagePath) {
    setOcrMessage("No image is selected.");
    return;
  }
  const mode = preprocessed ? "original" : (filterSelect.value === "mask_overlay" ? "enhance" : filterSelect.value);
  const inputMode = preprocessed ? "as_is" : ocrInputSelect.value;
  const annotationPath = preprocessed ? null : (asUploaded ? null : (state.selected?.annotation || null));
  const sourceNote = preprocessed ? ` on ${preprocessed.sourceName}` : " on filtered preview";
  const ocrRotation = preprocessed ? "0" : ocrRotationSelect.value;
  setOcrMessage(ocrEngineSelect.value === "easyocr" ? `Running EasyOCR${sourceNote}… first use may download/load models; later runs reuse them.` : `Running OCR${sourceNote}...`);
  runOcrButton.disabled = true;
  try {
    const response = await fetch("/api/ocr/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: ocrEngineSelect.value,
        language: ocrLanguageSelect.value,
        rotation: ocrRotation,
        model: ocrModelSelect.disabled || ocrEngineSelect.value === "tesseract" ? null : ocrModelSelect.value,
        tesseract_variant: ocrEngineSelect.value === "tesseract" ? ocrModelSelect.value : null,
        psm: ocrPsmSelect.value,
        strategy: ocrStrategySelect.value,
        input: inputMode,
        mode,
        image_path: imagePath,
        annotation_path: annotationPath,
        sample_id: state.selected?.id || null,
        preprocessed_id: preprocessed?.preprocessId || null,
      }),
    });
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error(`OCR route returned non-JSON (HTTP ${response.status}). Restart the dashboard.`);
    }
    const result = await response.json();
    if (!result.ok) {
      setOcrMessage(result.error || "OCR failed.");
      return;
    }
    renderOcrResult(result);
    if (!preprocessed && ocrRotationSelect.value === "auto" && Number.isInteger(result.rotation)) {
      state.autoRotation = { key: previewKey(), degrees: result.rotation };
      updateImages();
    }
    if (ocrEngineSelect.value === "tesseract" && window.runOcrSteps) {
      await window.runOcrSteps({ rotation: String(result.rotation ?? ocrRotation) });
    } else if (window.clearOcrSteps) {
      window.clearOcrSteps(`${selectedOcrEngine()?.label || "This engine"} does not expose Tesseract OCR steps.`);
    }
  } catch (error) {
    setOcrMessage(`OCR request failed: ${error}`);
  } finally {
    renderOcrStatus();
  }
}

function setOcrMessage(message) {
  labelPanel.hidden = true;
  labelRows.innerHTML = "";
  ocrSummaryOutput.innerHTML = `<span class="chip muted">${escapeHtml(message)}</span>`;
  parsedFieldsOutput.innerHTML = "";
  rawTextOutput.textContent = "";
  ocrJsonOutput.textContent = "{}";
  engineRawOutput.textContent = "{}";
}

function renderOcrResult(result) {
  ocrSummaryOutput.innerHTML = formatOcrSummary(result);
  parsedFieldsOutput.innerHTML = formatParsedFields(result.parsed || {});
  rawTextOutput.textContent = result.ocr?.full_text || result.parsed?.raw_text || "";
  ocrJsonOutput.textContent = JSON.stringify(result.ocr || {}, null, 2);
  engineRawOutput.textContent = formatRawEngineResponse(result.ocr);
  renderLabelPanel(result);
}

function renderLabelPanel(result) {
  const blocks = result.ocr?.blocks || [];
  labelPanel.hidden = !(result.run_id && blocks.length);
  labelRows.innerHTML = "";
  labelStatus.textContent = "";
  if (labelPanel.hidden) return;
  labelPanel.dataset.runId = result.run_id;
  blocks.forEach((block, index) => {
    if (!block.bounding_box) return;
    const box = block.bounding_box;
    const query = new URLSearchParams({ x: box.x, y: box.y, width: box.width, height: box.height });
    const row = document.createElement("div");
    row.className = "label-row";
    row.dataset.index = String(index);
    row.dataset.ocrText = block.text;
    row.dataset.bbox = JSON.stringify(box);
    row.innerHTML = `
      <input type="checkbox" title="Include this line">
      <img alt="line ${index + 1}" loading="lazy" src="/api/runs/${encodeURIComponent(result.run_id)}/crop?${query}">
      <input type="text" dir="auto" spellcheck="false">
      <select title="Script">
        <option value="ben">Bengali</option>
        <option value="eng">English</option>
        <option value="digits">Digits</option>
      </select>
      <span class="conf">${block.confidence == null ? "–" : Math.round(block.confidence * 100) + "%"}</span>`;
    const text = row.querySelector("input[type=text]");
    const checkbox = row.querySelector("input[type=checkbox]");
    text.value = block.text;
    row.querySelector("select").value = guessScript(block.text);
    text.addEventListener("input", () => {
      row.classList.toggle("edited", text.value !== block.text);
      checkbox.checked = true;
    });
    labelRows.appendChild(row);
  });
  refreshDatasetStatus();
}

function guessScript(text) {
  const bengali = (text.match(/[\u0980-\u09FF]/g) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  if (bengali && bengali >= latin) return "ben";
  if (latin) return "eng";
  return /\d/.test(text) ? "digits" : "eng";
}

async function refreshDatasetStatus(prefix = "") {
  try {
    const stats = await (await fetch("/api/training/datasets")).json();
    const current = stats.find((item) => item.dataset === labelDataset.value);
    const summary = current
      ? `${current.dataset}: ${current.lines} lines from ${current.cards} card(s) · ${Object.entries(current.scripts).map(([k, v]) => `${k} ${v}`).join(", ")}`
      : `${labelDataset.value}: no lines saved yet`;
    labelStatus.textContent = `${prefix}${summary}. Tick only lines whose text you verified exactly.`;
  } catch (error) {
    labelStatus.textContent = `${prefix}Could not load dataset stats (${error})`;
  }
}

async function saveLabeledLines() {
  const rows = [...labelRows.querySelectorAll(".label-row")].filter((row) => row.querySelector("input[type=checkbox]").checked);
  if (!rows.length) {
    labelStatus.textContent = "Tick at least one verified line.";
    return;
  }
  const lines = rows.map((row) => ({
    index: Number(row.dataset.index),
    text: row.querySelector("input[type=text]").value,
    ocr_text: row.dataset.ocrText,
    script: row.querySelector("select").value,
    bounding_box: JSON.parse(row.dataset.bbox),
  }));
  labelSave.disabled = true;
  try {
    const response = await fetch("/api/training/lines", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: labelDataset.value,
        run_id: labelPanel.dataset.runId,
        card_id: labelCardId.value,
        lines,
      }),
    });
    const result = await response.json();
    if (!result.ok) {
      labelStatus.textContent = result.error || "Save failed";
      return;
    }
    rows.forEach((row) => { row.querySelector("input[type=checkbox]").checked = false; });
    await refreshDatasetStatus(`Saved ${result.saved.length} line(s). `);
  } finally {
    labelSave.disabled = false;
  }
}

function formatRawEngineResponse(ocr) {
  const metadata = ocr?.metadata || {};
  const raw = metadata.raw_response ?? metadata.raw_content ?? ocr?.full_text ?? "";
  if (typeof raw === "string") {
    return raw;
  }
  return JSON.stringify(raw, null, 2);
}

function formatOcrSummary(result) {
  const candidates = (result.rotation_candidates || [])
    .map((candidate) => {
      const psm = candidate.psm === null || candidate.psm === undefined ? "" : ` psm ${candidate.psm}`;
      const kind = candidate.score_kind ? " (heuristic)" : "";
      return `rot ${candidate.rotation}${psm}: ${candidate.score}${kind}`;
    })
    .join(" · ");
  const chips = [
    chip(`Image ${result.image?.width} x ${result.image?.height}`),
    chip(`Source ${result.image?.source_name || "unknown"}`, "wide", result.image?.source_path || ""),
    chip(result.image?.mode || ""),
    chip(result.image?.input_mode || ""),
    chip(`Rotation ${result.rotation}`),
  ];
  if (result.psm !== null && result.psm !== undefined) {
    chips.push(chip(`PSM ${result.psm}`));
  }
  const meta = result.ocr?.metadata || {};
  if (result.ocr?.engine === "tesseract") {
    chips.push(chip(`Model ${meta.variant}`));
    const versions = (meta.models || []).map((item) => `${item.language}: ${item.version || "?"}`).join(" · ");
    if (versions) chips.push(chip(versions, "wide"));
    chips.push(chip(`${result.ocr_calls} OCR call(s)`));
    if (meta.dpi) chips.push(chip(`DPI ${meta.dpi} (${meta.dpi_source})`));
    const orientation = result.orientation || {};
    const orientationText = orientation.method === "osd"
      ? `OSD rotate ${orientation.rotate} (conf ${orientation.confidence})`
      : orientation.method === "rotation-check"
        ? `OSD unsure → 4-way check ${JSON.stringify(orientation.scores)}`
        : "manual rotation";
    chips.push(chip(orientationText, "wide"));
  }
  const model = result.ocr?.engine === "tesseract" ? null : result.ocr?.metadata?.model;
  chips.push(
    chip(`${Math.round(result.request_latency_ms ?? result.ocr?.latency_ms ?? 0)} ms total`),
    chip(result.ocr?.engine || ""),
    chip(result.ocr?.language || ""),
  );
  if (model) {
    chips.push(chip(`Model ${model}`));
  }
  if (result.ocr?.metadata?.rotation_strategy) {
    chips.push(chip(result.ocr.metadata.rotation_strategy, "wide"));
    chips.push(chip(result.ocr.metadata.reader_cached ? "Reader reused" : "Reader initialized"));
  }
  chips.push(chip(candidates || (result.ocr?.engine === "easyocr" ? "Single detection pass" : "No sweep candidates"), "wide"));
  return chips.join("");
}

function formatParsedFields(parsed) {
  const rows = [];
  for (const key of Object.keys(parsed)) {
    if (key === "raw_text") continue;
    const value = parsed[key]?.value ?? "";
    const empty = value === "";
    rows.push(`
      <div class="field-row ${empty ? "empty" : ""}">
        <span>${escapeHtml(labelForField(key))}</span>
        <strong>${escapeHtml(value || "Not found")}</strong>
      </div>
    `);
  }
  return rows.join("");
}

function labelForField(key) {
  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function chip(value, extraClass = "", title = "") {
  const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
  return `<span class="chip ${extraClass}"${titleAttr}>${escapeHtml(String(value || ""))}</span>`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

labelDataset.addEventListener("change", () => refreshDatasetStatus());
