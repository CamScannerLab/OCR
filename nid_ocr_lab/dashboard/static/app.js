const state = {
  samples: [],
  filtered: [],
  selected: null,
  annotation: null,
  health: null,
  ocrStatus: null,
  overlay: true,
  fit: true,
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

async function boot() {
  const response = await fetch("/api/samples");
  state.samples = await response.json();
  const healthResponse = await fetch("/api/health");
  state.health = await healthResponse.json();
  const ocrStatusResponse = await fetch("/api/ocr/status");
  state.ocrStatus = await ocrStatusResponse.json();
  state.filtered = state.samples;
  sampleCount.textContent = state.samples.length;
  healthDetails.textContent = JSON.stringify(state.health, null, 2);
  renderOcrStatus();
  renderList();
  if (state.samples.length) {
    selectSample(state.samples[0].id);
  }
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
  runOcrButton.disabled = false;
}

function selectedOcrEngine() {
  return state.ocrStatus?.engines?.find((item) => item.id === ocrEngineSelect.value);
}

function isVisionEngine(engineId) {
  return engineId === "gemini_vision" || engineId === "lmstudio_vision";
}

function renderModelOptions(engine) {
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
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample-item ${state.selected?.id === sample.id ? "active" : ""}`;
    button.innerHTML = `<strong>${sample.id}</strong><span>${sample.source} · ${badges(sample)}</span>`;
    button.addEventListener("click", () => selectSample(sample.id));
    sampleList.appendChild(button);
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
  sampleTitle.textContent = `Sample ${sample.id}`;
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
  if (filterSelect.value === "mask_overlay") {
    const mask = pairedMaskForSelectedSource();
    filterImage.src = mask ? maskOverlayUrl(path, mask) : imageUrl(path, "original");
  } else if (state.selected?.annotation) {
    filterImage.src = sdkCropUrl(cropBaseImagePath(), state.selected.annotation, filterSelect.value);
  } else {
    filterImage.src = imageUrl(path, filterSelect.value);
  }
}

function imageUrl(path, mode) {
  return `/image?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(mode)}&t=${Date.now()}`;
}

function maskOverlayUrl(imagePath, maskPath) {
  const query = new URLSearchParams({ image: imagePath, mask: maskPath, t: String(Date.now()) });
  return `/mask-overlay?${query.toString()}`;
}

function sdkCropUrl(imagePath, annotationPath, mode) {
  const query = new URLSearchParams({
    image: imagePath,
    annotation: annotationPath,
    mode,
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
runOcrButton.addEventListener("click", runOcr);

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

async function runOcr() {
  const imagePath = state.selected?.annotation ? cropBaseImagePath() : sourceSelect.value;
  if (!imagePath) {
    setOcrMessage("No image is selected.");
    return;
  }
  const mode = filterSelect.value === "mask_overlay" ? "enhance" : filterSelect.value;
  setOcrMessage("Running OCR...");
  runOcrButton.disabled = true;
  try {
    const response = await fetch("/api/ocr/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: ocrEngineSelect.value,
        language: ocrLanguageSelect.value,
        rotation: ocrRotationSelect.value,
        model: ocrModelSelect.disabled ? null : ocrModelSelect.value,
        mode,
        image_path: imagePath,
        annotation_path: state.selected?.annotation || null,
        sample_id: state.selected.id,
      }),
    });
    const result = await response.json();
    if (!result.ok) {
      setOcrMessage(result.error || "OCR failed.");
      return;
    }
    renderOcrResult(result);
  } catch (error) {
    setOcrMessage(`OCR request failed: ${error}`);
  } finally {
    renderOcrStatus();
  }
}

function setOcrMessage(message) {
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
      return `rot ${candidate.rotation}${psm}: ${candidate.score}`;
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
  const model = result.ocr?.metadata?.model;
  chips.push(
    chip(`${Math.round(result.ocr?.latency_ms || 0)} ms`),
    chip(result.ocr?.engine || ""),
    chip(result.ocr?.language || ""),
  );
  if (model) {
    chips.push(chip(`Model ${model}`));
  }
  chips.push(chip(candidates || "No rotation candidates", "wide"));
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
