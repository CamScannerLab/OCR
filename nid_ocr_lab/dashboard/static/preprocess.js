const preprocessState = { result: null, request: 0, open: {} };

const preprocessRecipe = document.querySelector("#preprocessRecipe");
const preprocessUseForSteps = document.querySelector("#preprocessUseForSteps");
const preprocessRun = document.querySelector("#preprocessRun");
const preprocessStatus = document.querySelector("#preprocessStatus");
const preprocessSummary = document.querySelector("#preprocessSummary");
const preprocessTimeline = document.querySelector("#preprocessTimeline");

function preprocessKey() {
  return [
    state.selected?.id || "",
    sourceSelect.value || "",
    ocrInputSelect.value || "",
    filterSelect.value || "",
    previewRotation().degrees,
    state.selected?.annotation || "",
  ].join("|");
}

function preprocessUrl(name) {
  return `/api/preprocess/${preprocessState.result.preprocess_id}/${name}`;
}

function selectedPreprocessInput() {
  if (!preprocessUseForSteps.checked || !preprocessState.result) return null;
  if (preprocessState.result.selection_key !== preprocessKey()) return null;
  const finalImage = preprocessState.result.final_image;
  if (!finalImage?.path) return null;
  return {
    path: finalImage.path,
    sourceName: `${preprocessState.result.recipe.label} final`,
    preprocessId: preprocessState.result.preprocess_id,
  };
}

window.currentPreprocessedInput = selectedPreprocessInput;

async function runPreprocessing() {
  const asUploaded = ocrInputSelect.value === "as_is";
  const imagePath = !asUploaded && state.selected?.annotation ? cropBaseImagePath() : sourceSelect.value;
  if (!imagePath) {
    preprocessStatus.textContent = "No image is selected.";
    return;
  }
  const request = ++preprocessState.request;
  const selectionKey = preprocessKey();
  const started = performance.now();
  preprocessRun.disabled = true;
  preprocessStatus.classList.remove("failed");
  preprocessStatus.textContent = "Running local preprocessing…";
  try {
    const response = await fetch("/api/preprocess/steps", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipe: preprocessRecipe.value,
        input: ocrInputSelect.value,
        mode: filterSelect.value === "mask_overlay" ? "enhance" : filterSelect.value,
        image_path: imagePath,
        annotation_path: asUploaded ? null : (state.selected?.annotation || null),
        prepared_rotation: previewRotation().degrees,
      }),
    });
    const result = await response.json();
    if (request !== preprocessState.request) return;
    if (!result.ok) {
      failPreprocess(result.error || "Preprocessing failed.");
      return;
    }
    preprocessState.result = { ...result, selection_key: selectionKey };
    preprocessTimeline.classList.remove("stale");
    preprocessStatus.textContent = `Ready for Run OCR · ${Math.round(performance.now() - started)} ms · ${result.recipe.label} · ${result.image.input_mode}/${result.image.filter_mode} · preview rotated ${previewRotation().degrees}°`;
    renderPreprocessing();
  } catch (error) {
    if (request === preprocessState.request) failPreprocess(`Preprocessing request failed: ${error.message || error}`);
  } finally {
    if (request === preprocessState.request) preprocessRun.disabled = false;
  }
}

function failPreprocess(message) {
  preprocessStatus.textContent = preprocessState.result ? `${message} — the steps below are still the previous run.` : message;
  preprocessStatus.classList.add("failed");
  preprocessTimeline.classList.toggle("stale", Boolean(preprocessState.result));
}

function markPreprocessStale() {
  if (!preprocessState.result) return;
  if (preprocessState.result.selection_key === preprocessKey()) return;
  preprocessTimeline.classList.add("stale");
  preprocessStatus.textContent = "Selection changed — run preprocessing again before Run OCR.";
}

function renderPreprocessing() {
  const result = preprocessState.result;
  const chips = [
    `recipe: ${result.recipe.label}`,
    `${result.image.width} x ${result.image.height}`,
    `${result.image.input_mode}/${result.image.filter_mode}`,
    `preprocess ${Math.round(result.timings.preprocess_ms)} ms`,
    `total ${Math.round(result.request_latency_ms)} ms`,
  ];
  preprocessSummary.innerHTML = chips.map((text) => `<span class="chip">${escapeHtml(text)}</span>`).join("");
  preprocessTimeline.innerHTML = "";
  result.steps.forEach((step, index) => {
    const details = document.createElement("details");
    details.className = `step step-${step.id}`;
    details.open = preprocessState.open[step.id] ?? false;
    details.addEventListener("toggle", () => {
      preprocessState.open[step.id] = details.open;
    });
    details.innerHTML = `
      <summary><span class="step-no">${index + 1}</span><span class="step-title">${escapeHtml(step.title)}</span><span class="step-key">${escapeHtml(preprocessStepKey(step))}</span></summary>
      <div class="step-body">
        <p class="step-explain">${escapeHtml(step.explain)}</p>
      </div>`;
    const body = details.querySelector(".step-body");
    const table = preprocessValuesTable(step.values || {});
    if (table) body.append(table);
    if (Object.keys(step.images || {}).length) body.append(preprocessFigureGrid(step));
    preprocessTimeline.append(details);
  });
}

function preprocessStepKey(step) {
  const values = step.values || {};
  const bits = [];
  if (values.width && values.height) bits.push(`${values.width}×${values.height}`);
  if (values.ink_share != null) bits.push(`ink ${(values.ink_share * 100).toFixed(1)}%`);
  if (values.median != null) bits.push(`median ${values.median}`);
  if (values.p05 != null && values.p95 != null) bits.push(`p05/p95 ${values.p05}/${values.p95}`);
  return bits.join(" · ");
}

function preprocessValuesTable(values) {
  const rows = Object.entries(values).filter(([, value]) => value !== null && typeof value !== "object");
  if (!rows.length) return null;
  const table = document.createElement("dl");
  table.className = "step-values";
  table.innerHTML = rows
    .map(([key, value]) => `<div><dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd>${escapeHtml(String(value))}</dd></div>`)
    .join("");
  return table;
}

function preprocessFigureGrid(step) {
  const grid = document.createElement("div");
  grid.className = "step-figures";
  for (const [key, name] of Object.entries(step.images)) {
    const figure = document.createElement("figure");
    const link = document.createElement("a");
    link.href = preprocessUrl(name);
    link.target = "_blank";
    link.rel = "noopener";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = preprocessUrl(name);
    img.alt = key;
    if (key.includes("mask")) img.classList.add("pixelated");
    link.append(img);
    const caption = document.createElement("figcaption");
    caption.textContent = key.replaceAll("_", " ");
    figure.append(link, caption);
    grid.append(figure);
  }
  return grid;
}

preprocessRun.addEventListener("click", runPreprocessing);
preprocessRecipe.addEventListener("change", () => {
  if (preprocessState.result) runPreprocessing();
});
for (const input of [sourceSelect, filterSelect, ocrInputSelect, ocrRotationSelect]) {
  input.addEventListener("change", markPreprocessStale);
}

