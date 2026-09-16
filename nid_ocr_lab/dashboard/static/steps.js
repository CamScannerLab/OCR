// Tesseract step inspector. Reuses the selections and helpers defined in app.js.
const stepsState = { result: null, request: 0, layers: { blobs: false, blocks: true, paragraphs: false, lines: true, words: false, symbols: false }, base: "binary", open: {}, canvases: [], baseImages: null };

const stepsFilter = document.querySelector("#stepsFilter");
const stepsMethod = document.querySelector("#stepsMethod");
const stepsInputs = {
  tile_size: document.querySelector("#stepsTileSize"),
  smooth_kernel_size: document.querySelector("#stepsSmooth"),
  score_fraction: document.querySelector("#stepsScore"),
  window_size: document.querySelector("#stepsWindow"),
  kfactor: document.querySelector("#stepsK"),
};
const stepsValue = document.querySelector("#stepsValue");
const stepsValueNumber = document.querySelector("#stepsValueNumber");
const stepsValueOut = document.querySelector("#stepsValueOut");
const stepsDpi = document.querySelector("#stepsDpi");
const stepsLayoutOnly = document.querySelector("#stepsLayoutOnly");
const stepsDebug = document.querySelector("#stepsDebug");
const stepsFields = document.querySelector("#stepsFields");
const stepsRun = document.querySelector("#stepsRun");
const stepsStatus = document.querySelector("#stepsStatus");
const stepsSummary = document.querySelector("#stepsSummary");
const stepsTimeline = document.querySelector("#stepsTimeline");

const LAYER_COLORS = {
  blobs: "#8a8f8b",
  blocks: "#d97706",
  paragraphs: "#7c3aed",
  lines: "#0f766e",
  words: "#2563eb",
  symbols: "#db2777",
};
const HIDDEN_VALUES = new Set(["histogram", "tile_thresholds", "channels", "log", "text", "fields", "page_fields", "baseline_slope_degrees"]);

function stepsUrl(name) {
  return `/api/steps/${stepsState.result.steps_id}/${name}`;
}

function stepsThreshold() {
  const threshold = { method: stepsMethod.value, value: Number(stepsValue.value) };
  for (const [key, input] of Object.entries(stepsInputs)) threshold[key] = input.value === "" ? null : Number(input.value);
  return threshold;
}

function syncFilterSelect() {
  if (stepsFilter.options.length !== filterSelect.options.length) {
    stepsFilter.innerHTML = filterSelect.innerHTML;
  }
  if (!filterSelect.value) filterSelect.value = filterSelect.options[0]?.value || "original";
  stepsFilter.value = filterSelect.value;
}

function syncMethodInputs() {
  for (const label of document.querySelectorAll(".steps-controls [data-method]")) {
    label.hidden = label.dataset.method !== stepsMethod.value;
  }
}

let stepsTimer = null;
function scheduleStepsRun() {
  if (!stepsState.result) return;
  clearTimeout(stepsTimer);
  stepsTimer = setTimeout(runSteps, 300);
}

async function runSteps(options = {}) {
  const asUploaded = ocrInputSelect.value === "as_is";
  const preprocessed = window.currentPreprocessedInput?.() || null;
  const imagePath = preprocessed?.path || (!asUploaded && state.selected?.annotation ? cropBaseImagePath() : sourceSelect.value);
  const stepRotation = options.rotation ?? (preprocessed ? "0" : ocrRotationSelect.value);
  if (!imagePath) {
    stepsStatus.textContent = "No image is selected.";
    return;
  }
  const request = ++stepsState.request;
  const started = performance.now();
  if (stepsRun) stepsRun.disabled = true;
  stepsStatus.textContent = stepsLayoutOnly.checked ? "Run OCR: calculating threshold and layout steps…" : "Run OCR: calculating threshold, layout and recognition steps…";
  try {
    const response = await fetch("/api/tesseract/steps", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_path: imagePath,
        annotation_path: preprocessed ? null : (asUploaded ? null : (state.selected?.annotation || null)),
        input: preprocessed ? "as_is" : ocrInputSelect.value,
        mode: preprocessed ? "original" : (stepsFilter.value === "mask_overlay" ? "enhance" : stepsFilter.value),
        preprocessed_id: preprocessed?.preprocessId || null,
        rotation: stepRotation,
        language: ocrLanguageSelect.value,
        tesseract_variant: ocrEngineSelect.value === "tesseract" ? ocrModelSelect.value : "system",
        psm: ocrPsmSelect.value,
        dpi: stepsDpi.value || null,
        threshold: stepsThreshold(),
        layout_only: stepsLayoutOnly.checked,
        debug_images: stepsDebug.checked,
        fields: stepsFields.checked,
      }),
    });
    if (!response.headers.get("content-type")?.includes("application/json")) {
      // The dashboard serves static files from disk but keeps its Python routes in memory.
      throw new Error(`the server has no /api/tesseract/steps route (HTTP ${response.status}). Restart the dashboard to load the new server code.`);
    }
    const result = await response.json();
    if (request !== stepsState.request) return; // a newer run started while this one was busy
    if (!result.ok) {
      failSteps(result.error || "Step inspection failed.");
      return;
    }
    stepsState.result = result;
    stepsTimeline.classList.remove("stale");
    stepsStatus.classList.remove("failed");
    const source = preprocessed ? `${preprocessed.sourceName} · ` : "";
    stepsStatus.textContent = `Done in ${Math.round(performance.now() - started)} ms · ${source}${result.image.source_name} · ${result.image.input_mode}/${result.image.filter_mode} · rotated ${result.image.rotation}°`;
    renderSteps();
  } catch (error) {
    if (request === stepsState.request) failSteps(`Step request failed: ${error.message || error}`);
  } finally {
    if (request === stepsState.request && stepsRun) stepsRun.disabled = false;
  }
}

function failSteps(message) {
  // The old timeline stays on screen, so say plainly that it is not the run you just asked for.
  stepsStatus.textContent = stepsState.result ? `${message} — the steps below are still the previous run.` : message;
  stepsStatus.classList.add("failed");
  stepsTimeline.classList.toggle("stale", Boolean(stepsState.result));
}

function renderSteps() {
  const result = stepsState.result;
  const settings = result.settings;
  const threshold = settings.threshold;
  const chips = [
    `Tesseract ${result.tesseract_version}`,
    `${settings.language} · ${settings.variant}`,
    `PSM ${settings.psm}`,
    `threshold: ${threshold.method}`,
    ...Object.entries(result.timings).map(([key, value]) => `${key.replace(/_ms$/, "").replaceAll("_", " ")} ${Math.round(value)} ms`),
  ];
  stepsSummary.innerHTML = chips.map((text) => `<span class="chip">${escapeHtml(text)}</span>`).join("");

  stepsTimeline.innerHTML = "";
  stepsResize.disconnect();
  stepsState.canvases = [];
  stepsState.baseImages = null;
  result.steps.forEach((step, index) => {
    const details = document.createElement("details");
    details.className = `step step-${step.id}`;
    details.open = stepsState.open[step.id] ?? false;
    details.addEventListener("toggle", () => {
      stepsState.open[step.id] = details.open;
      if (details.open) drawCanvases(); // a collapsed panel has no width to measure
    });
    details.innerHTML = `
      <summary><span class="step-no">${index + 1}</span><span class="step-title">${escapeHtml(step.title)}</span><span class="step-key">${escapeHtml(stepKeyLine(step))}</span></summary>
      <div class="step-body">
        <p class="step-explain">${escapeHtml(step.explain)}</p>
      </div>`;
    const body = details.querySelector(".step-body");
    body.append(...renderStepBody(step));
    stepsTimeline.appendChild(details);
    if (step.id === "words") stepsTimeline.appendChild(renderLayoutViewer());
  });
}

function stepKeyLine(step) {
  const v = step.values || {};
  switch (step.id) {
    case "input": return `${v.width}×${v.height} ${v.mode} · DPI ${v.dpi ?? "not set"} (${v.dpi_source})`;
    case "orientation": return `rotate ${v.rotate}° · ${v.decision}`;
    case "grey": return `${v.depth_bits}-bit input`;
    case "threshold": {
      const value = v.method === "adaptive_otsu" || v.method === "sauvola"
        ? `threshold map ${v.threshold_min}–${v.threshold_max} (median ${v.threshold_median})`
        : `threshold ${Array.isArray(v.threshold) ? v.threshold.join(" / ") : v.threshold}`;
      const match = v.replica_match_percent == null ? "" : ` · ${v.replica_match_percent === 100 ? "matches Tesseract ✓" : `matches ${v.replica_match_percent}%`}`;
      return `${v.method} · ${value} · ink ${(v.ink_share * 100).toFixed(1)}%${match}`;
    }
    case "blobs": return `${v.count} blobs · median height ${v.median_height ?? "–"} px`;
    case "blocks": return `${v.count} block(s) · ${Object.entries(v.types || {}).map(([k, n]) => `${n} ${k}`).join(", ")}`;
    case "paragraphs": return `${v.count} paragraph(s)`;
    case "lines": return `${v.count} line(s) · median height ${v.median_height ?? "–"} px · tallest ${v.max_height ?? "–"} px`;
    case "words": return `${v.words} words · ${v.symbols} symbols`;
    case "debug": return v.log ? `${Object.keys(step.images).length} image(s) · estimated resolution ${v.estimated_resolution ?? "–"}` : "not produced for this PSM";
    case "recognition": return v.mean_confidence == null ? `${v.lines} line(s) · skipped` : `${v.lines} line(s) · mean confidence ${v.mean_confidence}`;
    case "output": return `mean confidence ${v.mean_confidence} · ${filledCount(v.fields)}/6 fields`;
    case "fields": return `${filledCount(v.fields)}/6 fields (page pass: ${filledCount(v.page_fields)}/6) · ${v.calls} calls · ${v.elapsed_ms} ms · ${v.reader}`;
    default: return "";
  }
}

function filledCount(fields) {
  return Object.values(fields || {}).filter(Boolean).length;
}

function renderStepBody(step) {
  const nodes = [];
  const values = step.values || {};
  const table = valuesTable(values);
  if (table) nodes.push(table);
  if (step.id === "grey") nodes.push(histogramCanvas(values.histogram, stepsState.result.steps.find((item) => item.id === "threshold")?.values));
  if (step.id === "threshold" && values.channels) nodes.push(channelsTable(values.channels));
  if (step.id === "threshold") nodes.push(histogramCanvas(stepsState.result.steps.find((item) => item.id === "grey")?.values.histogram, values));
  if (Object.keys(step.images || {}).length) nodes.push(figureGrid(step));
  if (step.overlay) nodes.push(stepCanvas(step), layerToggles(step));
  if (step.id === "debug" && values.log?.length) nodes.push(preBlock(values.log.join("\n")));
  if (step.id === "recognition" && step.lines?.length) nodes.push(linesTable(step.lines));
  if (step.id === "fields") {
    nodes.push(fieldRowsTable(step));
    nodes.push(preBlock(step.values.text || ""));
  }
  if (step.id === "output" || step.id === "fields") {
    nodes.push(preBlock(values.text || ""));
    const grid = document.createElement("div");
    grid.className = "field-grid steps-fields";
    grid.innerHTML = Object.entries(values.fields || {}).map(([name, value]) =>
      `<div class="field-row ${value ? "" : "empty"}"><span>${escapeHtml(name)}</span><strong dir="auto">${escapeHtml(value ?? "Not found")}</strong></div>`).join("");
    nodes.push(grid);
  }
  return nodes;
}

function valuesTable(values) {
  const rows = Object.entries(values).filter(([key, value]) => !HIDDEN_VALUES.has(key) && value !== null && typeof value !== "object");
  const objects = Object.entries(values).filter(([key, value]) => !HIDDEN_VALUES.has(key) && value && typeof value === "object");
  if (!rows.length && !objects.length) return null;
  const table = document.createElement("dl");
  table.className = "step-values";
  table.innerHTML = [...rows, ...objects.map(([key, value]) => [key, JSON.stringify(value)])]
    .map(([key, value]) => `<div><dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join("");
  return table;
}

function channelsTable(channels) {
  const table = document.createElement("table");
  table.className = "steps-table";
  table.innerHTML = `<thead><tr><th>Channel</th><th>Otsu threshold</th><th>Used</th><th>Rule</th></tr></thead><tbody>${
    channels.map((item) => `<tr><td>${escapeHtml(item.channel)}</td><td>${item.threshold}</td><td>${item.used ? "yes" : "no"}</td><td>${escapeHtml(item.ink)}</td></tr>`).join("")
  }</tbody>`;
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);
  const note = document.createElement("p");
  note.className = "step-note";
  note.textContent = "A pixel is ink if any used channel satisfies its rule (Tesseract's global Otsu works per colour channel).";
  wrap.append(note);
  return wrap;
}

function histogramCanvas(histogram, threshold) {
  const figure = document.createElement("figure");
  figure.className = "histogram";
  const canvas = document.createElement("canvas");
  canvas.width = 768;
  canvas.height = 160;
  figure.append(canvas);
  const caption = document.createElement("figcaption");
  caption.textContent = threshold?.channels
    ? "Grey-level histogram (0 = black … 255 = white, log scale). Otsu thresholds are computed per colour channel; their markers are drawn here for reference."
    : "Grey-level histogram (0 = black … 255 = white, log scale) with the threshold used.";
  figure.append(caption);
  if (!histogram) return figure;
  const ctx = canvas.getContext("2d");
  const max = Math.log1p(Math.max(...histogram));
  const barWidth = canvas.width / 256;
  ctx.fillStyle = "#9aa39c";
  histogram.forEach((count, level) => {
    const height = (Math.log1p(count) / max) * (canvas.height - 18);
    ctx.fillRect(level * barWidth, canvas.height - height, Math.max(1, barWidth - 0.5), height);
  });
  let markers = 0;
  const marker = (level, color, label, dashed = false) => {
    const row = markers++;
    const x = (level + 0.5) * barWidth;
    ctx.strokeStyle = color;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 4);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = "12px system-ui, sans-serif";
    // Stack labels so close thresholds stay readable.
    ctx.fillText(label, Math.min(canvas.width - 90, x + 4), 14 + row * 15);
  };
  if (threshold?.channels) {
    const colors = { red: "#dc2626", green: "#16a34a", blue: "#2563eb", grey: "#111827", alpha: "#6b7280" };
    threshold.channels.filter((item) => item.used).forEach((item, i) => marker(item.threshold, colors[item.channel], `${item.channel} ${item.threshold}`, i > 0));
  } else if (typeof threshold?.threshold === "number") {
    marker(threshold.threshold, "#111827", `T = ${threshold.threshold}`);
  } else if (threshold?.threshold_median != null) {
    marker(threshold.threshold_min, "#2563eb", `min ${threshold.threshold_min}`, true);
    marker(threshold.threshold_median, "#111827", `median ${threshold.threshold_median}`);
    marker(threshold.threshold_max, "#dc2626", `max ${threshold.threshold_max}`, true);
  }
  return figure;
}

const IMAGE_CAPTIONS = {
  input: "Image handed to Tesseract",
  grey: "Greyscale (pixConvertTo8)",
  binary: "Binary image Tesseract uses (GetThresholdedImage)",
  threshold_map: "Threshold map (blue = low, red = high threshold)",
  difference: "Difference: red = only Tesseract ink, blue = only recomputed ink",
};

function figureGrid(step) {
  const grid = document.createElement("div");
  grid.className = "step-figures";
  for (const [key, name] of Object.entries(step.images)) {
    const figure = document.createElement("figure");
    const link = document.createElement("a");
    link.href = stepsUrl(name);
    link.target = "_blank";
    link.rel = "noopener";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = stepsUrl(name);
    img.alt = IMAGE_CAPTIONS[key] || key;
    if (key === "threshold_map") img.classList.add("pixelated");
    link.append(img);
    const caption = document.createElement("figcaption");
    caption.textContent = IMAGE_CAPTIONS[key] || key.replaceAll("_", " ");
    figure.append(link, caption);
    grid.append(figure);
  }
  return grid;
}

function layerToggles(step) {
  const row = document.createElement("div");
  row.className = "layer-toggles";
  for (const layer of Object.keys(step.overlay.boxes)) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" data-layer="${layer}"> <span class="swatch" style="background:${LAYER_COLORS[layer]}"></span> show ${layer} in layout viewer`;
    const input = label.querySelector("input");
    input.checked = stepsState.layers[layer];
    input.addEventListener("change", () => setLayer(layer, input.checked));
    row.append(label);
  }
  return row;
}

function setLayer(layer, on) {
  stepsState.layers[layer] = on;
  for (const input of stepsTimeline.querySelectorAll(`input[data-layer="${layer}"]`)) input.checked = on;
  drawCanvases();
}

// One shared Image per base so every step canvas draws from the same download.
function baseImage(base) {
  stepsState.baseImages = stepsState.baseImages || {};
  if (!stepsState.baseImages[base]) {
    const steps = stepsState.result.steps;
    const name = base === "binary"
      ? steps.find((step) => step.id === "threshold")?.images.binary
      : steps.find((step) => step.id === "input")?.images.input;
    const image = new Image();
    image.addEventListener("load", drawCanvases);
    if (name) image.src = stepsUrl(name);
    stepsState.baseImages[base] = image;
  }
  return stepsState.baseImages[base];
}

function stepCanvas(step) {
  const layers = Object.keys(step.overlay.boxes);
  const figure = document.createElement("figure");
  figure.className = "step-canvas";
  const canvas = document.createElement("canvas");
  const hover = document.createElement("figcaption");
  hover.textContent = `${stepsState.base === "binary" ? "Thresholded (binary) image" : "Input image"} with ${layers.join(" and ")} drawn on it · hover a box for details`;
  figure.append(canvas, hover);
  const entry = { canvas, hover, defaultCaption: hover.textContent, layers: () => layers };
  stepsState.canvases.push(entry);
  canvas.addEventListener("mousemove", (event) => hoverCanvas(event, entry));
  canvas.addEventListener("mouseleave", () => { hover.textContent = entry.defaultCaption; });
  queueMicrotask(() => watchCanvas(entry, figure.closest(".step-body") || figure.parentElement));
  return figure;
}

function drawCanvases() {
  for (const entry of stepsState.canvases) drawEntry(entry);
}

// A canvas cannot measure itself: watch the panel around it and draw once it has a width.
const stepsResize = new ResizeObserver((entries) => {
  for (const observed of entries) {
    const entry = stepsState.canvases.find((item) => item.container === observed.target);
    const width = Math.round(observed.contentRect.width);
    if (!entry || width < 1 || width === entry.lastWidth) continue;
    entry.lastWidth = width;
    drawEntry(entry);
  }
});

function watchCanvas(entry, container) {
  entry.container = container;
  stepsResize.observe(container);
}

function layoutBoxes() {
  const boxes = {};
  for (const step of stepsState.result.steps) {
    for (const [layer, items] of Object.entries(step.overlay?.boxes || {})) {
      boxes[layer] = items.map((item) => (item.box ? item : { box: item }));
    }
  }
  return boxes;
}

function renderLayoutViewer() {
  const section = document.createElement("section");
  section.className = "layout-viewer";
  section.innerHTML = `
    <div class="layout-head">
      <h4>Layout viewer <span>how Tesseract divided the image · hover a box for details</span></h4>
      <div class="layout-controls">
        <label><input type="radio" name="layoutBase" value="binary"> binary</label>
        <label><input type="radio" name="layoutBase" value="input"> input</label>
        ${Object.keys(LAYER_COLORS).map((layer) => `<label><input type="checkbox" data-layer="${layer}"> <span class="swatch" style="background:${LAYER_COLORS[layer]}"></span>${layer}</label>`).join("")}
      </div>
    </div>
    <div class="layout-stage"><canvas></canvas></div>
    <div class="layout-hover">Hover a box.</div>`;
  for (const radio of section.querySelectorAll('input[name="layoutBase"]')) {
    radio.checked = radio.value === stepsState.base;
    radio.addEventListener("change", () => {
      stepsState.base = radio.value;
      for (const entry of stepsState.canvases) {
        if (entry.defaultCaption) {
          entry.defaultCaption = entry.defaultCaption.replace(/^(Thresholded \(binary\)|Input) image/, radio.value === "binary" ? "Thresholded (binary) image" : "Input image");
          entry.hover.textContent = entry.defaultCaption;
        }
      }
      drawCanvases();
    });
  }
  for (const input of section.querySelectorAll("input[data-layer]")) {
    input.checked = stepsState.layers[input.dataset.layer];
    input.addEventListener("change", () => setLayer(input.dataset.layer, input.checked));
  }
  const canvas = section.querySelector("canvas");
  const hover = section.querySelector(".layout-hover");
  const entry = { canvas, hover, defaultCaption: "Hover a box.", layers: () => Object.keys(LAYER_COLORS).filter((layer) => stepsState.layers[layer]) };
  stepsState.canvases.push(entry);
  canvas.addEventListener("mousemove", (event) => hoverCanvas(event, entry));
  queueMicrotask(() => watchCanvas(entry, section.querySelector(".layout-stage")));
  return section;
}

function drawEntry(entry) {
  const image = baseImage(stepsState.base);
  if (!image.naturalWidth) return;
  const { canvas } = entry;
  // Fill the panel; small cards are enlarged up to 3× so thin boxes stay visible.
  // Width comes from the observed panel, never from the canvas's own box.
  const available = entry.container?.clientWidth || entry.lastWidth || image.naturalWidth;
  const width = Math.min(image.naturalWidth * 3, available);
  const scale = width / image.naturalWidth;
  canvas.width = Math.round(width);
  canvas.height = Math.round(image.naturalHeight * scale);
  entry.scale = scale;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const boxes = layoutBoxes();
  for (const layer of entry.layers()) {
    ctx.strokeStyle = LAYER_COLORS[layer];
    ctx.lineWidth = layer === "blocks" ? 3 : layer === "symbols" || layer === "blobs" ? 1 : 2;
    for (const item of boxes[layer] || []) {
      const { x, y, width: w, height: h } = item.box;
      ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);
      if (layer === "lines" && item.baseline) {
        const [x1, y1, x2, y2] = item.baseline;
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(x1 * scale, y1 * scale);
        ctx.lineTo(x2 * scale, y2 * scale);
        ctx.stroke();
        ctx.restore();
      }
    }
  }
}

function hoverCanvas(event, entry) {
  const { canvas, hover: output } = entry;
  if (!entry.scale) return;
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) * (canvas.width / rect.width)) / entry.scale;
  const y = ((event.clientY - rect.top) * (canvas.height / rect.height)) / entry.scale;
  let best = null;
  const boxes = layoutBoxes();
  for (const layer of entry.layers()) {
    for (const item of boxes[layer] || []) {
      const b = item.box;
      if (x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height) {
        const area = b.width * b.height;
        if (!best || area < best.area) best = { layer, item, area };
      }
    }
  }
  if (!best) {
    output.textContent = `x ${Math.round(x)}, y ${Math.round(y)} · no visible box here`;
    return;
  }
  const { layer, item } = best;
  const b = item.box;
  const parts = [`${layer.replace(/s$/, "")}${item.id != null ? ` #${item.id}` : ""}`, `x ${b.x} y ${b.y} · ${b.width}×${b.height}`];
  if (item.block_type) parts.push(item.block_type);
  if (item.text) parts.push(`"${item.text}"`);
  if (item.confidence != null) parts.push(`conf ${item.confidence}`);
  output.textContent = parts.join(" · ");
}

function linesTable(lines) {
  const heights = lines.map((line) => line.box.height).sort((a, b) => a - b);
  const medianHeight = heights[Math.floor(heights.length / 2)] || 0;
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.className = "steps-table lines-table";
  table.innerHTML = `<thead><tr><th>#</th><th>What the LSTM reads (original)</th><th>Binary crop</th><th>Text · words</th><th>Conf</th></tr></thead>`;
  const body = document.createElement("tbody");
  for (const line of lines) {
    const tall = medianHeight && line.box.height > medianHeight * 1.6;
    const row = document.createElement("tr");
    if (tall) row.classList.add("tall-line");
    const words = (line.words || []).map((word) => {
      const symbols = (word.symbols || []).map((symbol) => {
        const alternatives = (symbol.choices || []).map((choice) => `${choice.text} ${choice.confidence}`).join("  ·  ");
        return `<span class="symbol" title="${escapeHtml(`conf ${symbol.confidence}${alternatives ? ` — alternatives: ${alternatives}` : ""}`)}">${escapeHtml(symbol.text)}</span>`;
      }).join("");
      return `<span class="word-chip ${word.language === "ben" ? "lang-ben" : "lang-eng"}" title="${escapeHtml(`conf ${word.confidence} · ${word.language} · ${word.from_dictionary ? "in dictionary" : "not in dictionary"}`)}"><span dir="auto">${symbols || escapeHtml(word.text)}</span><small>${escapeHtml(word.language || "?")} ${Math.round(word.confidence)}${word.from_dictionary ? " ★" : ""}</small></span>`;
    }).join("");
    row.innerHTML = `
      <td>${line.id}${tall ? '<br><span class="warn-chip" title="Much taller than the median line: rows were probably merged">tall</span>' : ""}</td>
      <td>${line.images?.original ? `<img src="${stepsUrl(line.images.original)}" alt="line ${line.id} original">` : ""}</td>
      <td>${line.images?.binary ? `<img src="${stepsUrl(line.images.binary)}" alt="line ${line.id} binary">` : ""}</td>
      <td><div class="line-text" dir="auto">${escapeHtml(line.text ?? "(layout only)")}</div><div class="word-chips">${words}</div></td>
      <td>${line.confidence ?? "–"}</td>`;
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  return wrap;
}

function fieldRowsTable(step) {
  const page = step.values.page_fields || {};
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.className = "steps-table lines-table";
  table.innerHTML = `<thead><tr><th>Row</th><th>Crop that was re-read</th><th>Page pass gave</th><th>Chosen value</th><th>From</th></tr></thead>`;
  const body = document.createElement("tbody");
  for (const row of step.rows || []) {
    const before = page[row.field] ?? "";
    const gained = row.text && row.text !== before;
    const tr = document.createElement("tr");
    if (gained) tr.classList.add("gained-row");
    tr.innerHTML = `
      <td>${escapeHtml(row.label)}<br><small>${escapeHtml((row.languages || []).join("+"))}</small></td>
      <td>${row.crop ? `<img src="${stepsUrl(row.crop)}" alt="${escapeHtml(row.field)} crop">` : "—"}</td>
      <td dir="auto">${escapeHtml(before || "—")}</td>
      <td dir="auto"><strong>${escapeHtml(row.text || "—")}</strong></td>
      <td>${escapeHtml(row.source)}${row.confidence == null ? "" : `<br><small>conf ${Math.round(row.confidence * 100)}</small>`}</td>`;
    body.append(tr);
  }
  table.append(body);
  wrap.append(table);
  return wrap;
}

function preBlock(text) {
  const pre = document.createElement("pre");
  pre.className = "ocr-text steps-pre";
  pre.textContent = text;
  return pre;
}

stepsFilter.addEventListener("change", () => {
  if (!stepsFilter.value) return;
  filterSelect.value = stepsFilter.value;
  filterSelect.dispatchEvent(new Event("change")); // keep the Filter panel preview in step
  scheduleStepsRun();
});
filterSelect.addEventListener("change", syncFilterSelect);
stepsMethod.addEventListener("change", () => { syncMethodInputs(); scheduleStepsRun(); });
for (const input of Object.values(stepsInputs)) input.addEventListener("change", scheduleStepsRun);
stepsValue.addEventListener("input", () => { stepsValueNumber.value = stepsValue.value; stepsValueOut.textContent = stepsValue.value; });
stepsValue.addEventListener("change", scheduleStepsRun);
stepsValueNumber.addEventListener("change", () => {
  const value = Math.max(0, Math.min(255, Math.round(Number(stepsValueNumber.value) || 0)));
  stepsValue.value = stepsValueNumber.value = value;
  stepsValueOut.textContent = value;
  scheduleStepsRun();
});
for (const input of [stepsDpi, stepsLayoutOnly, stepsDebug, stepsFields]) input.addEventListener("change", scheduleStepsRun);
if (stepsRun) stepsRun.addEventListener("click", () => runSteps());
window.runOcrSteps = runSteps;
window.clearOcrSteps = (message) => {
  stepsStatus.textContent = message;
  stepsStatus.classList.remove("failed");
  stepsSummary.innerHTML = "";
  stepsTimeline.innerHTML = "";
};
window.addEventListener("resize", drawCanvases);
syncMethodInputs();
syncFilterSelect();
