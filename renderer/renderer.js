// Renderer: handles file load, slider state, debounced sidecar calls, preview.

const els = {
  status: document.getElementById('status'),
  drop: document.getElementById('drop'),
  pick: document.getElementById('pick'),
  file: document.getElementById('file'),
  numColors: document.getElementById('numColors'),
  numColorsVal: document.getElementById('numColorsVal'),
  transparent: document.getElementById('transparent'),
  cleanEdges: document.getElementById('cleanEdges'),
  keyEnabled: document.getElementById('keyEnabled'),
  keyColor: document.getElementById('keyColor'),
  keyTol: document.getElementById('keyTol'),
  keyTolVal: document.getElementById('keyTolVal'),
  outline: document.getElementById('outline'),
  outlineDiagonal: document.getElementById('outlineDiagonal'),
  outlineThick: document.getElementById('outlineThick'),
  outlineThickVal: document.getElementById('outlineThickVal'),
  saturation: document.getElementById('saturation'),
  saturationVal: document.getElementById('saturationVal'),
  palettePreset: document.getElementById('palettePreset'),
  dither: document.getElementById('dither'),
  importPaletteBtn: document.getElementById('importPaletteBtn'),
  paletteFile: document.getElementById('paletteFile'),
  clearPaletteBtn: document.getElementById('clearPaletteBtn'),
  paletteSwatches: document.getElementById('paletteSwatches'),
  pxw: document.getElementById('pxw'),
  pxwVal: document.getElementById('pxwVal'),
  save: document.getElementById('save'),
  meta: document.getElementById('meta'),
  srcImg: document.getElementById('srcImg'),
  outImg: document.getElementById('outImg'),
  busy: document.getElementById('busy'),
  selectBtn: document.getElementById('selectBtn'),
  selInfo: document.getElementById('selInfo'),
  selRect: document.getElementById('selRect'),
  batchInputBtn: document.getElementById('batchInputBtn'),
  batchInputInfo: document.getElementById('batchInputInfo'),
  batchOutputBtn: document.getElementById('batchOutputBtn'),
  batchOutputInfo: document.getElementById('batchOutputInfo'),
  batchStartBtn: document.getElementById('batchStartBtn'),
  batchCancelBtn: document.getElementById('batchCancelBtn'),
  batchProgressWrap: document.getElementById('batchProgressWrap'),
  batchProgressFill: document.getElementById('batchProgressFill'),
  batchProgressLabel: document.getElementById('batchProgressLabel'),
  batchStatus: document.getElementById('batchStatus'),
};

let port = null;
let sourceB64 = null;     // raw base64 of the loaded source image
let sourceName = 'image.png';
let lastOutputB64 = null; // for save
let inflight = null;      // AbortController for the in-flight request
let pending = false;      // a request was queued while one was running
let debounceTimer = null;

// ---- bootstrapping ----
(async function init() {
  port = await window.ppa.getPort();
  els.status.textContent = `sidecar @ 127.0.0.1:${port}`;
  els.status.classList.add('ok');
})();

// ---- file loading ----
function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  sourceName = file.name || 'image.png';
  const reader = new FileReader();
  reader.onload = () => {
    // reader.result is a data URL: "data:image/png;base64,XXXX"
    const dataUrl = reader.result;
    sourceB64 = dataUrl.split(',', 2)[1];
    els.srcImg.src = dataUrl;
    scheduleRender(0); // immediate
  };
  reader.readAsDataURL(file);
}

els.pick.addEventListener('click', () => els.file.click());
els.file.addEventListener('change', (e) => loadFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((ev) =>
  els.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    els.drop.classList.add('over');
  }));
['dragleave', 'drop'].forEach((ev) =>
  els.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    els.drop.classList.remove('over');
  }));
els.drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f);
});
// Also accept drops anywhere on the window.
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f);
});

// ---- slider wiring ----
function wireSlider(input, label, fmt) {
  const update = () => {
    label.textContent = fmt(input.value);
    scheduleRender();
  };
  input.addEventListener('input', update);
  update();
}
wireSlider(els.numColors, els.numColorsVal, (v) => v);
wireSlider(els.pxw,       els.pxwVal,       (v) => (v === '0' ? 'auto' : v));
wireSlider(els.outlineThick, els.outlineThickVal, (v) => v);
wireSlider(els.saturation, els.saturationVal, (v) => `${v}%`);
wireSlider(els.keyTol, els.keyTolVal, (v) => v);
els.keyEnabled.addEventListener('change', () => scheduleRender());
els.keyColor.addEventListener('input', () => scheduleRender());
els.transparent.addEventListener('change', () => scheduleRender());
els.cleanEdges.addEventListener('change', () => scheduleRender());
els.outline.addEventListener('change', () => scheduleRender());
els.outlineDiagonal.addEventListener('change', () => scheduleRender());
els.dither.addEventListener('change', () => scheduleRender());
els.palettePreset.addEventListener('change', () => {
  // Choosing a preset clears any imported palette.
  if (els.palettePreset.value) {
    importedPalette = null;
    renderSwatches([]);
    els.clearPaletteBtn.classList.add('hidden');
  }
  scheduleRender();
});

// ---- imported palette state ----
let importedPalette = null; // array of [r,g,b] or null

function renderSwatches(colors) {
  els.paletteSwatches.innerHTML = '';
  for (const [r, g, b] of colors) {
    const sw = document.createElement('div');
    sw.className = 'sw';
    sw.style.background = `rgb(${r},${g},${b})`;
    sw.title = `${r}, ${g}, ${b}`;
    els.paletteSwatches.appendChild(sw);
  }
}

els.importPaletteBtn.addEventListener('click', () => els.paletteFile.click());
els.clearPaletteBtn.addEventListener('click', () => {
  importedPalette = null;
  renderSwatches([]);
  els.clearPaletteBtn.classList.add('hidden');
  scheduleRender();
});
els.paletteFile.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const b64 = reader.result.split(',', 2)[1];
    try {
      const res = await fetch(`http://127.0.0.1:${port}/extract_palette`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_b64: b64, max_colors: 32 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      importedPalette = json.palette;
      renderSwatches(importedPalette);
      els.clearPaletteBtn.classList.remove('hidden');
      // Importing a palette clears any preset.
      els.palettePreset.value = '';
      scheduleRender();
    } catch (err) {
      console.error(err);
      els.status.textContent = `palette import failed: ${err.message}`;
      els.status.classList.add('err');
    }
    els.paletteFile.value = '';
  };
  reader.readAsDataURL(file);
});

// ---- debounced render ----
function scheduleRender(delay = 250) {
  if (!sourceB64) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runRender, delay);
}

// Build a /pixelate request body from the current control state.
// `imageB64` lets the batch loop substitute each file's bytes while keeping
// every other setting identical to the live preview.
function buildPixelateBody(imageB64) {
  const body = {
    image_b64: imageB64,
    num_colors: Number(els.numColors.value),
    scale_result: 1,
    transparent_background: els.transparent.checked,
    clean_edges: els.cleanEdges.checked,
    outline: els.outline.checked,
    outline_thickness: Number(els.outlineThick.value),
    outline_diagonal: els.outlineDiagonal.checked,
    dither: els.dither.checked,
    saturation: Number(els.saturation.value) / 100,
  };
  if (els.keyEnabled.checked) {
    const hex = els.keyColor.value;
    body.key_color = [
      parseInt(hex.slice(1, 3), 16),
      parseInt(hex.slice(3, 5), 16),
      parseInt(hex.slice(5, 7), 16),
    ];
    body.key_tolerance = Number(els.keyTol.value);
  }
  if (importedPalette && importedPalette.length) {
    body.palette = importedPalette;
  } else if (els.palettePreset.value) {
    body.palette_preset = els.palettePreset.value;
  }
  const pxwVal = Number(els.pxw.value);
  if (pxwVal > 0) body.pixel_width = pxwVal;
  return body;
}

async function runRender() {
  if (!sourceB64) return;
  if (inflight) {
    // mark that another render is needed; current one will trigger it on completion
    pending = true;
    return;
  }
  inflight = new AbortController();
  els.busy.classList.remove('hidden');

  const body = buildPixelateBody(sourceB64);

  const t0 = performance.now();
  try {
    const res = await fetch(`http://127.0.0.1:${port}/pixelate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: inflight.signal,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`${res.status}: ${detail}`);
    }
    const json = await res.json();
    lastOutputB64 = json.image_b64;
    els.outImg.src = `data:image/png;base64,${json.image_b64}`;
    els.save.disabled = false;
    const dt = (performance.now() - t0).toFixed(0);
    els.meta.textContent =
      `output: ${json.width} × ${json.height}\n` +
      `render: ${dt} ms`;
    els.status.textContent = `sidecar @ 127.0.0.1:${port}`;
    els.status.classList.remove('err');
    els.status.classList.add('ok');
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error(err);
      els.status.textContent = `error: ${err.message}`;
      els.status.classList.remove('ok');
      els.status.classList.add('err');
    }
  } finally {
    inflight = null;
    els.busy.classList.add('hidden');
    if (pending) {
      pending = false;
      scheduleRender(0);
    }
  }
}

// ---- preview zoom (mouse wheel) ----
let gridState = { cell: 1, originX: 0, originY: 0, nw: 0, nh: 0 };
let selectMode = false;

(function setupPreviewZoom() {
  const wrap = els.outImg.parentElement;
  let zoom = 1, panX = 0, panY = 0;
  let dragging = false, lastX = 0, lastY = 0;
  let selecting = false, selStartPx = null, selEndPx = null;
  els.outImg.style.transformOrigin = '0 0';
  wrap.style.overflow = 'hidden';
  els.outImg.style.imageRendering = 'pixelated';
  els.outImg.style.cursor = 'grab';

  const apply = () => {
    els.outImg.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    // Sync checker background to the output's pixel grid.
    const nw = els.outImg.naturalWidth, nh = els.outImg.naturalHeight;
    if (!nw || !nh) return;
    const wrapRect = wrap.getBoundingClientRect();
    const imgRect = els.outImg.getBoundingClientRect();
    // object-fit: contain — figure out displayed image area inside the img box.
    const fit = Math.min(imgRect.width / nw, imgRect.height / nh);
    const dispW = nw * fit, dispH = nh * fit;
    const originX = (imgRect.left - wrapRect.left) + (imgRect.width - dispW) / 2;
    const originY = (imgRect.top - wrapRect.top) + (imgRect.height - dispH) / 2;
    const cell = fit; // one output pixel in screen px
    wrap.style.backgroundSize = `${cell * 2}px ${cell * 2}px`;
    wrap.style.backgroundPosition =
      `${originX}px ${originY}px, ${originX}px ${originY + cell}px, ` +
      `${originX + cell}px ${originY - cell}px, ${originX - cell}px ${originY}px`;
    gridState = { cell, originX, originY, nw, nh };
    drawSelRect();
  };

  function clientToPixel(clientX, clientY) {
    const wrapRect = wrap.getBoundingClientRect();
    const x = (clientX - wrapRect.left - gridState.originX) / gridState.cell;
    const y = (clientY - wrapRect.top - gridState.originY) / gridState.cell;
    return {
      x: Math.max(0, Math.min(gridState.nw, x)),
      y: Math.max(0, Math.min(gridState.nh, y)),
    };
  }

  function drawSelRect() {
    if (!selStartPx || !selEndPx) { els.selRect.classList.add('hidden'); return; }
    const x0 = Math.floor(Math.min(selStartPx.x, selEndPx.x));
    const y0 = Math.floor(Math.min(selStartPx.y, selEndPx.y));
    const x1 = Math.ceil(Math.max(selStartPx.x, selEndPx.x));
    const y1 = Math.ceil(Math.max(selStartPx.y, selEndPx.y));
    const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
    const { cell, originX, originY } = gridState;
    els.selRect.style.left = `${originX + x0 * cell}px`;
    els.selRect.style.top = `${originY + y0 * cell}px`;
    els.selRect.style.width = `${w * cell}px`;
    els.selRect.style.height = `${h * cell}px`;
    els.selRect.classList.remove('hidden');
    els.selInfo.textContent = `${w} × ${h} px`;
  }

  function clearSelection() {
    selStartPx = null; selEndPx = null;
    els.selRect.classList.add('hidden');
    els.selInfo.textContent = '';
  }

  els.selectBtn.addEventListener('click', () => {
    selectMode = !selectMode;
    els.selectBtn.classList.toggle('active', selectMode);
    wrap.style.cursor = selectMode ? 'crosshair' : '';
    if (!selectMode) clearSelection();
  });
  const reset = () => { zoom = 1; panX = 0; panY = 0; apply(); };

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = els.outImg.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // image-space coords under cursor
    const ix = (mx) / zoom;
    const iy = (my) / zoom;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.min(40, Math.max(0.2, zoom * factor));
    // keep cursor anchored: adjust pan so that ix,iy stays under cursor
    panX += mx - ix * newZoom - (mx - ix * zoom);
    panY += my - iy * newZoom - (my - iy * zoom);
    zoom = newZoom;
    apply();
  }, { passive: false });

  wrap.addEventListener('mousedown', (e) => {
    if (selectMode) {
      selecting = true;
      selStartPx = clientToPixel(e.clientX, e.clientY);
      selEndPx = selStartPx;
      drawSelRect();
      return;
    }
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    els.outImg.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', (e) => {
    if (selecting) {
      selEndPx = clientToPixel(e.clientX, e.clientY);
      drawSelRect();
      return;
    }
    if (!dragging) return;
    panX += e.clientX - lastX;
    panY += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    apply();
  });
  window.addEventListener('mouseup', () => {
    selecting = false;
    dragging = false;
    els.outImg.style.cursor = 'grab';
  });
  wrap.addEventListener('dblclick', reset);
  els.outImg.addEventListener('load', apply);
  window.addEventListener('resize', apply);
})();

// ---- source preview zoom (mouse wheel) ----
(function setupSourceZoom() {
  const img = els.srcImg;
  const wrap = img.parentElement;
  let zoom = 1, panX = 0, panY = 0;
  let dragging = false, lastX = 0, lastY = 0;
  img.style.transformOrigin = '0 0';
  wrap.style.overflow = 'hidden';
  wrap.style.cursor = 'grab';

  const apply = () => {
    img.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  };
  const reset = () => { zoom = 1; panX = 0; panY = 0; apply(); };

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = img.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const ix = mx / zoom, iy = my / zoom;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const nz = Math.min(40, Math.max(0.2, zoom * factor));
    panX += mx - ix * nz - (mx - ix * zoom);
    panY += my - iy * nz - (my - iy * zoom);
    zoom = nz;
    apply();
  }, { passive: false });

  wrap.addEventListener('mousedown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    wrap.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    panX += e.clientX - lastX;
    panY += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    apply();
  });
  window.addEventListener('mouseup', () => {
    dragging = false;
    wrap.style.cursor = 'grab';
  });
  wrap.addEventListener('dblclick', reset);
  img.addEventListener('load', reset);
})();

// ---- batch processing ----
let batchInputFolder = null;
let batchInputFiles = [];   // [{ name, path }]
let batchOutputFolder = null;
let batchCancelRequested = false;
let batchRunning = false;

function updateBatchStartEnabled() {
  els.batchStartBtn.disabled = !(
    batchInputFiles.length > 0 && batchOutputFolder && !batchRunning
  );
}

function setBatchProgress(done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  els.batchProgressFill.style.width = `${pct}%`;
  els.batchProgressLabel.textContent = `${pct}%`;
}

els.batchInputBtn.addEventListener('click', async () => {
  const res = await window.ppa.pickInputFolder();
  if (!res) return;
  batchInputFolder = res.folder;
  batchInputFiles = res.files || [];
  if (batchInputFiles.length === 0) {
    els.batchInputInfo.textContent = `${batchInputFolder} (no images found)`;
  } else {
    els.batchInputInfo.textContent =
      `${batchInputFolder} (${batchInputFiles.length} image${batchInputFiles.length === 1 ? '' : 's'})`;
  }
  updateBatchStartEnabled();
});

els.batchOutputBtn.addEventListener('click', async () => {
  const folder = await window.ppa.pickOutputFolder();
  if (!folder) return;
  batchOutputFolder = folder;
  els.batchOutputInfo.textContent = folder;
  updateBatchStartEnabled();
});

els.batchCancelBtn.addEventListener('click', () => {
  if (!batchRunning) return;
  batchCancelRequested = true;
  els.batchStatus.textContent = 'cancelling…';
});

els.batchStartBtn.addEventListener('click', async () => {
  if (batchRunning) return;
  if (!batchInputFiles.length || !batchOutputFolder) return;

  batchRunning = true;
  batchCancelRequested = false;
  updateBatchStartEnabled();
  els.batchCancelBtn.classList.remove('hidden');
  els.batchProgressWrap.classList.remove('hidden');
  setBatchProgress(0, batchInputFiles.length);

  // Path separator: assume Windows uses '\\' but accept either.
  const sep = batchOutputFolder.includes('\\') ? '\\' : '/';

  let done = 0;
  let failures = 0;
  const total = batchInputFiles.length;

  for (const file of batchInputFiles) {
    if (batchCancelRequested) break;
    els.batchStatus.textContent = `processing ${file.name} (${done + 1}/${total})`;

    try {
      const inputB64 = await window.ppa.readImageB64(file.path);
      const body = buildPixelateBody(inputB64);

      const res = await fetch(`http://127.0.0.1:${port}/pixelate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status}: ${detail}`);
      }
      const json = await res.json();

      // Output filename: same stem as the source, always written as PNG.
      const dot = file.name.lastIndexOf('.');
      const stem = dot > 0 ? file.name.slice(0, dot) : file.name;
      const outPath = `${batchOutputFolder}${sep}${stem}.png`;
      await window.ppa.saveImageB64(outPath, json.image_b64);
    } catch (err) {
      console.error(`batch failed on ${file.name}:`, err);
      failures += 1;
    }

    done += 1;
    setBatchProgress(done, total);
  }

  batchRunning = false;
  els.batchCancelBtn.classList.add('hidden');
  updateBatchStartEnabled();

  if (batchCancelRequested) {
    els.batchStatus.textContent = `cancelled at ${done}/${total}` +
      (failures ? ` (${failures} failed)` : '');
  } else {
    els.batchStatus.textContent =
      `done: ${done - failures}/${total} saved` +
      (failures ? `, ${failures} failed` : '');
  }
});

// ---- save output ----
els.save.addEventListener('click', () => {
  if (!lastOutputB64) return;
  const a = document.createElement('a');
  a.href = `data:image/png;base64,${lastOutputB64}`;
  const dot = sourceName.lastIndexOf('.');
  const stem = dot > 0 ? sourceName.slice(0, dot) : sourceName;
  a.download = `${stem}.png`;
  a.click();
});
