/* ENR Control Center — vanilla JS SPA. Talks to the same REST API as every client. */
"use strict";

const state = {
  token: localStorage.getItem("enr_token") || "",
  page: "dashboard",
  timers: [],
  models: [],
  providers: [],
  routable: [],
  chat: { messages: [], model: "", streaming: false },
  compare: { a: { messages: [], model: "" }, b: { messages: [], model: "" } },
  modelManager: { provider: "ollama", pullModel: "", pulling: false, jobs: [] },
};

const $ = (sel, root) => (root || document).querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
};

function authHeaders() {
  return state.token ? { Authorization: "Bearer " + state.token } : {};
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || (opts.body ? "POST" : "GET"),
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status === 401) {
    promptToken();
    throw new Error("Unauthorized — enter the admin token (Key button)");
  }
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(data?.error?.message || data?.message || ("HTTP " + res.status));
  return data;
}

function toast(message, kind = "ok") {
  const node = $("#toast");
  node.textContent = message;
  node.className = "toast " + kind;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => node.classList.add("hidden"), 3500);
}

function fmtBytes(n) {
  if (n == null || n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + " " + units[i];
}

function fmtDuration(ms) {
  if (ms == null) return "—";
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60), h = Math.floor(m / 60), d = Math.floor(h / 24);
  if (d > 0) return d + "d " + (h % 24) + "h";
  if (h > 0) return h + "h " + (m % 60) + "m";
  return m + "m " + (s % 60) + "s";
}

function badge(text, kind) { return el("span", { class: "badge " + kind, text }); }
function stateBadge(loaded, state) {
  if (loaded && state === "ready") return badge("running", "ok");
  if (loaded) return badge(state || "loaded", "warn");
  return badge("stopped", "muted");
}

function promptToken() {
  const body = $("#modal-body");
  body.innerHTML = "";
  body.append(
    el("label", { text: "Admin token (server.auth_token)" }),
    el("input", { type: "password", id: "token-input", value: state.token, placeholder: "Bearer token" }),
  );
  $("#modal-title").textContent = "Authentication";
  $("#modal-ok").onclick = () => {
    state.token = $("#token-input").value.trim();
    localStorage.setItem("enr_token", state.token);
    closeModal();
    render();
  };
  $("#modal-cancel").onclick = closeModal;
  $("#modal").classList.remove("hidden");
}

function openModal(title, builder) {
  const body = $("#modal-body");
  body.innerHTML = "";
  builder(body);
  $("#modal-title").textContent = title;
  $("#modal-ok").onclick = null;
  $("#modal-ok").classList.remove("hidden");
  $("#modal-cancel").onclick = closeModal;
  $("#modal").classList.remove("hidden");
  return body;
}

function closeModal() { $("#modal").classList.add("hidden"); }

// ---------------------------------------------------------------- pages

async function render() {
  const main = $("#main");
  main.innerHTML = "";
  state.timers.forEach(clearInterval);
  state.timers = [];
  document.querySelectorAll("#nav a").forEach(a =>
    a.classList.toggle("active", a.dataset.page === state.page));
  try {
    switch (state.page) {
      case "dashboard": await renderDashboard(main); break;
      case "models": await renderModels(main); break;
      case "modelmanager": await renderModelManager(main); break;
      case "providers": await renderProviders(main); break;
      case "runtime": await renderRuntime(main); break;
      case "playground": await renderPlayground(main); break;
      case "diagnostics": await renderDiagnostics(main); break;
    }
  } catch (e) {
    main.append(el("div", { class: "card", text: "Failed to load: " + e.message }));
  }
}

async function renderDashboard(main) {
  const data = await api("/api/v1/dashboard");
  const status = await api("/api/v1/status");
  const server = data.server;
  main.append(
    el("h1", { text: "Dashboard" }),
    el("p", { class: "sub", text: "Server status, resources and active inference" }),
  );
  const g = el("div", { class: "grid cols-4" });
  const statusKind = server.status === "healthy" ? "ok" : "err";
  g.append(
    statCard(server.status === "healthy" ? "healthy" : "degraded", "Server status", statusKind),
    statCard(fmtDuration(server.uptime_ms), "Uptime"),
    statCard(String(data.models_loaded) + " / " + String(data.models_installed), "Models loaded / installed"),
    statCard(String(data.active_requests?.length || 0), "Active requests"),
    statCard(String(status.total_requests || 0), "Total requests"),
    statCard(String(status.total_generated_tokens || 0), "Tokens generated"),
    statCard(String(status.tokens_per_second || 0), "Tokens / second"),
    statCard(String(data.providers || 0), "Providers configured"),
  );
  main.append(el("h2", { text: "API endpoints" }), g);

  const endpoints = el("div", { class: "grid cols-2" });
  endpoints.append(kvCard("API endpoints", {
    "Web UI": server.web_ui,
    "Native API": server.api,
    "Ollama API": server.ollama_api,
    "OpenAI API": server.openai_api,
  }));
  const hw = data.hardware || {};
  const cpu = hw.cpu || {}, ram = hw.ram || {}, gpus = hw.gpus || [];
  const res = el("div", { class: "grid cols-2" });
  res.append(kvCard("CPU", {
    Processors: cpu.processors,
    "Load average": cpu.load_average,
    Utilization: cpu.utilization_percent == null ? "n/a" : cpu.utilization_percent + " %",
  }));
  const ramCard = kvCard("RAM", {
    "JVM heap used": fmtBytes(ram.jvm_heap_used_bytes),
    "JVM heap max": fmtBytes(ram.jvm_heap_max_bytes),
    "System total": fmtBytes(ram.total_bytes),
    "System free": fmtBytes(ram.free_bytes),
  });
  res.append(ramCard);
  endpoints.append(res);
  main.append(el("h2", { text: "Hardware" }), endpoints);

  if (gpus.length) {
    const gpuGrid = el("div", { class: "grid cols-2" });
    for (const gpu of gpus) {
      gpuGrid.append(kvCard("GPU " + gpu.index + " — " + (gpu.name || "unknown"), {
        Utilization: gpu.utilization_percent == null ? "n/a" : gpu.utilization_percent + " %",
        "VRAM used": fmtBytes(gpu.used_memory_bytes ?? (gpu.total_memory_bytes - (gpu.free_memory_bytes || 0))),
        "VRAM total": fmtBytes(gpu.total_memory_bytes),
        Temperature: gpu.temperature_c == null ? "n/a" : gpu.temperature_c + " °C",
        Driver: gpu.driver_version || "—",
      }));
    }
    main.append(el("h2", { text: "GPU" }), gpuGrid);
  }

  if (data.active_requests?.length) {
    const reqCard = el("div", { class: "card" });
    reqCard.append(el("h3", { text: "Active inference requests" }));
    const table = el("table");
    table.append(tr(["Request", "Model", "Elapsed", "Max tokens"], true));
    for (const r of data.active_requests) {
      table.append(tr([
        el("span", { class: "mono", text: (r.request_id || "").slice(0, 8) }),
        r.model, fmtDuration(r.elapsed_ms), String(r.max_tokens),
      ]));
    }
    reqCard.append(table);
    main.append(reqCard);
  }
  state.timers.push(setInterval(render, 5000));
}

function statCard(value, label, kind = "") {
  return el("div", { class: "card stat" },
    el("span", { class: "value " + kind, text: value }),
    el("span", { class: "label", text: label }));
}

function kvCard(title, entries) {
  const card = el("div", { class: "card" });
  card.append(el("h3", { text: title }));
  const dl = el("dl", { class: "kv" });
  for (const [k, v] of Object.entries(entries)) {
    dl.append(el("dt", { text: k }), el("dd", { text: String(v ?? "—") }));
  }
  card.append(dl);
  return card;
}

function tr(cells, header = false) {
  const row = el("tr");
  for (const cell of cells) {
    row.append(header ? el("th", { text: cell }) : el("td", {}, cell));
  }
  return row;
}

// ---------------------------------------------------------------- models

async function renderModels(main) {
  const data = await api("/api/v1/catalog/models");
  state.models = data.models || [];
  main.append(el("h1", { text: "Models" }),
    el("p", { class: "sub", text: "Installed local models — discover, pull, import, start, stop" }));
  const actions = el("div", { class: "row", style: "margin-bottom:16px" });
  const search = el("input", { placeholder: "Search catalogs (Ollama Library, Hugging Face)…" });
  search.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(search.value); });
  actions.append(search,
    el("button", { class: "btn primary", text: "Search", onclick: () => doSearch(search.value) }),
    el("button", { class: "btn", text: "Pull by name…", onclick: () => promptPull() }),
    el("button", { class: "btn", text: "Import GGUF…", onclick: () => promptImport() }),
  );
  main.append(actions);

  const downloadsCard = el("div", { class: "card", id: "downloads-card", style: "margin-bottom:16px" });
  main.append(downloadsCard);
  const refreshDownloads = () => renderDownloads(downloadsCard);
  refreshDownloads();
  state.timers.push(setInterval(refreshDownloads, 1500));

  const card = el("div", { class: "card" });
  if (!state.models.length) {
    card.append(el("div", { class: "empty", text: "No models installed. Search or pull a model to get started." }));
  } else {
    const table = el("table");
    table.append(tr(["Name", "Status", "Size", "Source", "Format", "Verified", "Actions"], true));
    for (const m of state.models) {
      const row = tr([
        el("span", { class: "mono", text: m.id }),
        stateBadge(m.loaded, m.state),
        fmtBytes(m.size_bytes),
        m.source || "—",
        m.format || "—",
        m.verified ? badge("yes", "ok") : badge("no", "muted"),
        el("div", { class: "row", style: "gap:6px" },
          m.loaded
            ? el("button", { class: "btn small", text: "Stop", onclick: () => stopModel(m.id) })
            : el("button", { class: "btn small primary", text: "Start", onclick: () => startModel(m.id) }),
          el("button", { class: "btn small", text: "Details", onclick: () => showModelDetails(m.id) }),
          el("button", { class: "btn small", text: "Verify", onclick: () => verifyModel(m.id) }),
          el("button", { class: "btn small danger", text: "Remove", onclick: () => removeModel(m.id) }),
        ),
      ]);
      table.append(row);
    }
    card.append(table);
  }
  main.append(card);
}

// Model Manager: provider pull/list/remove + shared download jobs (scenario C).
async function renderModelManager(main) {
  main.append(el("h1", { text: "Model Manager" }),
    el("p", { class: "sub", text: "Pull, list and remove models on registered providers" }));
  const mm = state.modelManager;
  mm.jobs = [];

  const pullCard = el("div", { class: "card", style: "margin-bottom:16px" });
  const providerSelect = el("select");
  let providers = [];
  try {
    const data = await api("/api/v1/models/providers");
    providers = (data.providers || []).filter(p => (p.capabilities || []).includes("MODEL_PULL"));
  } catch (e) {
    pullCard.append(el("div", { class: "empty", text: "Providers unavailable: " + e.message }));
  }
  for (const p of providers) {
    providerSelect.append(el("option", { value: p.id, text: p.display_name + " (" + p.id + ")" }));
  }
  if (!mm.provider || !providers.some(p => p.id === mm.provider)) {
    mm.provider = providers.length ? providers[0].id : "ollama";
  }
  providerSelect.value = mm.provider;
  providerSelect.addEventListener("change", () => {
    mm.provider = providerSelect.value;
    refreshInstalled();
  });

  const modelInput = el("input", { placeholder: "Model identifier, e.g. qwen3-vl:30b", value: mm.pullModel || "" });
  modelInput.addEventListener("input", () => { mm.pullModel = modelInput.value; });
  const errorBox = el("div", { class: "sub", style: "color:var(--err);min-height:18px" });
  const pullButton = el("button", { class: "btn primary", text: "Pull Model" });
  pullButton.addEventListener("click", async () => {
    const model = modelInput.value.trim();
    if (!model) { errorBox.textContent = "Enter a model identifier"; return; }
    if (mm.pulling) return;
    mm.pulling = true;
    pullButton.disabled = true;
    errorBox.textContent = "";
    try {
      const job = await api("/api/v1/models/providers/" + encodeURIComponent(mm.provider) + "/pull",
        { body: { model } });
      mm.jobs.unshift(job.jobId || job.job_id);
      toast("Pull started: " + model + " (" + (job.jobId || job.job_id) + ")");
      refreshDownloads();
    } catch (e) {
      errorBox.textContent = e.message;
    } finally {
      mm.pulling = false;
      pullButton.disabled = false;
    }
  });
  pullCard.append(el("h3", { text: "Pull Model" }),
    el("div", { class: "row" },
      el("label", { text: "Provider", style: "flex:0 0 auto" }),
      providerSelect,
      el("label", { text: "Model", style: "flex:0 0 auto" }),
      modelInput,
      pullButton,
      el("button", { class: "btn", text: "Refresh", onclick: () => refreshInstalled() }),
    ),
    errorBox);
  main.append(pullCard);

  const downloadsCard = el("div", { class: "card", id: "mm-downloads-card", style: "margin-bottom:16px" });
  main.append(downloadsCard);
  const refreshDownloads = () => renderModelDownloads(downloadsCard);
  refreshDownloads();
  state.timers.push(setInterval(refreshDownloads, 1000));

  const installedCard = el("div", { class: "card", id: "mm-installed-card" });
  main.append(installedCard);
  const refreshInstalled = () => renderInstalled(installedCard);
  refreshInstalled();
}

async function renderModelDownloads(card) {
  try {
    const data = await api("/api/v1/models/downloads");
    const all = data.downloads || [];
    const recent = all.slice(0, 8);
    if (!recent.length) { card.classList.add("hidden"); return; }
    card.classList.remove("hidden");
    card.innerHTML = "";
    card.append(el("h3", { text: "Downloads" }));
    for (const d of recent) {
      const row = el("div", { style: "margin-bottom:12px" });
      const info = el("div", { class: "row" });
      const pct = d.total_bytes > 0
        ? Math.min(100, Math.round(d.completed_bytes * 100 / d.total_bytes)) : null;
      const stateCls = d.status === "FAILED" ? "err" : d.status === "COMPLETED" ? "ok"
        : d.status === "CANCELLED" || d.status === "CANCEL_REQUESTED" ? "muted" : "warn";
      info.append(
        el("span", { class: "mono", text: d.provider + " / " + d.model }),
        badge(d.status, stateCls),
        d.operation ? el("span", { class: "sub", text: d.operation }) : el("span"),
        pct == null ? el("span", { class: "mono", text: fmtBytes(d.completed_bytes) })
          : el("span", { class: "mono", text: pct + "% (" + fmtBytes(d.completed_bytes) + " / " + fmtBytes(d.total_bytes) + ")" }),
      );
      const bar = el("div", { class: "progress" },
        el("div", { style: "width:" + (pct == null ? 0 : pct) + "%" }));
      row.append(info, bar);
      if (d.error) row.append(el("div", { class: "sub", style: "color:var(--err)", text: d.error }));
      if (!["COMPLETED", "FAILED", "CANCELLED"].includes(d.status)) {
        row.append(el("button", {
          class: "btn small danger", text: "Cancel",
          onclick: async () => {
            try { await api("/api/v1/models/downloads/" + encodeURIComponent(d.job_id), { method: "DELETE" }); }
            catch (e) { toast(e.message, "err"); }
          },
        }));
      }
      card.append(row);
    }
  } catch (e) {
    card.innerHTML = "";
    card.classList.remove("hidden");
    card.append(el("h3", { text: "Downloads" }),
      el("div", { class: "empty", text: "Download list unavailable: " + e.message }));
  }
}

async function renderInstalled(card) {
  card.innerHTML = "";
  const mm = state.modelManager;
  card.append(el("h3", { text: "Installed Models — " + mm.provider }));
  try {
    const data = await api("/api/v1/models/providers/" + encodeURIComponent(mm.provider) + "/installed");
    const models = data.models || [];
    if (!models.length) {
      card.append(el("div", { class: "empty", text: "No installed models on this provider." }));
      return;
    }
    const table = el("table");
    table.append(tr(["Model", "Size", "Digest", "Actions"], true));
    for (const m of models) {
      const row = tr([
        el("span", { class: "mono", text: m.id }),
        fmtBytes(m.size_bytes),
        el("span", { class: "mono", text: (m.digest || "").slice(0, 16) + ((m.digest || "").length > 16 ? "…" : "") }),
        el("button", {
          class: "btn small danger", text: "Remove",
          onclick: async () => {
            if (!confirm("Remove " + m.id + " from " + mm.provider + "?")) return;
            try {
              await api("/api/v1/models/providers/" + encodeURIComponent(mm.provider) + "/"
                + encodeURIComponent(m.id), { method: "DELETE" });
              toast("Removed " + m.id);
              renderInstalled(card);
            } catch (e) { toast(e.message, "err"); }
          },
        }),
      ]);
      table.append(row);
    }
    card.append(table);
  } catch (e) {
    card.append(el("div", { class: "empty", text: "Installed models unavailable: " + e.message }));
  }
}

async function renderDownloads(card) {  card.innerHTML = "";
  try {
    const data = await api("/api/v1/downloads");
    const active = (data.downloads || []).filter(d => d.status === "running");
    const recent = (data.downloads || []).slice(-5).reverse();
    if (!recent.length) { card.classList.add("hidden"); return; }
    card.classList.remove("hidden");
    card.append(el("h3", { text: "Downloads" }));
    for (const d of recent) {
      const row = el("div", { style: "margin-bottom:10px" });
      const info = el("div", { class: "row" });
      const pct = d.total_bytes > 0 ? Math.min(100, Math.round(d.completed_bytes * 100 / d.total_bytes)) : null;
      info.append(
        el("span", { class: "mono", text: d.name + " — " + d.status }),
        pct == null ? el("span", { class: "mono", text: fmtBytes(d.completed_bytes) })
          : el("span", { class: "mono", text: pct + "% (" + fmtBytes(d.completed_bytes) + " / " + fmtBytes(d.total_bytes) + ")" }),
        d.status === "running" && d.status === "running"
          ? el("button", { class: "btn small danger", text: "Cancel", onclick: () => cancelDownload(d.id) })
          : el("span", { class: "spacer" }),
      );
      row.append(info);
      if (pct != null) {
        row.append(el("div", { class: "progress" }, el("div", { style: "width:" + pct + "%" })));
      }
      if (d.error) row.append(el("div", { class: "mono", style: "color:var(--err);font-size:12px", text: d.error }));
      card.append(row);
    }
  } catch { card.classList.add("hidden"); }
}

function doSearch(query) {
  if (!query.trim()) return toast("Enter a search query", "err");
  openModal("Catalog search: " + query, async (body) => {
    body.append(el("div", { class: "empty", text: "Searching…" }));
    try {
      const data = await api("/api/v1/catalog/search?q=" + encodeURIComponent(query) + "&limit=25");
      body.innerHTML = "";
      if (data.errors?.length) {
        for (const e of data.errors) body.append(el("div", { style: "color:var(--warn);font-size:12px;margin-bottom:6px", text: e }));
      }
      if (!data.models?.length) { body.append(el("div", { class: "empty", text: "No results" })); return; }
      const table = el("table");
      table.append(tr(["Model", "Source", "Description", ""], true));
      for (const m of data.models) {
        table.append(tr([
          el("span", { class: "mono", text: m.id }),
          m.source, m.description || "",
          el("button", { class: "btn small primary", text: "Pull", onclick: () => pullModel((m.source === "ollama" ? "" : "hf:") + m.id) }),
        ]));
      }
      body.append(table);
    } catch (e) { body.innerHTML = ""; body.append(el("div", { class: "empty", text: e.message })); }
  });
  $("#modal-ok").classList.add("hidden");
}

function promptPull() {
  const body = openModal("Pull model", (b) => {
    b.append(
      el("label", { text: "Model reference — NAME[:TAG] (Ollama Library) or hf:REPO/FILE.gguf (Hugging Face)" }),
      el("input", { id: "pull-ref", placeholder: "llama3.2:1b or hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf" }),
    );
  });
  $("#modal-ok").onclick = () => pullModel($("#pull-ref").value.trim());
}

async function pullModel(ref) {
  if (!ref) return toast("Enter a model reference", "err");
  closeModal();
  try {
    const data = await api("/api/v1/catalog/pull", { body: { name: ref } });
    toast("Pull started: " + ref + " (id " + (data.download_id || "").slice(0, 8) + ")");
    setTimeout(render, 800);
  } catch (e) { toast("Pull failed: " + e.message, "err"); }
}

function promptImport() {
  const body = openModal("Import GGUF/EVO model", (b) => {
    b.append(
      el("label", { text: "Local file path" }),
      el("input", { id: "import-path", placeholder: "/home/user/models/llama3.gguf" }),
      el("label", { text: "Model id (optional)" }),
      el("input", { id: "import-id", placeholder: "llama3" }),
    );
  });
  $("#modal-ok").onclick = async () => {
    try {
      const data = await api("/api/v1/catalog/import", {
        body: { path: $("#import-path").value.trim(), id: $("#import-id").value.trim() },
      });
      closeModal();
      toast("Imported: " + data.model.id);
      render();
    } catch (e) { toast("Import failed: " + e.message, "err"); }
  };
}

async function startModel(id, backend) {
  try {
    await api("/api/v1/models/start", { body: { id, backend: backend || "" } });
    toast("Started: " + id);
    render();
  } catch (e) { toast("Start failed: " + e.message, "err"); }
}

async function stopModel(id) {
  try {
    await api("/api/v1/models/stop", { body: { id } });
    toast("Stopped: " + id);
    render();
  } catch (e) { toast("Stop failed: " + e.message, "err"); }
}

async function removeModel(id) {
  if (!confirm("Remove model '" + id + "'? This deletes the downloaded artifact.")) return;
  try {
    await api("/api/v1/catalog/models/" + encodeURIComponent(id), { method: "DELETE" });
    toast("Removed: " + id);
    render();
  } catch (e) { toast("Remove failed: " + e.message, "err"); }
}

async function verifyModel(id) {
  try {
    const data = await api("/api/v1/catalog/verify", { body: { id } });
    toast(data.verified ? id + ": artifact verified" : id + ": verification FAILED", data.verified ? "ok" : "err");
  } catch (e) { toast("Verify failed: " + e.message, "err"); }
}

async function showModelDetails(id) {
  const body = openModal("Model: " + id, (b) => { b.append(el("div", { class: "empty", text: "Loading…" })); });
  try {
    const data = await api("/api/v1/catalog/models/" + encodeURIComponent(id));
    const entries = {};
    for (const [k, v] of Object.entries(data)) {
      if (typeof v !== "object") entries[k] = v;
    }
    body.innerHTML = "";
    body.append(kvCard("Details", entries).firstChild);
  } catch (e) { body.innerHTML = ""; body.append(el("div", { class: "empty", text: e.message })); }
}

async function cancelDownload(id) {
  try { await api("/api/v1/downloads/cancel", { body: { id } }); toast("Cancelled"); }
  catch (e) { toast(e.message, "err"); }
}

// ---------------------------------------------------------------- providers

async function renderProviders(main) {
  const data = await api("/api/v1/providers");
  state.providers = data.providers || [];
  main.append(el("h1", { text: "Providers" }),
    el("p", { class: "sub", text: "Local and cloud LLM providers. API keys are stored server-side and never shown." }));
  main.append(el("button", { class: "btn primary", text: "Add provider", onclick: () => promptAddProvider() }));
  const grid = el("div", { class: "grid cols-2", style: "margin-top:16px" });
  if (!state.providers.length) {
    grid.append(el("div", { class: "card empty", text: "No providers configured yet." }));
  }
  for (const p of state.providers) {
    const card = el("div", { class: "card" });
    const head = el("div", { class: "row" });
    head.append(
      el("h3", { text: p.id + " — " + p.type, style: "margin:0;flex:1" }),
      p.enabled ? badge("enabled", "ok") : badge("disabled", "muted"),
      p.connected ? badge("connected", "ok") : badge(p.last_test_ok ? "stale" : "untested", "muted"),
    );
    card.append(head);
    const dl = el("dl", { class: "kv" });
    dl.append(
      el("dt", { text: "Name" }), el("dd", { text: p.name || "—" }),
      el("dt", { text: "Base URL" }), el("dd", { text: p.base_url || "—" }),
      el("dt", { text: "API key" }), el("dd", { text: p.api_key_configured ? "configured" : "not set" }),
      el("dt", { text: "Local" }), el("dd", { text: p.local ? "yes" : "no" }),
      el("dt", { text: "Last test" }), el("dd", { text: p.last_test_at ? new Date(p.last_test_at).toLocaleString() + " — " + (p.last_test_message || "") : "never" }),
      el("dt", { text: "Models" }), el("dd", { text: (p.models || []).slice(0, 12).join(", ") + ((p.models || []).length > 12 ? "…" : "") || "—" }),
    );
    card.append(dl);
    card.append(el("div", { class: "row", style: "margin-top:12px;gap:6px" },
      el("button", { class: "btn small primary", text: "Test connection", onclick: () => testProvider(p.id) }),
      el("button", { class: "btn small", text: "Refresh models", onclick: () => refreshProviderModels(p.id) }),
      el("button", { class: "btn small", text: "Edit", onclick: () => promptEditProvider(p) }),
      el("button", { class: "btn small danger", text: "Remove", onclick: () => removeProvider(p.id) }),
    ));
    grid.append(card);
  }
  main.append(grid);
}

function providerForm(b, p) {
  b.append(
    el("label", { text: "Provider id" }),
    el("input", { id: "p-id", value: p?.id || "", placeholder: "deepseek" }),
    el("label", { text: "Type" }),
    el("select", { id: "p-type" }, ...["deepseek", "openai", "anthropic", "gemini", "openai-compatible", "ollama", "llama.cpp"].map(t =>
      el("option", { value: t, text: t, ...(p?.type === t ? { selected: "true" } : {}) }))),
    el("label", { text: "Display name" }),
    el("input", { id: "p-name", value: p?.name || "", placeholder: "DeepSeek" }),
    el("label", { text: "Base URL (leave blank for the provider default)" }),
    el("input", { id: "p-url", value: p?.base_url || "", placeholder: "https://api.deepseek.com/v1" }),
    el("label", { text: "API key" }),
    el("input", { id: "p-key", type: "password", placeholder: p?.api_key_configured ? "leave blank to keep current key" : "sk-…" }),
    el("label", {}, el("input", { id: "p-enabled", type: "checkbox", style: "width:auto", ...(p?.enabled !== false ? { checked: "true" } : {}) }),
      el("span", { text: " Enabled", style: "margin-left:6px" })),
  );
}

function promptAddProvider() {
  const body = openModal("Add provider", (b) => providerForm(b));
  $("#modal-ok").onclick = async () => {
    try {
      await api("/api/v1/providers", { body: {
        id: $("#p-id").value.trim(),
        type: $("#p-type").value,
        name: $("#p-name").value.trim(),
        base_url: $("#p-url").value.trim(),
        api_key: $("#p-key").value,
        enabled: $("#p-enabled").checked,
      }});
      closeModal();
      toast("Provider added");
      render();
    } catch (e) { toast("Failed: " + e.message, "err"); }
  };
}

function promptEditProvider(p) {
  const body = openModal("Edit provider: " + p.id, (b) => providerForm(b, p));
  $("#modal-ok").onclick = async () => {
    try {
      await api("/api/v1/providers/" + encodeURIComponent(p.id), { method: "PUT", body: {
        name: $("#p-name").value.trim(),
        base_url: $("#p-url").value.trim(),
        api_key: $("#p-key").value,
        enabled: $("#p-enabled").checked,
      }});
      closeModal();
      toast("Provider updated");
      render();
    } catch (e) { toast("Failed: " + e.message, "err"); }
  };
}

async function testProvider(id) {
  toast("Testing " + id + "…");
  try {
    const data = await api("/api/v1/providers/" + encodeURIComponent(id) + "/test", { body: {} });
    toast(data.ok ? (id + ": " + data.message + " (" + data.elapsed_ms + " ms)") : (id + ": " + data.message), data.ok ? "ok" : "err");
    render();
  } catch (e) { toast("Test failed: " + e.message, "err"); }
}

async function refreshProviderModels(id) {
  try {
    const data = await api("/api/v1/providers/" + encodeURIComponent(id) + "/models");
    toast(id + ": " + (data.models || []).length + " models");
    render();
  } catch (e) { toast(e.message, "err"); }
}

async function removeProvider(id) {
  if (!confirm("Remove provider '" + id + "'?")) return;
  try {
    await api("/api/v1/providers/" + encodeURIComponent(id), { method: "DELETE" });
    toast("Provider removed");
    render();
  } catch (e) { toast("Failed: " + e.message, "err"); }
}

// ---------------------------------------------------------------- runtime

async function renderRuntime(main) {
  const [status, devices, modelsData] = await Promise.all([
    api("/api/v1/status"),
    api("/api/v1/devices"),
    api("/api/v1/catalog/models"),
  ]);
  main.append(el("h1", { text: "Runtime" }),
    el("p", { class: "sub", text: "Loaded models, compute backends and resource allocation" }));
  main.append(el("div", { class: "grid cols-4" },
    statCard(String(status.models_loaded || 0), "Loaded models"),
    statCard(String(status.sessions_active || 0), "Active sessions"),
    statCard(String((devices.devices || []).filter(d => d.available).length) + " / " + String((devices.devices || []).length), "Backends available"),
    statCard(status.running ? "running" : "stopped", "Runtime", status.running ? "ok" : "err"),
  ));

  const card = el("div", { class: "card" });
  card.append(el("h3", { text: "Loaded model instances" }));
  const loaded = (modelsData.models || []).filter(m => m.loaded);
  if (!loaded.length) card.append(el("div", { class: "empty", text: "No models loaded. Start a model from the Models page." }));
  else {
    const table = el("table");
    table.append(tr(["Model", "State", "Backend", "Sessions", "Requests", "Actions"], true));
    for (const m of loaded) {
      table.append(tr([
        el("span", { class: "mono", text: m.id }),
        stateBadge(true, m.state),
        m.backend || "—",
        String(m.active_sessions || 0),
        String(m.total_requests || 0),
        el("button", { class: "btn small", text: "Stop", onclick: () => stopModel(m.id) }),
      ]));
    }
    card.append(table);
  }
  main.append(card);

  const devCard = el("div", { class: "card" });
  devCard.append(el("h3", { text: "Compute devices" }));
  if (!devices.devices?.length) devCard.append(el("div", { class: "empty", text: "No devices reported." }));
  else {
    const table = el("table");
    table.append(tr(["Type", "Index", "Name", "Memory total", "Memory free", "Available", "Driver"], true));
    for (const d of devices.devices) {
      table.append(tr([
        el("span", { class: "mono", text: d.type }),
        String(d.index), d.name || "—",
        fmtBytes(d.total_memory_bytes), fmtBytes(d.free_memory_bytes),
        d.available ? badge("yes", "ok") : badge("no", "err"),
        d.driver_version || "—",
      ]));
    }
    devCard.append(table);
  }
  main.append(devCard);
}

// ---------------------------------------------------------------- playground

async function renderPlayground(main) {
  try {
    const data = await api("/v1/models");
    state.routable = (data.data || []).map(m => m.id);
  } catch { state.routable = state.models.map(m => m.id); }
  main.append(el("h1", { text: "Playground" }),
    el("p", { class: "sub", text: "Interactive chat with streaming. Compare two models side by side." }));
  const controls = el("div", { class: "row", style: "margin-bottom:14px" });
  const modelSelect = el("select", { id: "chat-model" },
    ...(state.routable.length ? state.routable.map(m => el("option", { value: m, text: m }))
      : [el("option", { value: "", text: "no models available" })]));
  if (!state.routable.length) modelSelect.disabled = true;
  const temp = el("input", { id: "chat-temp", type: "number", step: "0.1", min: "0", max: "2", value: "0.7", style: "max-width:90px", title: "temperature" });
  const maxTokens = el("input", { id: "chat-max", type: "number", min: "1", value: "256", style: "max-width:90px", title: "max tokens" });
  controls.append(
    el("label", { text: "Model", style: "margin:0" }, modelSelect),
    el("label", { text: "Temp", style: "margin:0" }, temp),
    el("label", { text: "Max tokens", style: "margin:0" }, maxTokens),
    el("div", { class: "spacer" }),
  );
  main.append(controls);

  const chatBox = chatPanel("chat", state.chat);
  main.append(chatBox);

  const cmp = el("div", { class: "card", style: "margin-top:16px" });
  cmp.append(el("h3", { text: "Model comparison" }));
  const cmpModels = el("div", { class: "compare", style: "margin-bottom:10px" },
    modelPicker("cmp-a", state.compare.a),
    modelPicker("cmp-b", state.compare.b),
  );
  cmp.append(cmpModels);
  const cmpBtn = el("div", { class: "row", style: "margin-top:8px" },
    el("input", { id: "cmp-prompt", placeholder: "Prompt for both models…" }),
    el("button", { class: "btn primary", text: "Compare", onclick: runCompare }),
  );
  cmp.append(cmpBtn);
  const cmpResults = el("div", { class: "compare", style: "margin-top:10px" },
    chatPanel("cmp-res-a", state.compare.a, true),
    chatPanel("cmp-res-b", state.compare.b, true),
  );
  cmp.append(cmpResults);
  main.append(cmp);
}

function modelPicker(id, slot) {
  const select = el("select", { id },
    ...(state.routable.length ? state.routable.map(m => el("option", { value: m, text: m }))
      : [el("option", { value: "", text: "no models" })]));
  select.value = slot.model;
  select.addEventListener("change", () => { slot.model = select.value; });
  return select;
}

function chatPanel(id, slot, readOnly = false) {
  const box = el("div", { class: "card" });
  const messages = el("div", { class: "messages", id: id + "-messages", style: "height:260px" });
  box.append(messages);
  if (!readOnly) {
    const input = el("div", { class: "chat-input" },
      el("textarea", { id: id + "-input", placeholder: "Type a message…", rows: "2" }),
      el("button", { class: "btn primary", text: "Send", style: "max-width:110px", onclick: () => sendChat(id, slot) }),
    );
    box.append(input);
    $(id + "-input", box).addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(id, slot); }
    });
  }
  return box;
}

function appendMsg(id, role, text, meta) {
  const box = $("#" + id + "-messages");
  const msg = el("div", { class: "msg " + role }, el("span", { text }));
  if (meta) msg.append(el("div", { class: "meta", text: meta }));
  box.append(msg);
  box.scrollTop = box.scrollHeight;
  return msg;
}

async function sendChat(id, slot) {
  const input = $("#" + id + "-input");
  const text = input.value.trim();
  if (!text || slot.streaming) return;
  input.value = "";
  const model = slot.model || $("#chat-model")?.value || state.routable[0];
  if (!model) return toast("No model selected", "err");
  slot.messages.push({ role: "user", content: text });
  appendMsg(id, "user", text);
  appendMsg(id, "assistant", "");
  const out = $("#" + id + "-messages").lastChild;
  slot.streaming = true;
  const started = Date.now();
  try {
    const res = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        model,
        messages: slot.messages,
        stream: true,
        max_tokens: parseInt($("#chat-max")?.value || "256", 10),
        temperature: parseFloat($("#chat-temp")?.value || "0.7"),
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", full = "", done = false;
    while (!done) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const chunk = JSON.parse(line.slice(5).trim());
          if (chunk.content) { full += chunk.content; out.firstChild.textContent = full; }
          if (chunk.done) {
            done = true;
            out.append(el("div", { class: "meta", text: (chunk.finish_reason || "stop") + " · "
              + (chunk.prompt_tokens || 0) + "→" + (chunk.completion_tokens || 0) + " tok · "
              + (Date.now() - started) + " ms" }));
          }
          if (chunk.error) out.append(el("div", { class: "meta", text: "error: " + chunk.error }));
        }
      }
    }
    slot.messages.push({ role: "assistant", content: full });
    out.scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    out.firstChild.textContent = "(error: " + e.message + ")";
  } finally {
    slot.streaming = false;
  }
}

async function runCompare() {
  const prompt = $("#cmp-prompt").value.trim();
  if (!prompt) return toast("Enter a prompt", "err");
  const a = state.compare.a.model || state.routable[0];
  const b = state.compare.b.model || state.routable[0];
  if (!a || !b) return toast("Select two models", "err");
  const am = $("#cmp-res-a-messages"), bm = $("#cmp-res-b-messages");
  am.innerHTML = ""; bm.innerHTML = "";
  appendMsg("cmp-res-a", "user", prompt, a);
  appendMsg("cmp-res-b", "user", prompt, b);
  await Promise.all([streamOnce(a, prompt, "cmp-res-a"), streamOnce(b, prompt, "cmp-res-b")]);
}

async function streamOnce(model, prompt, id) {
  const started = Date.now();
  let full = "", meta = "";
  const res = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], stream: true, max_tokens: 256 }),
  });
  if (!res.ok) { appendMsg(id, "assistant", "HTTP " + res.status); return; }
  appendMsg(id, "assistant", "");
  const out = $("#" + id + "-messages").lastChild;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", done = false;
  while (!done) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const chunk = JSON.parse(line.slice(5).trim());
        if (chunk.content) { full += chunk.content; out.firstChild.textContent = full; }
        if (chunk.done) { done = true; meta = (chunk.finish_reason || "stop") + " · " + (Date.now() - started) + " ms"; }
      }
    }
  }
  out.append(el("div", { class: "meta", text: meta }));
}

// ---------------------------------------------------------------- diagnostics

async function renderDiagnostics(main) {
  main.append(el("h1", { text: "Diagnostics" }),
    el("p", { class: "sub", text: "System, runtime and provider state — genuine values only" }));
  const tabs = el("div", { class: "tabs" },
    el("button", { class: "btn active", text: "Logs", onclick: (e) => diagTab("logs", e.target) }),
    el("button", { class: "btn", text: "System", onclick: (e) => diagTab("system", e.target) }),
    el("button", { class: "btn", text: "API endpoints", onclick: (e) => diagTab("api", e.target) }),
  );
  main.append(tabs);
  const container = el("div", { id: "diag-body" });
  main.append(container);
  diagTab("logs", $(".tabs .btn.active"));
}

function diagTab(tab, btn) {
  document.querySelectorAll(".tabs .btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const container = $("#diag-body");
  container.innerHTML = "";
  state.timers.forEach(clearInterval);
  state.timers = [];
  if (tab === "logs") {
    const logs = el("div", { class: "logs", id: "log-view" });
    container.append(logs);
    const refresh = async () => {
      try {
        const data = await api("/api/v1/logs?source=all&tail=300");
        logs.innerHTML = "";
        for (const line of data.logs || []) {
          logs.append(el("div", { class: "log-line" },
            el("span", { class: "lvl " + line.level, text: line.level.padEnd(7) }),
            el("span", { class: "mono", text: new Date(line.timestamp_ms).toLocaleTimeString() + " " }),
            el("span", { text: "[" + line.logger + "] " + line.message })));
        }
        logs.scrollTop = logs.scrollHeight;
      } catch { /* auth or transient */ }
    };
    refresh();
    state.timers.push(setInterval(refresh, 3000));
  } else if (tab === "system") {
    (async () => {
      try {
        const data = await api("/api/v1/diagnostics");
        container.append(el("div", { class: "grid cols-2" },
          kvCard("Health", { status: data.health?.status, checks: (data.health_checks || []).join("; ") || "none" }),
          kvCard("Runtime status", {
            running: data.status?.running,
            uptime: fmtDuration(data.status?.uptime_ms),
            requests: data.status?.total_requests,
            tokens: data.status?.total_generated_tokens,
            "tokens/sec": data.status?.tokens_per_second,
          }),
          kvCard("CPU", data.hardware?.cpu || {}),
          kvCard("RAM", data.hardware?.ram || {}),
        ));
        const gpus = data.hardware?.gpus || [];
        if (gpus.length) container.append(el("div", { class: "grid cols-2" }, ...gpus.map(g => kvCard("GPU " + g.index, g))));
      } catch (e) { container.append(el("div", { class: "empty", text: e.message })); }
    })();
  } else if (tab === "api") {
    (async () => {
      const endpoints = [
        ["GET  /api/v1/health", "/api/v1/health"],
        ["GET  /api/v1/status", "/api/v1/status"],
        ["GET  /api/v1/devices", "/api/v1/devices"],
        ["GET  /api/v1/metrics", "/api/v1/metrics"],
        ["GET  /api/tags", "/api/tags"],
        ["GET  /api/version", "/api/version"],
        ["GET  /api/ps", "/api/ps"],
        ["GET  /v1/models", "/v1/models"],
        ["GET  /api/v1/catalog/models", "/api/v1/catalog/models"],
        ["GET  /api/v1/downloads", "/api/v1/downloads"],
        ["GET  /api/v1/providers", "/api/v1/providers"],
        ["GET  /api/v1/dashboard", "/api/v1/dashboard"],
        ["GET  /", "/api/wcc/info"],
      ];
      const table = el("table");
      table.append(tr(["Endpoint", "Description", "Status"], true));
      for (const [desc, path] of endpoints) {
        const row = tr([el("span", { class: "mono", text: desc }), "", "…"]);
        table.append(row);
        api(path).then(() => {
          row.cells[1].textContent = "—";
          row.cells[2].innerHTML = "";
          row.cells[2].append(badge("200 OK", "ok"));
        }).catch(e => {
          row.cells[2].innerHTML = "";
          row.cells[2].append(badge(String(e.message).slice(0, 60), "err"));
        });
      }
      container.append(el("div", { class: "card" }, table));
    })();
  }
}

// ---------------------------------------------------------------- boot

async function boot() {
  try {
    const info = await fetch("/api/wcc/info").then(r => r.json());
    if (info.auth_required) $("#token-btn").classList.remove("hidden");
    if (info.auth_required && !info.auth_ok && !state.token) promptToken();
  } catch { /* server not reachable */ }
  $("#token-btn").addEventListener("click", promptToken);
  // Theme toggle: UPM design standard light/dark pair (default dark).
  (function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("enr.theme"); } catch (e) { /* ignore */ }
    if (saved === "light" || saved === "dark") {
      document.documentElement.dataset.theme = saved;
    }
    const btn = $("#theme-btn");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        try { localStorage.setItem("enr.theme", next); } catch (e) { /* ignore */ }
      });
    }
  })();
  setInterval(ping, 5000);
  ping();
  window.addEventListener("hashchange", () => {
    const page = location.hash.replace("#", "") || "dashboard";
    if (state.page !== page) { state.page = page; render(); }
  });
  state.page = location.hash.replace("#", "") || "dashboard";
  render();
}

async function ping() {
  const dot = $("#conn-dot");
  try {
    const res = await fetch("/api/v1/health");
    dot.className = "dot " + (res.ok ? "ok" : "");
  } catch { dot.className = "dot"; }
}

document.addEventListener("DOMContentLoaded", boot);
