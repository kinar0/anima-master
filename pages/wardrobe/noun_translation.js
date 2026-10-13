const bridge = window.AstrBotPluginPage;
const byId = (id) => document.getElementById(id);
const provider = byId("batch-provider");
const status = byId("batch-status");
const rowsNode = byId("batch-rows");
const filter = byId("batch-filter");
let snapshot = null;
let timer = null;
let loading = null;
const details = byId("batch-details");
const pagination = new window.WardrobePagination([byId("batch-pagination")], render);

function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("is-error", error);
}

function cell(tr, value) {
  const td = document.createElement("td");
  td.textContent = value;
  tr.append(td);
}

function rowStatus(row) {
  const details = row.alias_statuses.map((item) => `${item.alias}：${item.reason}`);
  if (!details.length) details.push(row.approved_aliases === null ? "待模型复核" : "初译未提供中文别名，未导入");
  return `${row.needs_manual_review ? "需人工复核；" : ""}${details.join("；")}`;
}

function filteredRows() {
  const rows = snapshot?.rows || [];
  switch (filter.value) {
    case "review": return rows.filter((row) => row.needs_manual_review);
    case "ready": return rows.filter((row) => row.ready);
    case "existing": return rows.filter((row) => row.in_lexicon);
    case "pending": return rows.filter((row) => row.approved_aliases === null);
    case "conflict": return rows.filter((row) => row.alias_statuses.some((item) => item.state === "conflict"));
    case "rejected": return rows.filter((row) => row.alias_statuses.some((item) => item.state === "rejected") || (row.approved_aliases !== null && !row.aliases.length));
    default: return rows;
  }
}

function render() {
  if (!snapshot) return;
  byId("batch-translated").textContent = `${snapshot.translated} / ${snapshot.total}`;
  byId("batch-verified").textContent = `${snapshot.verified} / ${snapshot.translated}`;
  byId("batch-importable").textContent = snapshot.importable;
  byId("batch-review-count").textContent = snapshot.manual_review;
  byId("batch-existing").textContent = snapshot.existing;
  byId("batch-enabled").checked = snapshot.enabled !== false;
  byId("batch-enabled").disabled = false;
  const running = snapshot.job.state === "running";
  byId("batch-translate").disabled = running || snapshot.translated >= snapshot.total;
  byId("batch-verify").disabled = running || snapshot.verified >= snapshot.translated;
  byId("batch-import").disabled = running || snapshot.importable === 0;
  const message = snapshot.job.message || (snapshot.translated ? "翻译断点已加载；可继续下一批或导入复核通过的别名。" : "尚无翻译断点。");
  setStatus(snapshot.enabled === false ? `名词映射已停用，不参与生图。${message}` : message, snapshot.job.state === "error");

  const visible = filteredRows();
  const { start, end } = pagination.update(visible.length, snapshot.rows.length);
  rowsNode.replaceChildren();
  for (const row of visible.slice(start, end)) {
    const tr = document.createElement("tr");
    cell(tr, row.tag);
    cell(tr, row.aliases.join("、") || "—");
    cell(tr, row.approved_aliases === null ? "待模型复核" : row.approved_aliases.join("、") || "无通过别名");
    cell(tr, rowStatus(row));
    rowsNode.append(tr);
  }
  if (!visible.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = "没有符合条件的翻译条目。";
    tr.append(td);
    rowsNode.append(tr);
  }
  clearTimeout(timer);
  if (running) timer = setTimeout(loadStatus, 3000);
}

async function load() {
  if (!bridge) { setStatus("请在 AstrBot WebUI 中打开此页面。", true); return; }
  if (loading) return loading;
  loading = (async () => {
  try {
    snapshot = await bridge.apiGet("noun-translation");
    render();
  } catch (error) {
    setStatus(`读取失败：${error.message || error}`, true);
    clearTimeout(timer);
  }
  })();
  try { await loading; } finally { loading = null; }
}

async function loadStatus() {
  if (!bridge) return;
  try {
    const result = await bridge.apiGet("noun-translation/status");
    byId("batch-enabled").checked = result.enabled !== false;
    byId("batch-enabled").disabled = false;
    const job = result.job || { state: "idle", message: "" };
    if (snapshot) {
      const finished = snapshot.job.state === "running" && job.state !== "running";
      snapshot.job = job;
      snapshot.enabled = result.enabled;
      if (finished && details?.open) { await load(); return; }
      render();
    } else {
      setStatus(job.message || "展开面板或点击刷新后读取翻译进度。", job.state === "error");
    }
    clearTimeout(timer);
    if (job.state === "running") timer = setTimeout(loadStatus, 3000);
  } catch (error) { setStatus(`读取任务状态失败：${error.message || error}`, true); }
}

async function run(action) {
  const providerId = provider.value.trim();
  if (!providerId) { setStatus("请填写 AstrBot 中配置的 DeepSeek provider ID。", true); provider.focus(); return; }
  try {
    const result = await bridge.apiPost("noun-translation/run", { action, provider_id: providerId });
    snapshot.job = result.job;
    render();
  } catch (error) {
    setStatus(`启动失败：${error.message || error}`, true);
  }
}

async function importTerms() {
  if (!byId("dirty-bar").hidden) { setStatus("请先保存上方词库的未保存修改，再导入。", true); return; }
  try {
    const result = await bridge.apiPost("noun-translation/import", { mode: "apply" });
    await load();
    setStatus(`已导入 ${result.imported} 个别名；已有 ${result.existing} 个相同映射；${result.skipped} 个因冲突或格式问题未导入。需处理的条目请筛选“需人工复核”。刷新页面可在“名词翻译”中编辑。`);
  } catch (error) {
    setStatus(`导入失败：${error.message || error}`, true);
  }
}

async function setEnabled() {
  const control = byId("batch-enabled");
  const enabled = control.checked;
  control.disabled = true;
  try {
    const result = await bridge.apiPost("noun-translation/enabled", { enabled });
    if (snapshot) { snapshot.enabled = result.enabled; render(); }
    else control.disabled = false;
    setStatus(result.enabled ? "名词翻译已启用，后续生图请求会使用现有映射。" : "名词翻译已停用，后续生图请求不会使用现有映射。");
  } catch (error) {
    control.checked = snapshot ? snapshot.enabled !== false : !enabled;
    control.disabled = false;
    setStatus(`开关保存失败：${error.message || error}`, true);
  }
}

byId("batch-refresh").addEventListener("click", () => { if (details) details.open = true; load(); });
details?.addEventListener("toggle", () => { if (details.open) load(); });
byId("batch-translate").addEventListener("click", () => run("translate"));
byId("batch-verify").addEventListener("click", () => run("verify"));
byId("batch-import").addEventListener("click", importTerms);
byId("batch-enabled").addEventListener("change", setEnabled);
filter.addEventListener("change", () => { pagination.reset(); render(); });
if (bridge) bridge.ready().then(loadStatus).catch((error) => setStatus(`页面初始化失败：${error.message || error}`, true));
else load();
