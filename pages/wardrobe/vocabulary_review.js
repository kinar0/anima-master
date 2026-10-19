const bridge = window.AstrBotPluginPage;
const el = (id) => document.getElementById(id);
const details = el("review-details");
const drafts = new Map();
const labels = { pending: "未审核", keep_current: "保留现值", proposed: "采纳待验收", modified: "修改待验收", deferred: "暂缓", rejected: "拒绝" };
let snapshot = null;
let page = 1;
let requestVersion = 0;
let saving = false;

function node(tag, text, className = "") {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  value.className = className;
  return value;
}
function status(text, error = false) { el("review-status").textContent = text; el("review-status").classList.toggle("is-error", error); }
const split = (text) => [...new Set(text.split(/[,，\n]+/).map((a) => a.trim()).filter(Boolean))];

async function load() {
  if (!bridge) { status("请在 AstrBot WebUI 中打开审核资料。", true); return; }
  const version = ++requestVersion;
  const params = { query: el("review-query").value.trim(), state: el("review-filter").value, page: String(page), page_size: "30" };
  status("正在读取候选资料…");
  try {
    await bridge.ready();
    if (version !== requestVersion) return;
    const data = await bridge.apiGet("vocabulary-review", params);
    if (version !== requestVersion) return;
    snapshot = data;
    page = data.page;
    render();
    status(`显示 ${data.total} 个概念中的第 ${data.page} / ${data.pages} 页；审核记录与正式词库分开保存。`);
  } catch (error) { if (version === requestVersion) { page = snapshot?.page || 1; status(`读取失败：${error.message || error}`, true); } }
}

function render() {
  el("review-rows").replaceChildren();
  el("review-page").textContent = `第 ${snapshot.page} / ${snapshot.pages} 页`;
  el("review-prev").disabled = page <= 1;
  el("review-next").disabled = page >= snapshot.pages;
  for (const row of snapshot.rows) {
    const card = node("article", undefined, "review-card");
    card.append(node("h3", row.tag), node("span", labels[row.decision.state] || row.decision.state, "pill"));
    card.append(node("p", row.definition || "来源暂无释义，保留待核对。"));
    if (row.trap) card.append(node("p", `易误读：${row.trap}`, "review-warning"));
    for (const pair of row.contrasts || []) card.append(node("p", `与 ${pair[0]} 区别：${pair[1]}`));
    card.append(node("p", `分类：${row.categories.join(" / ") || "未分类"}；本地目录：${row.local_coverage === "exact_present" ? "已有精确证据" : row.local_coverage === "not_checked" ? "尚未核对" : "当前未命中"}`, "mapping-note"));
    card.append(node("p", `当前词库：${row.current_aliases.join("、") || "无对应映射"}`));
    for (const issue of row.issues || []) card.append(node("p", `${issue.alias}：${issue.reason}${issue.counterexample ? `；邻近反例：${issue.counterexample}` : ""}`, "review-warning"));
    if (row.positive_examples.length) card.append(node("p", `正例：${row.positive_examples.join("；")}`));
    if (row.counterexamples.length) card.append(node("p", `反例：${row.counterexamples.join("；")}`));
    if (row.prior_basis) card.append(node("p", `首轮审查：${row.prior_basis}`, "mapping-note"));
    if (row.search_reason) card.append(node("p", row.search_reason, "mapping-note"));
    const draft = drafts.get(row.tag) || { aliases: row.decision.aliases.join("，"), note: row.decision.note || "" };
    const aliasInput = node("textarea");
    aliasInput.value = draft.aliases;
    aliasInput.placeholder = "选择现成别名，或填写修改后的自然表达，以逗号分隔";
    aliasInput.setAttribute("aria-label", `${row.tag} 待验收别名`);
    const notes = node("textarea");
    notes.value = draft.note;
    notes.maxLength = 2000;
    notes.placeholder = "审核依据或暂缓原因";
    notes.setAttribute("aria-label", `${row.tag} 审核备注`);
    const remember = () => drafts.set(row.tag, { aliases: aliasInput.value, note: notes.value });
    aliasInput.addEventListener("input", remember);
    notes.addEventListener("input", remember);
    const choices = node("div", undefined, "review-aliases");
    for (const alias of row.aliases) {
      const label = node("label");
      const check = node("input"); check.type = "checkbox";
      check.checked = split(aliasInput.value).includes(alias.alias);
      check.addEventListener("change", () => {
        const values = new Set(split(aliasInput.value));
        if (check.checked) values.add(alias.alias); else values.delete(alias.alias);
        aliasInput.value = [...values].join("，"); remember();
      });
      let text = alias.alias;
      if (alias.mapping_state === "existing") text += " · 已有同值";
      if (alias.mapping_state === "conflict") text += ` · 当前指向 ${alias.current_tag}`;
      if (alias.ambiguous) text += ` · 多义：${alias.owners.join(" / ")}`;
      if (alias.language_note) text += ` · ${alias.language_note}`;
      label.append(check, node("span", text)); choices.append(label);
    }
    card.append(choices, aliasInput, notes);
    const actions = node("div", undefined, "review-actions");
    for (const [state, label] of [["keep_current", "保留现值"], ["proposed", "采纳为待验收"], ["modified", "保存修改为待验收"], ["deferred", "暂缓"], ["rejected", "拒绝"]]) {
      const button = node("button", label, "button ghost"); button.type = "button";
      button.addEventListener("click", async () => {
        if (saving) return;
        const payload = { base_revision: snapshot.revision, tag: row.tag, state, aliases: split(aliasInput.value), note: notes.value };
        const submitted = { aliases: aliasInput.value, note: notes.value };
        remember(); saving = true; button.disabled = true; ++requestVersion;
        try {
          const result = await bridge.apiPost("vocabulary-review/decision", payload);
          snapshot.revision = result.revision;
          const current = drafts.get(row.tag);
          if (current?.aliases === submitted.aliases && current?.note === submitted.note) drafts.delete(row.tag);
          await load(); status(result.message || "审核记录已保存，未应用词库。");
        } catch (error) { status(`保存失败：${error.message || error}；编辑内容仍保留。`, true); }
        finally { saving = false; button.disabled = false; }
      });
      actions.append(button);
    }
    card.append(actions); el("review-rows").append(card);
  }
  if (!snapshot.rows.length) el("review-rows").append(node("p", "没有符合条件的候选资料。"));
}

details.addEventListener("toggle", () => { if (details.open && !snapshot) load(); });
el("review-search").addEventListener("submit", (event) => { event.preventDefault(); page = 1; load(); });
el("review-filter").addEventListener("change", () => { page = 1; load(); });
el("review-refresh").addEventListener("click", load);
el("review-prev").addEventListener("click", () => { page -= 1; load(); });
el("review-next").addEventListener("click", () => { page += 1; load(); });
window.addEventListener("beforeunload", (event) => { if (drafts.size) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("anima:before-page-leave", (event) => { if (drafts.size || saving) event.preventDefault(); });
