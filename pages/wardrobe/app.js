const demoData = {
  revision: "preview000000000000000000000000000000000000000000000000000000000",
  outfits: [
    { key: "amoris", aliases: ["Amoris"], sourceTags: ["amoris_(bang_dream!)"], tags: ["black_corset", "red_shorts", "see-through_sleeves", "masquerade_mask", "red_thighhighs"], appearanceTags: ["long_hair", "blue_eyes"], qualifier: "default", evidence: { sampleMode: "single_character_anchor", sampleCount: 21 } },
    { key: "千早爱音::stage", aliases: ["千早爱音的演出服"], sourceTags: ["chihaya_anon"], tags: ["blue_jacket", "cropped_jacket", "white_skirt", "black_choker"], qualifier: "stage", evidence: { sampleMode: "stage_single_character_anchor", sampleCount: 24 } },
  ],
  outfitSets: [
    { alias: "月之森校服", aliases: ["月之森校服", "tsukinomori school uniform"], tag: "tsukinomori_school_uniform", variant: "default", origin: "configured", profileKey: "" },
    { alias: "羽丘冬季校服", aliases: ["羽丘冬季校服", "haneoka school winter uniform"], tag: "haneoka_school_uniform", componentTags: ["grey_jacket", "white_shirt", "green_skirt", "green_necktie"], variant: "winter", origin: "learned", profileKey: "羽丘冬季校服" },
  ],
  terms: [{ alias: "百褶裙", tag: "pleated_skirt" }, { alias: "舞会面具", tag: "masquerade_mask" }],
};

const fallbackBridge = {
  async ready() {}, getLocale: () => "zh-CN", t: (_key, fallback) => fallback,
  onContext: () => () => {},
  async apiGet() { return structuredClone(demoData); },
  async apiPost(_path, payload) { return { ...structuredClone(payload), revision: crypto.randomUUID().replaceAll("-", "").padEnd(64, "0"), saved: true }; },
};
const bridge = window.AstrBotPluginPage || fallbackBridge;

const META = {
  outfits: { kicker: "CHARACTER WARDROBES", title: "角色自己的变体衣柜", description: "按角色保存任意命名服装；服装名称、所属角色与学习来源分别维护。通用校服和命名套组请放到右侧分类。" },
  outfitSets: { kicker: "NAMED OUTFIT SETS", title: "通用服装套组", description: "可由任意角色穿着的命名套组。每套独立保存 canonical tag、可见组件和别名；不会进入任何角色自己的变体衣柜。" },
  terms: { kicker: "TERM TRANSLATIONS", title: "名词翻译", description: "将常用中文名词稳定映射为 canonical Danbooru tag，命中提示词时直接使用。" },
};
const $ = (selector) => document.querySelector(selector);
const elements = {
  list: $("#list"), empty: $("#empty-state"), search: $("#search-input"), sort: $("#outfit-sort"), sortControl: $("#outfit-sort-control"), add: $("#add-button"), save: $("#save-button"), dirtySave: $("#dirty-save-button"), discard: $("#discard-button"), refresh: $("#refresh-button"), dirtyBar: $("#dirty-bar"), toast: $("#toast"), connection: $("#connection"), sectionKicker: $("#section-kicker"), sectionTitle: $("#section-title"), sectionDescription: $("#section-description"), dataStatus: $("#data-status"), revision: $("#revision-label"), tabs: [...document.querySelectorAll(".tab")],
};
let state = { revision: "", outfits: [], outfitSets: [], terms: [], tab: "outfits", search: "", outfitSort: "added_desc", dirty: false, busy: false };
let pristine = null;
let toastTimer = null;

const clone = (value) => typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
const splitList = (value) => [...new Set(String(value || "").split(/[,，\n]+/).map((item) => item.trim()).filter(Boolean))];
const matches = (item) => !state.search || JSON.stringify(item).toLowerCase().includes(state.search.toLowerCase());
const addedAt = (item) => {
  const value = Number(item?.evidence?.createdAt ?? item?.evidence?.updatedAt ?? 0);
  return Number.isFinite(value) ? value : 0;
};
function node(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; }
function showToast(message) { clearTimeout(toastTimer); elements.toast.textContent = message; elements.toast.hidden = false; toastTimer = setTimeout(() => elements.toast.hidden = true, 3800); }
function errorText(error) { return error instanceof Error ? error.message : String(error); }
function setBusy(value) { state.busy = value; elements.refresh.disabled = value; elements.add.disabled = value; updateDirty(); }
function updateDirty() { const disabled = state.busy || !state.dirty; elements.save.disabled = disabled; elements.dirtySave.disabled = disabled; elements.dirtyBar.hidden = !state.dirty; elements.dataStatus.textContent = state.dirty ? "待保存" : "已同步"; }
function markDirty() { state.dirty = true; updateDirty(); renderCounts(); }
function inputField(label, value, onInput, { textarea = false, placeholder = "", className = "" } = {}) {
  const wrap = node("div", `field ${className}`); const caption = node("label", "", label); const control = node(textarea ? "textarea" : "input"); control.value = value || ""; control.placeholder = placeholder; control.addEventListener("input", () => { onInput(control.value); markDirty(); }); wrap.append(caption, control); return wrap;
}
function chips(values, tone = "") { const wrap = node("div", "tag-preview"); values.slice(0, 8).forEach((value) => wrap.append(node("span", `chip ${tone}`, value))); if (values.length > 8) wrap.append(node("span", "chip", `+${values.length - 8}`)); return wrap; }
function deleteButton(index) { const button = node("button", "delete-button", "×"); button.type = "button"; button.title = "删除条目"; button.addEventListener("click", () => { state[state.tab].splice(index, 1); markDirty(); render(); }); return button; }

function outfitEntry(item, index) {
  const card = node("article", "entry profile");
  const identity = node("div", "field-stack");
  identity.append(inputField("档案名称", item.key, (value) => item.key = value));
  const qualifier = inputField("服装名称 / 变体", item.qualifier, (value) => item.qualifier = value, { placeholder: "Master of Melodia、生日礼服、default…" });
  const suggestions = node("datalist"); suggestions.id = `wardrobe-names-${index}`;
  ["default", "casual", "summer", "winter", "stage"].forEach((value) => { const option = node("option"); option.value = value; suggestions.append(option); });
  qualifier.querySelector("input").setAttribute("list", suggestions.id);
  qualifier.append(suggestions, node("p", "mapping-note", "可自由命名，同一角色可保存多套。旧的五种名称仅作快捷建议。")); identity.append(qualifier);
  identity.append(inputField("所属角色 Tags", (item.ownerTags || item.sourceTags).join(", "), (value) => item.ownerTags = splitList(value), { placeholder: "togawa_sakiko" }));
  const evidence = node("div", "meta-line"); evidence.append(node("span", "pill", item.evidence?.sampleCount ? `${item.evidence.sampleCount} 个聚类样本` : "手动档案")); if (item.evidence?.sampleMode) evidence.append(node("span", "pill", item.evidence.sampleMode)); identity.append(evidence);
  const aliases = inputField("触发别名", item.aliases.join(", "), (value) => item.aliases = splitList(value), { textarea: true, placeholder: "Amoris, 爱音演出服" }); aliases.append(chips(item.aliases, "mint"));
  const tags = node("div", "field-stack wide"); tags.append(inputField("学习来源 Tags", item.sourceTags.join(", "), (value) => item.sourceTags = splitList(value), { placeholder: "amoris_(bang_dream!)" })); const outfitTags = inputField("服装 Tags", item.tags.join(", "), (value) => item.tags = splitList(value), { textarea: true, placeholder: "black_corset, red_shorts, ..." }); outfitTags.append(chips(item.tags)); tags.append(outfitTags); const stableTags = item.appearanceTags || []; const appearance = inputField("Stable Appearance Tags", stableTags.join(", "), (value) => item.appearanceTags = splitList(value), { textarea: true, placeholder: "long_hair, blue_eyes, animal_ears, ..." }); appearance.append(node("p", "mapping-note", "稳定发色、瞳色、耳朵、角、尾巴等身份特征；可修正自动学习结果，也可留空。"), chips(stableTags, "mint")); tags.append(appearance);
  card.append(identity, aliases, tags, deleteButton(index)); return card;
}
function mappingEntry(item, index, isSet) {
  const card = node("article", "entry");
  const setAliases = item.aliases?.length ? item.aliases : [item.alias].filter(Boolean);
  const left = isSet
    ? inputField("触发别名（不限数量）", setAliases.join(", "), (value) => { item.aliases = splitList(value); item.alias = item.aliases[0] || ""; }, { textarea: true, placeholder: "羽丘校服, 羽丘制服, haneoka uniform, haneoka academy uniform" })
    : inputField("中文名词 / 触发词", item.alias, (value) => item.alias = value, { placeholder: "百褶裙" });
  if (isSet) left.append(chips(setAliases, "mint"));
  const right = inputField("Canonical Danbooru Tag", item.tag, (value) => item.tag = value, { placeholder: isSet ? "tsukinomori_school_uniform" : "pleated_skirt", className: "wide" }); right.append(chips(item.tag ? [item.tag] : []));
  const meta = node("div", "field-stack"); if (isSet) { const variantField = node("div", "field"); variantField.append(node("label", "", "套组变体")); const variantSelect = node("select"); [["default","通用 / 未区分"],["casual","官方常服 / Casual"],["summer","夏季"],["winter","冬季"],["stage","演出 / 舞台"]].forEach(([value,label]) => { const option = node("option", "", label); option.value = value; option.selected = (item.variant || "default") === value; variantSelect.append(option); }); variantSelect.addEventListener("change", () => { item.variant = variantSelect.value; markDirty(); }); variantField.append(variantSelect); const components = inputField("套组组件 Tags", (item.componentTags || []).join(", "), (value) => item.componentTags = splitList(value), { textarea: true, placeholder: "grey_jacket, white_shirt, green_skirt, ..." }); components.append(node("p", "mapping-note", "这些组件属于套组本身，可绑定给任何穿着者；不要在这里保存角色外貌。"), chips(item.componentTags || [])); const line = node("div", "meta-line"); line.append(node("span", `pill ${item.origin === "learned" ? "learned" : ""}`, item.origin === "learned" ? "自动学习" : "手动配置")); meta.append(variantField, components, line); } else meta.append(node("p", "mapping-note", "用户提示词出现左侧名词时，右侧 tag 会直接加入 hard tags。"));
  card.append(left, right, meta, deleteButton(index)); return card;
}
function renderCounts() {
  const counts = { outfits: state.outfits.length, outfitSets: state.outfitSets.length, terms: state.terms.length };
  $("#outfit-count").textContent = counts.outfits; $("#set-count").textContent = counts.outfitSets; $("#term-count").textContent = counts.terms; $("#outfit-badge").textContent = counts.outfits; $("#set-badge").textContent = counts.outfitSets; $("#term-badge").textContent = counts.terms;
}
function render() {
  const meta = META[state.tab]; elements.sectionKicker.textContent = meta.kicker; elements.sectionTitle.textContent = meta.title; elements.sectionDescription.textContent = meta.description;
  elements.tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === state.tab)); elements.sortControl.hidden = state.tab !== "outfits"; elements.list.replaceChildren();
  const indexed = state[state.tab].map((item, index) => ({ item, index })).filter(({ item }) => matches(item));
  if (state.tab === "outfits") {
    const direction = state.outfitSort === "added_asc" ? 1 : -1;
    indexed.sort((left, right) => direction * (addedAt(left.item) - addedAt(right.item)) || left.index - right.index);
  }
  indexed.forEach(({ item, index }) => elements.list.append(state.tab === "outfits" ? outfitEntry(item, index) : mappingEntry(item, index, state.tab === "outfitSets")));
  elements.empty.hidden = indexed.length > 0; renderCounts(); updateDirty();
}
function hydrate(data) {
  state.revision = data.revision || ""; state.outfits = clone(data.outfits || []); state.outfitSets = clone(data.outfitSets || []); state.terms = clone(data.terms || []); state.dirty = false; pristine = clone({ revision: state.revision, outfits: state.outfits, outfitSets: state.outfitSets, terms: state.terms }); elements.revision.textContent = `revision ${state.revision.slice(0, 9) || "—"}`; render();
}
async function load() { setBusy(true); try { hydrate(await bridge.apiGet("wardrobe")); elements.connection.classList.add("is-ready"); elements.connection.querySelector("span").textContent = "已连接"; } catch (error) { showToast(`读取失败：${errorText(error)}`); elements.connection.querySelector("span").textContent = "连接失败"; } finally { setBusy(false); } }
async function save() {
  if (!state.dirty || state.busy) return;
  setBusy(true);
  try {
    const result = await bridge.apiPost("wardrobe/save", {
      baseRevision: state.revision,
      outfits: state.outfits,
      outfitSets: state.outfitSets,
      terms: state.terms,
    });
    if (!result || typeof result !== "object" || typeof result.revision !== "string"
      || !Array.isArray(result.outfits) || !Array.isArray(result.outfitSets) || !Array.isArray(result.terms)) {
      throw new Error("服务端没有返回可用的词库数据，未保存的修改仍保留。");
    }
    hydrate(result);
    showToast("服装词库已保存并立即生效。");
  } catch (error) {
    showToast(`保存失败：${errorText(error)}`);
  } finally {
    setBusy(false);
  }
}
function addEntry() { if (state.tab === "outfits") state.outfits.unshift({ key: "新角色服装档案", aliases: ["服装别名"], sourceTags: [], ownerTags: [], tags: [], appearanceTags: [], qualifier: "default", evidence: {} }); else if (state.tab === "outfitSets") state.outfitSets.unshift({ alias: "新通用服装套组", aliases: ["新通用服装套组", "canonical outfit tag"], tag: "canonical_outfit_tag", componentTags: [], variant: "default", origin: "learned", profileKey: `manual_named_${Date.now()}` }); else state.terms.unshift({ alias: "新名词", tag: "canonical_tag" }); markDirty(); render(); }

elements.tabs.forEach((tab) => tab.addEventListener("click", () => { state.tab = tab.dataset.tab; render(); }));
elements.sort.addEventListener("change", () => { state.outfitSort = elements.sort.value; render(); });
elements.search.addEventListener("input", () => { state.search = elements.search.value.trim(); render(); }); elements.add.addEventListener("click", addEntry); elements.save.addEventListener("click", save); elements.dirtySave.addEventListener("click", save); elements.refresh.addEventListener("click", load); elements.discard.addEventListener("click", () => pristine && hydrate(pristine));
window.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); save(); } }); window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });

async function initialize() { await bridge.ready(); document.documentElement.dataset.theme = bridge.getTheme?.() || new URLSearchParams(location.search).get("theme") || "light"; bridge.onContext?.(() => { document.documentElement.dataset.theme = bridge.getTheme?.() || "light"; }); await load(); }
initialize().catch((error) => showToast(`页面初始化失败：${errorText(error)}`));
