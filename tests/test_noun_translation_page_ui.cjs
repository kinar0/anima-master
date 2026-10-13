const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map();
function element(id = "") {
  return {
    id, value: "", textContent: "", disabled: true, hidden: true,
    handlers: {}, children: [],
    classList: { toggle() {}, add() {} },
    setAttribute() {},
    addEventListener(kind, handler) { this.handlers[kind] = handler; },
    replaceChildren() { this.children = []; }, append(child) { this.children.push(child); },
  };
}
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  },
  createElement: () => element(),
};
document.getElementById("batch-filter").value = "all";

const context = {
  document, clearTimeout, setTimeout,
  window: { AstrBotPluginPage: {
    ready: async () => {},
    apiGet: async () => ({
      total: 1000, translated: 50, verified: 50,
      importable: 20, skipped: 2, manual_review: 2, existing: 30,
      rows: [], additions: {}, skipped_aliases: {},
      enabled: true, job: { state: "idle", message: "" },
    }),
    apiPost: async (endpoint, body) => {
      assert.equal(endpoint, "noun-translation/enabled");
      return { enabled: body.enabled };
    },
  } },
};
Object.defineProperty(context, "localStorage", {
  get() { throw new Error("SecurityError: sandboxed iframe has no storage"); },
});
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../pages/wardrobe/pagination.js"), "utf8"), context);
const source = fs.readFileSync(path.join(__dirname, "../pages/wardrobe/noun_translation.js"), "utf8");
vm.runInContext(source, context);
setImmediate(async () => {
  assert.equal(document.getElementById("batch-translated").textContent, "");
  await vm.runInContext("load()", context);
  assert.equal(document.getElementById("batch-translated").textContent, "50 / 1000");
  assert.equal(document.getElementById("batch-translate").disabled, false);
  assert.equal(document.getElementById("batch-import").disabled, false);
  assert.equal(document.getElementById("batch-enabled").checked, true);
  assert.equal(document.getElementById("batch-enabled").disabled, false);
  assert.equal(document.getElementById("batch-review-count").textContent, 2);
  assert.equal(document.getElementById("batch-existing").textContent, 30);
  vm.runInContext(`
    snapshot.rows = [
      {tag: 'dress', aliases: ['连衣裙'], approved_aliases: ['连衣裙'],
       in_lexicon: true, ready: false, needs_manual_review: false,
       alias_statuses: [{alias: '连衣裙', state: 'existing', reason: '已在词库'}]},
      {tag: 'large_breasts', aliases: ['大胸', '巨乳'], approved_aliases: ['大胸', '巨乳'],
       in_lexicon: false, ready: true, needs_manual_review: true,
       alias_statuses: [{alias: '大胸', state: 'ready', reason: '可导入'},
         {alias: '巨乳', state: 'conflict', reason: '与现有映射冲突：当前指向 huge_breasts'}]},
      {tag: 'skirt', aliases: ['短裙'], approved_aliases: [],
       in_lexicon: false, ready: false, needs_manual_review: true,
       alias_statuses: [{alias: '短裙', state: 'rejected', reason: '模型复核未通过，未导入'}]},
      {tag: 'blouse', aliases: ['女式衬衫'], approved_aliases: null,
       in_lexicon: false, ready: false, needs_manual_review: false,
       alias_statuses: [{alias: '女式衬衫', state: 'pending', reason: '待模型复核'}]},
      {tag: 'empty', aliases: [], approved_aliases: [],
       in_lexicon: false, ready: false, needs_manual_review: true, alias_statuses: []}
    ];
  `, context);
  const selected = document.getElementById("batch-filter");
  function visibleTags(value) {
    selected.value = value;
    selected.handlers.change();
    return document.getElementById("batch-rows").children.map((row) => row.children[0].textContent);
  }
  assert.deepEqual(visibleTags("review"), ["large_breasts", "skirt", "empty"]);
  const mixedStatus = document.getElementById("batch-rows").children[0].children[3].textContent;
  assert.match(mixedStatus, /大胸：可导入/);
  assert.match(mixedStatus, /巨乳：与现有映射冲突/);
  assert.deepEqual(visibleTags("existing"), ["dress"]);
  assert.deepEqual(visibleTags("ready"), ["large_breasts"]);
  assert.deepEqual(visibleTags("pending"), ["blouse"]);
  assert.deepEqual(visibleTags("conflict"), ["large_breasts"]);
  assert.deepEqual(visibleTags("rejected"), ["skirt", "empty"]);
  vm.runInContext(`
    snapshot.rows[1].ready = false;
    snapshot.rows[1].alias_statuses[0] = {alias: '大胸', state: 'existing', reason: '已在词库'};
  `, context);
  assert.deepEqual(visibleTags("review"), ["large_breasts", "skirt", "empty"]);
  assert.match(document.getElementById("batch-rows").children[0].children[3].textContent, /大胸：已在词库/);
  document.getElementById("batch-enabled").checked = false;
  document.getElementById("batch-enabled").handlers.change();
  setImmediate(() => {
    assert.equal(document.getElementById("batch-enabled").checked, false);
    assert.equal(document.getElementById("batch-enabled").disabled, false);
    assert.match(document.getElementById("batch-status").textContent, /已停用/);
  });
});
