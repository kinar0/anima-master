const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const elements = new Map();
const handlers = {};
let reads = 0;
let intervals = 0;
const document = {
  hidden: false,
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, { textContent: "", hidden: false, classList: { toggle() {}, add() {} } });
    return elements.get(id);
  },
  addEventListener() {},
};
const context = {
  document,
  setInterval() { intervals++; return intervals; },
  clearInterval() { intervals--; },
  window: {
    addEventListener(event, handler) { handlers[event] = handler; },
    AstrBotPluginPage: {
      ready: async () => {},
      apiGet: async (endpoint) => { assert.equal(endpoint, "opus-usage"); reads++; return { enabled: true, interval_minutes: 120, status: "waiting" }; },
    },
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../pages/wardrobe/opus_usage.js"), "utf8"), context);
setImmediate(async () => {
  assert.equal(intervals, 1);
  handlers.pageshow();
  assert.equal(intervals, 1);
  function render(subscription, extra={}) {
    context.data = { enabled: true, interval_minutes: 120, status: "ok", subscription, ...extra };
    vm.runInContext("renderOpusUsage(data)", context);
  }
  render({ active: true, tier: 3, percent: 0, is_negative: false });
  assert.equal(elements.get("opus-percent").textContent, "0%");
  assert.equal(elements.get("opus-progress").hidden, false);
  render({ active: true, tier: 3, percent: 130, is_negative: false });
  assert.equal(elements.get("opus-percent").textContent, "130%");
  assert.equal(elements.get("opus-progress").value, 100);
  render({ active: true, tier: 3, percent: 5, is_negative: true });
  assert.equal(elements.get("opus-percent").textContent, "−5%");
  assert.equal(elements.get("opus-progress").value, 0);
  render(null);
  assert.equal(elements.get("opus-percent").textContent, "—");
  assert.equal(elements.get("opus-progress").hidden, true);
  render({ active: true, tier: 2 });
  assert.match(elements.get("opus-status").textContent, /不是 Opus/);
  render({ active: false, tier: 3 });
  assert.match(elements.get("opus-status").textContent, /未生效/);
  render({ active: true, tier: 3, percent: 85, is_negative: false }, { status: "unavailable", stale: true });
  assert.equal(elements.get("opus-percent").textContent, "85%");
  assert.match(elements.get("opus-status").textContent, /上次成功值/);
  render(null, { enabled: false });
  assert.match(elements.get("opus-status").textContent, /已停用/);
  handlers.pagehide();
  assert.equal(intervals, 0);
  handlers.pageshow();
  assert.equal(intervals, 1);
  document.hidden = true;
  const previous = reads;
  await vm.runInContext("loadOpusUsage()", context);
  assert.equal(reads, previous);
  console.log("Opus usage UI checks passed");
});
