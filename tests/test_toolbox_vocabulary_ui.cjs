// Full imported vocabulary in the existing editor; all API writes are mocked.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

async function run() {
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, "../toolbox_import/applied.json"), "utf8"));
  const terms = [...manifest.previous_rows, ...Object.entries(manifest.additions).map(([a, t]) => `${a}=${t}`)].map(row => {
    const match = [/^\s*(.+?)\s*(?:=>|=|＝)\s*(.+?)\s*$/, /^\s*(.+?)\s*→\s*(.+?)\s*$/, /^\s*(.+?)\s*[:：]\s*(.+?)\s*$/].map(pattern => row.match(pattern)).find(Boolean);
    return { alias: match[1], tag: match[2] };
  }).sort((a, b) => a.alias < b.alias ? -1 : a.alias > b.alias ? 1 : 0);
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const root = path.join(__dirname, "../pages/wardrobe");
    await page.route("http://wardrobe.test/**", route => {
      const file = new URL(route.request().url()).pathname.slice(1) || "index.html";
      if (!["index.html", "style.css", "app.js", "pagination.js", "noun_translation.js", "autofilter.js"].includes(file)) return route.abort();
      return route.fulfill({ body: fs.readFileSync(path.join(root, file)), contentType: file.endsWith("js") ? "text/javascript" : file.endsWith("css") ? "text/css" : "text/html" });
    });
    await page.addInitScript(terms => {
      window.__fixture = { revision: "original", terms, outfits: [], outfitSets: [] };
      window.AstrBotPluginPage = {
        ready: async () => {}, getTheme: () => "light",
        apiGet: async endpoint => endpoint === "wardrobe" ? structuredClone(window.__fixture) : { enabled: true, rows: [], job: { state: "idle" } },
        apiPost: async (_endpoint, payload) => {
          window.__saved = structuredClone(payload);
          return { ...payload, revision: "saved" };
        },
      };
    }, terms);
    await page.goto("http://wardrobe.test/");
    await page.waitForFunction(total => document.querySelector("#term-count").textContent === String(total), terms.length);
    await page.locator('.tab[data-tab="terms"]').click();
    const entries = page.locator("#list > article");
    const top = page.locator("#library-pagination-top");
    assert.equal(await entries.count(), 30);
    await entries.first().locator("input").nth(1).fill("first_manual_edit");
    await top.locator(".page-last").click();
    assert.equal(await entries.count(), terms.length % 30 || 30);
    await entries.last().locator("input").nth(1).fill("last_manual_edit");
    const lastAlias = terms.at(-1).alias;
    await page.locator("#search-input").fill(lastAlias);
    assert.ok(await entries.count() >= 1);
    assert.equal(await entries.last().locator("input").nth(1).inputValue(), "last_manual_edit");
    await page.locator("#save-button").click();
    await page.waitForFunction(() => document.querySelector("#dirty-bar").hidden);
    const saved = await page.evaluate(() => window.__saved);
    assert.equal(saved.terms.length, terms.length);
    assert.equal(saved.terms[0].tag, "first_manual_edit");
    assert.equal(saved.terms.at(-1).tag, "last_manual_edit");
    assert.deepEqual(saved.terms.slice(1, -1), terms.slice(1, -1));
    assert.deepEqual(errors, []);
    console.log(`Toolbox vocabulary editor passed: ${terms.length} entries, full search and cross-page save.`);
  } finally { await browser.close(); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
