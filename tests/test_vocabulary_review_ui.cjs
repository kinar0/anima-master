// Offline browser behavior: mocks API calls, never writes actual plugin data.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

async function run() {
  const root = path.join(__dirname, "../pages/wardrobe");
  const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, "../third_party/llm_nai_toolbox/vocabulary_review_catalog.json"), "utf8"));
  const baseline = JSON.parse(fs.readFileSync(path.join(__dirname, "../vocabulary_quality/round_02/baseline.json"), "utf8"));
  const existing = Object.fromEntries(baseline.original_rows.map(row => row.split("=")));
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (url.origin !== "http://wardrobe.test") return route.abort();
      const filename = url.pathname.slice(1) || "index.html";
      if (!["index.html", "style.css", "pagination.js", "app.js", "noun_translation.js", "vocabulary_review.js", "autofilter.js"].includes(filename)) return route.abort();
      return route.fulfill({ body: fs.readFileSync(path.join(root, filename)), contentType: filename.endsWith("js") ? "text/javascript" : filename.endsWith("css") ? "text/css" : "text/html" });
    });
    await page.addInitScript(({ catalog, existing }) => {
      window.__requests = [];
      window.__decisions = [];
      window.__rows = catalog.records.filter(r => r.focus);
      window.__fixture = { revision: "initial", outfits: [], outfitSets: [], terms: [{ alias: "白衬衫", tag: "white_shirt" }, { alias: "事后", tag: "aftersex" }] };
      window.__reviewRevision = "review0";
      window.AstrBotPluginPage = {
        ready: async () => {}, getTheme: () => "light",
        apiGet: async (endpoint, params = {}) => {
          window.__requests.push(endpoint);
          if (endpoint === "wardrobe") {
            if (window.__delayRead) { await new Promise(resolve => window.__finishRead = resolve); }
            return structuredClone(window.__fixture);
          }
          if (endpoint === "noun-translation/status") return { enabled: true, job: { state: "idle" } };
          if (endpoint === "noun-translation") return { enabled: true, job: { state: "idle" }, total: 100, translated: 30, verified: 20, importable: 0, existing: 0, manual_review: 0, rows: [] };
          if (endpoint === "vocabulary-review") {
            const query = params.query;
            const page = Number(params.page);
            let rows = window.__rows;
            if (query) rows = rows.filter(r => r.tag.includes(query) || r.aliases.some(a => a.alias.includes(query)));
            const total = rows.length;
            return { revision: window.__reviewRevision, total, catalog_total: catalog.records.length, page, pages: Math.max(1, Math.ceil(total / 30)), rows: rows.slice((page - 1) * 30, page * 30).map(r => ({ ...r, decision: window.__notes?.[r.tag] || { state: "pending", aliases: [], note: "" }, current_aliases: Object.entries(existing).filter(([a,t]) => t === r.tag).map(([a]) => a), aliases: r.aliases.map(a => ({ ...a, mapping_state: existing[a.alias] === r.tag ? "existing" : existing[a.alias] ? "conflict" : "candidate", current_tag: existing[a.alias] })) })) };
          }
          return { sessions: [] };
        },
        apiPost: async (endpoint, body) => {
          window.__requests.push(endpoint);
          if (endpoint === "vocabulary-review/decision") {
            window.__decisions.push(structuredClone(body));
            if (window.__failDecision) throw new Error("审查记录已变化，请刷新后重试");
            window.__notes ||= {};
            window.__notes[body.tag] = { state: body.state, aliases: body.aliases, note: body.note };
            window.__reviewRevision = "review" + window.__decisions.length;
            return { revision: window.__reviewRevision, applied: false, message: "审查记录已保存；未写入生图词库" };
          }
          if (endpoint === "wardrobe/save") {
            window.__saved = structuredClone(body);
            if (window.__delaySave) await new Promise(resolve => window.__finishSave = resolve);
            window.__fixture = { ...body, revision: "saved" };
            return structuredClone(window.__fixture);
          }
          return {};
        },
      };
    }, { catalog, existing });
    await page.goto("http://wardrobe.test/index.html");
    await page.waitForFunction(() => document.querySelector("#connection span").textContent === "已连接");
    assert.equal(await page.evaluate(() => window.__requests.includes("noun-translation")), false);
    assert.equal(await page.evaluate(() => window.__requests.includes("vocabulary-review")), false);
    assert.equal(await page.locator("#batch-enabled").isEnabled(), true);

    await page.locator("#review-details > summary").click();
    await page.waitForFunction(() => document.querySelectorAll(".review-card").length === 30);
    const firstTag = await page.locator(".review-card h3").first().textContent();
    const first = page.locator(".review-card").first();
    await first.locator("textarea").first().fill("自定义候选别名");
    await first.locator("textarea").nth(1).fill("人工依据暂存");
    await page.locator("#review-next").click();
    await page.waitForFunction(() => document.querySelector("#review-page").textContent.includes("第 2"));
    await page.locator("#review-prev").click();
    await page.waitForFunction(() => document.querySelector("#review-page").textContent.includes("第 1"));
    assert.equal(await first.locator("textarea").first().inputValue(), "自定义候选别名");
    await first.getByRole("button", { name: "采纳为待验收", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("#review-status").textContent.includes("未写入"));
    const decision = await page.evaluate(() => window.__decisions[0]);
    assert.equal(decision.tag, firstTag);
    assert.deepEqual(decision.aliases, ["自定义候选别名"]);
    assert.equal(await page.evaluate(() => window.__requests.includes("noun-translation/import")), false);
    assert.equal(await page.evaluate(() => window.__requests.includes("wardrobe/save")), false);

    await first.locator("textarea").nth(1).fill("冲突时也保留我的修改");
    await page.evaluate(() => window.__failDecision = true);
    await first.getByRole("button", { name: "暂缓", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("#review-status").textContent.includes("保存失败"));
    assert.equal(await first.locator("textarea").nth(1).inputValue(), "冲突时也保留我的修改");
    await page.evaluate(() => window.__failDecision = false);
    await first.getByRole("button", { name: "暂缓", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("#review-status").textContent.includes("未写入"));

    await page.locator("#batch-details > summary").click();
    await page.waitForFunction(() => window.__requests.includes("noun-translation"));
    await page.locator("button[data-tab=terms]").click();
    await page.evaluate(() => window.__delayRead = true);
    await page.locator("#refresh-button").click();
    await page.waitForFunction(() => !!window.__finishRead);
    await page.locator("#list .entry input").first().fill("读取时的新编辑");
    await page.evaluate(() => { window.__finishRead(); window.__delayRead = false; });
    await page.waitForFunction(() => !document.querySelector("#save-button").disabled);
    assert.equal(await page.locator("#list .entry input").first().inputValue(), "读取时的新编辑");
    const reads = await page.evaluate(() => window.__requests.filter(p => p === "wardrobe").length);
    await page.locator("#refresh-button").click();
    assert.equal(await page.evaluate(() => window.__requests.filter(p => p === "wardrobe").length), reads);

    await page.evaluate(() => window.__delaySave = true);
    await page.locator("#save-button").click();
    await page.waitForFunction(() => !!window.__finishSave);
    await page.locator("#list .entry input").first().fill("保存期间的新编辑");
    await page.evaluate(() => window.__finishSave());
    await page.waitForFunction(() => !document.querySelector("#save-button").disabled);
    assert.equal(await page.locator("#list .entry input").first().inputValue(), "保存期间的新编辑");
    assert.equal(await page.evaluate(() => window.__saved.terms[0].alias), "读取时的新编辑");
    assert.equal(await page.evaluate(() => window.__saved.terms.length), 2);
    assert.equal(await page.evaluate(() => window.__saved.terms[1].alias), "事后");
    await page.evaluate(() => window.__delaySave = false);
    await page.locator("#save-button").click();
    await page.waitForFunction(() => document.querySelector("#dirty-bar").hidden);
    assert.equal(await page.evaluate(() => window.__saved.terms[0].alias), "保存期间的新编辑");

    await page.locator("#review-query").fill("hug");
    await page.locator("#review-search").getByRole("button", { name: "查找", exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll(".review-card").length === 1);
    await page.locator("#vocabulary-review").scrollIntoViewIfNeeded();
    await page.locator("#toast").waitFor({ state: "hidden" });
    await page.screenshot({ path: path.join(__dirname, "../vocabulary_quality/round_02/review-preview.png"), fullPage: false });
    assert.deepEqual(errors, []);
    console.log("Review UI, lazy loading, drafts, conflicts and slow read/save passed");
  } finally { await browser.close(); }
}
run().catch(e => { console.error(e); process.exitCode = 1; });
