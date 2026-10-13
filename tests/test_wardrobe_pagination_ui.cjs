// Offline browser regression. Requires Playwright and an installed Chrome.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

async function run() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const root = path.join(__dirname, "../pages/wardrobe");
    await page.route("http://wardrobe.test/**", (route) => {
      const filename = new URL(route.request().url()).pathname.slice(1) || "index.html";
      if (!["index.html", "style.css", "pagination.js", "app.js", "noun_translation.js", "vocabulary_review.js", "autofilter.js"].includes(filename)) return route.abort();
      return route.fulfill({ body: fs.readFileSync(path.join(root, filename)), contentType: filename.endsWith("js") ? "text/javascript" : filename.endsWith("css") ? "text/css" : "text/html" });
    });
    await page.addInitScript(() => {
      const terms = Array.from({ length: 1229 }, (_, i) => ({ alias: `词条${i}`, tag: `tag_${i}` }));
      window.__fixture = {
        revision: "initial", terms,
        outfits: Array.from({ length: 61 }, (_, i) => ({ key: `角色${i}`, aliases: [`角色${i}`], sourceTags: [`character_${i}`], tags: [], appearanceTags: [], qualifier: "default", evidence: { createdAt: i + 1 } })),
        outfitSets: Array.from({ length: 95 }, (_, i) => ({ alias: `套组${i}`, aliases: [`套组${i}`], tag: `set_${i}`, variant: "default", componentTags: [] })),
      };
      window.AstrBotPluginPage = {
        ready: async () => {}, getTheme: () => "light",
        apiGet: async (endpoint) => {
          if (endpoint === "wardrobe") return structuredClone(window.__fixture);
          if (endpoint === "noun-translation") return { total: 95, translated: 95, verified: 95, importable: 0, skipped: 95, enabled: true, job: { state: "idle" }, additions: {}, skipped_aliases: {}, rows: Array.from({ length: 95 }, (_, i) => ({ tag: `batch_${i}`, aliases: [`批量${i}`], approved_aliases: [], review_flags: [], alias_statuses: [{ alias: `批量${i}`, state: "rejected", reason: "模型未批准" }], needs_manual_review: true, ready: false, in_lexicon: false })) };
          return { sessions: [] };
        },
        apiPost: async (endpoint, body) => {
          if (endpoint === "wardrobe/save") {
            window.__saved = structuredClone(body);
            if (window.__failSave) throw new Error("revision conflict");
            window.__fixture = { ...body, revision: "saved" };
            return structuredClone(window.__fixture);
          }
          return {};
        },
      };
      // This page also needs to work in an iframe without storage permission.
      Object.defineProperty(window, "localStorage", { get() { throw new Error("SecurityError"); } });
    });
    await page.goto("http://wardrobe.test/");
    await page.waitForFunction(() => document.querySelector("#outfit-count").textContent === "61");
    const top = page.locator("#library-pagination-top");
    const bottom = page.locator("#library-pagination-bottom");
    const entries = page.locator("#list > article");
    const summary = () => top.locator(".pagination-summary").textContent();
    const tab = (name) => page.locator(`.tab[data-tab="${name}"]`).click();
    const jump = async (value, keyboard = false) => {
      await top.locator(".page-number").fill(String(value));
      if (keyboard) await top.locator(".page-number").press("Enter");
      else await top.locator(".page-go").click();
    };
    assert.equal(await entries.count(), 30);
    assert.equal(await entries.first().locator("input").first().inputValue(), "角色60");
    await top.locator(".page-last").click();
    assert.equal(await entries.count(), 1);
    await entries.first().locator(".delete-button").click();
    assert.match(await summary(), /第 2 \/ 2 页/);
    assert.equal(await entries.count(), 30);
    await page.locator("#outfit-sort").selectOption("added_asc");
    assert.equal(await entries.first().locator("input").first().inputValue(), "角色1");

    await tab("outfitSets");
    assert.equal(await entries.count(), 30);
    await top.locator(".page-size").selectOption("50");
    assert.equal(await entries.count(), 50);
    await bottom.locator(".page-next").click();
    assert.equal(await entries.count(), 45);
    assert.equal(await top.locator(".page-number").inputValue(), "2");

    await tab("terms");
    assert.equal(await entries.count(), 30);
    assert.match(await summary(), /共 1229 条.*第 1 \/ 41 页/);
    await top.locator(".page-next").click();
    assert.equal(await entries.first().locator("input").first().inputValue(), "词条30");
    await entries.first().locator("input").first().fill("跨页修改");
    await jump(41, true);
    assert.equal(await entries.count(), 29);
    await entries.last().locator("input").nth(1).fill("edited_tag");
    await jump(9999);
    assert.match(await summary(), /第 41 \/ 41 页/);
    await jump("", true);
    assert.equal(await top.locator(".page-number").inputValue(), "41");
    await jump(-3);
    assert.match(await summary(), /第 1 \/ 41 页/);
    await jump("", true);
    assert.equal(await top.locator(".page-number").inputValue(), "1");
    await jump("1.5", true);
    assert.equal(await top.locator(".page-number").inputValue(), "1");
    await jump(2);
    assert.equal(await entries.first().locator("input").first().inputValue(), "跨页修改");
    await tab("outfitSets");
    assert.equal(await top.locator(".page-size").inputValue(), "50");
    assert.equal(await top.locator(".page-number").inputValue(), "2");
    await tab("terms");
    assert.equal(await top.locator(".page-number").inputValue(), "2");
    await page.locator("#search-input").fill("词条1228");
    assert.equal(await entries.count(), 1);
    assert.match(await summary(), /筛选 1 \/ 1229 条.*第 1 \/ 1 页/);
    assert.equal(await entries.first().locator("input").first().inputValue(), "词条1228");
    await page.locator("#search-input").fill("no-such-term");
    assert.equal(await entries.count(), 0);
    assert.equal(await top.locator(".page-go").isDisabled(), true);
    await page.locator("#add-button").click();
    assert.equal(await page.locator("#search-input").inputValue(), "");
    assert.equal(await entries.first().locator("input").first().inputValue(), "新名词");
    await page.locator("#save-button").click();
    await page.waitForFunction(() => document.querySelector("#dirty-bar").hidden);
    const saved = await page.evaluate(() => window.__saved);
    assert.equal(saved.terms.length, 1230);
    assert.equal(saved.terms[31].alias, "跨页修改");
    assert.equal(saved.terms[1229].tag, "edited_tag");
    assert.equal(saved.outfits.length, 60);
    assert.equal(saved.outfitSets.length, 95);

    await page.evaluate(() => {
      window.__fixture.terms = Array.from({ length: 30001 }, (_, i) => ({ alias: `大词库${i}`, tag: `large_${i}` }));
    });
    await page.locator("#refresh-button").click();
    await page.waitForFunction(() => document.querySelector("#term-count").textContent === "30001");
    assert.equal(await entries.count(), 30);
    await top.locator(".page-last").click();
    assert.equal(await entries.count(), 1);
    await entries.first().locator(".delete-button").click();
    assert.match(await summary(), /第 1000 \/ 1000 页/);
    await top.locator(".page-size").selectOption("100");
    assert.equal(await entries.count(), 100);
    assert.match(await summary(), /第 300 \/ 300 页/);
    await entries.first().locator("input").first().fill("失败后保留");
    await page.evaluate(() => window.__failSave = true);
    await page.locator("#save-button").click();
    await page.waitForFunction(() => document.querySelector("#toast").textContent.includes("revision conflict"));
    assert.equal(await page.locator("#dirty-bar").isVisible(), true);
    assert.equal(await entries.first().locator("input").first().inputValue(), "失败后保留");
    await page.locator("#discard-button").click();
    assert.equal(await page.locator("#term-count").textContent(), "30001");

    await page.locator("#batch-details > summary").click();
    await page.waitForFunction(() => document.querySelectorAll("#batch-rows > tr").length === 30);
    const batch = page.locator("#batch-pagination");
    assert.equal(await page.locator("#batch-rows > tr").count(), 30);
    await batch.locator(".page-number").fill("4");
    await batch.locator(".page-number").press("Enter");
    assert.equal(await page.locator("#batch-rows > tr").count(), 5);
    await batch.locator(".page-size").selectOption("50");
    assert.equal(await page.locator("#batch-rows > tr").count(), 45);
    await page.locator("#batch-filter").selectOption("ready");
    assert.match(await batch.locator(".pagination-summary").textContent(), /筛选 0 \/ 95 条.*第 1 \/ 1 页/);

    if (process.env.PAGINATION_SCREENSHOT_DIR) {
      await page.locator("#search-input").fill("大词库30000");
      await top.scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(process.env.PAGINATION_SCREENSHOT_DIR, "pagination-desktop.png") });
      await page.setViewportSize({ width: 390, height: 844 });
      await top.scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(process.env.PAGINATION_SCREENSHOT_DIR, "pagination-mobile.png") });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
      assert.equal(overflow, false, "mobile page should not scroll horizontally");
      assert.ok((await entries.first().locator("input").first().boundingBox()).width > 200, "mobile editor fields should use available width");
    }
    assert.deepEqual(errors, []);
    console.log("Wardrobe pagination browser regressions passed (1229 and 30001 terms).");
  } finally { await browser.close(); }
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
