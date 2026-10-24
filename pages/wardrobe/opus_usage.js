const opusBridge = window.AstrBotPluginPage;
const opusElement = (id) => document.getElementById(id);
let opusReading = false;
let opusTimer;
let opusReady = false;

function opusTime(value) {
  const date = new Date(value);
  return value && Number.isFinite(date.getTime()) ? date.toLocaleString("zh-CN", { hour12: false }) : "—";
}

function renderOpusUsage(data) {
  const subscription = data.subscription;
  const valid = subscription?.active === true && subscription?.tier === 3 && Number.isFinite(subscription?.percent);
  const negative = subscription?.is_negative === true;
  const progress = opusElement("opus-progress");
  progress.hidden = !valid;
  progress.value = valid && !negative ? Math.min(100, Math.max(0, subscription.percent)) : 0;
  opusElement("opus-percent").textContent = valid ? `${negative ? "−" : ""}${subscription.percent}%` : "—";
  opusElement("opus-usage").classList.toggle("opus-low", valid && (negative || subscription.percent < 20));
  opusElement("opus-usage").classList.toggle("opus-stale", data.stale || !data.enabled);
  let status = "等待首次读取额度。";
  if (!data.enabled) status = "已停用自动查询，可在插件配置中开启。";
  else if (data.status === "unavailable") status = data.message || "查询暂不可用，显示上次成功值。";
  else if (subscription?.active === false) status = "当前 API 账户的订阅未生效。";
  else if (subscription && subscription.tier !== 3) status = "当前 API 账户不是 Opus 订阅。";
  else if (valid) status = data.stale ? "显示已过期缓存，等待下次更新。" : "账户额度缓存；与插件的每日生图次数分别计算。";
  opusElement("opus-status").textContent = status;
  const details = [];
  if (valid && Number.isFinite(subscription.purchased_percent)) details.push(`额外购买额度：${subscription.purchased_percent}%`);
  if (valid && Number.isFinite(subscription.time_until_next_percent)) {
    const minutes = Math.ceil(subscription.time_until_next_percent / 60);
    details.push(`查询时距下一百分点恢复约 ${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分钟`);
  }
  opusElement("opus-details").textContent = details.join(" · ");
  opusElement("opus-updated").textContent = `上次成功：${opusTime(data.checked_at)} · 每 ${data.interval_minutes} 分钟更新${data.enabled ? ` · 下次：${opusTime(data.next_update_at)}` : ""}`;
}

async function loadOpusUsage() {
  if (opusReading || document.hidden) return;
  opusReading = true;
  try {
    renderOpusUsage(await opusBridge.apiGet("opus-usage"));
  } catch {
    opusElement("opus-usage").classList.add("opus-stale");
    opusElement("opus-status").textContent = "无法读取额度缓存，请检查登录状态或稍后重试。";
  } finally {
    opusReading = false;
  }
}

document.addEventListener("visibilitychange", () => { if (!document.hidden) loadOpusUsage(); });
window.addEventListener("pagehide", () => { clearInterval(opusTimer); opusTimer = null; });
window.addEventListener("pageshow", () => {
  if (opusReady && !opusTimer) { loadOpusUsage(); opusTimer = setInterval(loadOpusUsage, 60000); }
});
opusBridge.ready().then(() => {
  opusReady = true;
  loadOpusUsage();
  if (!opusTimer) opusTimer = setInterval(loadOpusUsage, 60000);
}).catch(() => {
  opusElement("opus-status").textContent = "无法连接插件页面，请刷新后重试。";
});
