const autofilterBridge = window.AstrBotPluginPage;
const af = (selector) => document.querySelector(selector);
function autofilterStatus(message) { af("#autofilter-status").textContent = message; }
async function loadAutofilter() {
  autofilterStatus("正在读取…");
  try {
    const data = await autofilterBridge.apiGet("autofilter");
    const list = af("#autofilter-list"); list.replaceChildren();
    const sessions = Object.entries(data.sessions || {}).sort(([a], [b]) => a.localeCompare(b));
    autofilterStatus(sessions.length ? `共 ${sessions.length} 个会话` : "暂无会话；首次请求图片后会自动出现。");
    for (const [session, enabled] of sessions) {
      const row = document.createElement("div"); row.className = "autofilter-row";
      const name = document.createElement("code"); name.textContent = session;
      const label = document.createElement("label"); label.textContent = "发送前打码";
      const toggle = document.createElement("input"); toggle.type = "checkbox"; toggle.checked = enabled === true;
      toggle.addEventListener("change", async () => {
        toggle.disabled = true;
        try { await autofilterBridge.apiPost("autofilter/save", { session, enabled: toggle.checked }); autofilterStatus(`已保存 ${session}`); }
        catch (error) { toggle.checked = !toggle.checked; autofilterStatus(`保存失败：${error.message || error}`); }
        finally { toggle.disabled = false; }
      });
      label.append(toggle); row.append(name, label); list.append(row);
    }
  } catch (error) { autofilterStatus(`读取失败：${error.message || error}`); }
}
af("#autofilter-refresh").addEventListener("click", loadAutofilter);
af("#autofilter-add").addEventListener("click", async () => {
  const session = af("#autofilter-session").value.trim();
  if (!session) return autofilterStatus("请填写完整会话 ID");
  try { await autofilterBridge.apiPost("autofilter/save", { session, enabled: false }); af("#autofilter-session").value = ""; await loadAutofilter(); }
  catch (error) { autofilterStatus(`添加失败：${error.message || error}`); }
});
autofilterBridge.ready().then(loadAutofilter).catch((error) => autofilterStatus(`初始化失败：${error.message || error}`));
