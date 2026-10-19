const bridge = window.AstrBotPluginPage;
const $ = id => document.getElementById(id);
let snapshot, current, revision, selected = '', category = '', page = 1, dirty = false, busy = false, loading = 0, detailRequest = 0;
const fields = ['aliases','definition','trap','contrasts','entry-categories','entry-disabled','tag'];
const make = (tag, text, cls = '') => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; n.className = cls; return n; };
const lines = value => [...new Set(value.split(/\r?\n/).map(x => x.trim()).filter(Boolean))];
function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function canNavigate() { if (busy) return false; if (dirty) { status('有未保存的修改，请先保存或点击“放弃修改”。', true); return false; } return true; }
function markDirty() { dirty = true; $('save').textContent = '保存修改 · 未保存'; }
function freeze(value) { busy = value; for (const e of $('editor').elements) e.disabled = value; }
async function load() {
  const version = ++loading;
  try {
    status('正在读取词库…'); await bridge.ready();
    const data = await bridge.apiGet('toolbox/library', { query: $('query').value.trim(), category, state: $('state').value, page: String(page), page_size: '30' });
    if (version !== loading) return;
    snapshot = data; page = data.page;
    $('total').textContent = data.stats.total.toLocaleString(); $('modified').textContent = data.stats.modified; $('disabled').textContent = data.stats.disabled;
    $('mode').textContent = ({off:'参考已关闭',observe:'仅记录候选',reference:'词汇参考已开启'})[data.mode] || data.mode;
    $('budget').textContent = `新增参考预算 ${data.max_chars} 字符`; $('directory').textContent = data.directory;
    $('result-count').textContent = `${category || '全部分类'} · ${data.total.toLocaleString()} 条`;
    $('page').value = page; $('page').max = data.pages; $('pages').textContent = data.pages;
    $('prev').disabled = page <= 1; $('next').disabled = page >= data.pages;
    renderCategories(); renderRows(); status(`已读取本地词库。保存修改后，下一次检索生效。`);
  } catch (error) { if (version === loading) status(`读取失败：${error.message || error}`, true); }
}
function renderCategories() {
  const container = $('categories'); const open = new Set([...container.querySelectorAll('details[open]')].map(e => e.dataset.group)); container.replaceChildren();
  const item = (name, count, label = name) => { const b = make('button', undefined, `category${category === name ? ' active' : ''}`); b.type = 'button'; b.append(make('span',label),make('span',count.toLocaleString())); b.addEventListener('click', () => { if (canNavigate()) { category = name; page = 1; load(); } }); return b; };
  container.append(item('',snapshot.stats.total,'全部分类'));
  const groups = new Map();
  for (const c of snapshot.categories) { const [group,...rest] = c.name.split('/'); if (!rest.length) container.append(item(c.name,c.count)); else { if (!groups.has(group)) groups.set(group,[]); groups.get(group).push(c); } }
  for (const [group, values] of groups) { const d = make('details',undefined,'category-group'); d.dataset.group = group; d.open = open.has(group) || values.some(c => c.name === category); d.append(make('summary',group)); for (const c of values) d.append(item(c.name,c.count,c.name.slice(group.length+1))); container.append(d); }
}
function renderRows() {
  $('rows').replaceChildren();
  for (const row of snapshot.rows) { const b = make('button',undefined,`word-row${selected === row.tag ? ' active' : ''}`); b.type = 'button'; b.append(make('strong',row.tag),make('p',row.definition || row.aliases.join(' · ') || '暂无中文释义')); const flags = [row.custom ? '本地新增' : row.modified ? '本地维护' : '',row.disabled ? '已停用' : '',row.deprecated ? '废弃提醒' : ''].filter(Boolean); if (flags.length) b.append(make('small',flags.join(' · '))); b.addEventListener('click', () => { if (canNavigate()) select(row.tag); }); $('rows').append(b); }
  if (!snapshot.rows.length) $('rows').append(make('p','没有符合条件的词条。可以更换关键词或新建词条。','empty'));
}
async function select(tag) {
  const version = ++detailRequest;
  try { status('正在打开词条…'); const result = await bridge.apiGet('toolbox/entry',{tag}); if (version !== detailRequest) return; if (dirty) { status('打开期间有新的编辑，已保留草稿；请先保存或放弃修改。',true); return; } current = result.entry; revision = result.revision; selected = tag; fill(); renderRows(); status('可编辑本地别名与释义，原始文件保持可恢复。'); }
  catch (error) { if (version === detailRequest) status(`打开失败：${error.message || error}`,true); }
}
function fill() {
  dirty = false; $('welcome').hidden = true; $('editor').hidden = false;
  $('entry-title').textContent = current.tag || '新的词条'; $('entry-origin').textContent = current.custom ? '本地新增' : current.modified ? '已使用本地维护值' : 'Toolbox 原始资料';
  $('tag').value = current.tag; $('tag').readOnly = Boolean(current.tag); $('aliases').value = current.aliases.join('\n'); $('definition').value = current.definition; $('trap').value = current.trap;
  $('contrasts').value = current.contrasts.map(x => `${x[0]} | ${x[1]}`).join('\n'); $('entry-categories').value = current.categories.join('\n'); $('entry-disabled').checked = current.disabled;
  $('original-panel').hidden = !current.original; $('original').textContent = current.original ? `中文别名：${current.original.aliases.join('、')}\n释义：${current.original.definition}\n易误读：${current.original.trap}\n辨析：${current.original.contrasts.map(x=>x.join('：')).join('；')}\n分类：${current.original.categories.join(' / ')}` : '';
  $('restore').hidden = !current.modified; $('deprecated').hidden = !current.deprecated; $('deprecated').textContent = current.deprecated || ''; $('copy').disabled = !current.tag; $('save').textContent = '保存修改';
}
for (const id of fields) $(id).addEventListener('input', markDirty);
$('search-form').addEventListener('submit',event => { event.preventDefault(); if (canNavigate()) { page = 1; load(); } });
$('state').addEventListener('change',() => { if (canNavigate()) { page = 1; load(); } });
$('refresh').addEventListener('click',() => { if (canNavigate()) { load(); if (selected) select(selected); } });
$('prev').addEventListener('click',() => { if (canNavigate() && page > 1) { page--; load(); } });
$('next').addEventListener('click',() => { if (canNavigate() && page < snapshot.pages) { page++; load(); } });
$('page').addEventListener('change',() => { if (canNavigate()) { page = Math.max(1,Math.min(snapshot.pages,Number($('page').value)||1)); load(); } });
$('new').addEventListener('click',() => { if (!canNavigate() || !snapshot) return; ++detailRequest; selected = ''; revision = snapshot.revision; current = {tag:'',aliases:[],definition:'',trap:'',contrasts:[],categories:category ? [category] : [],disabled:false,custom:true}; fill(); renderRows(); $('tag').focus(); });
$('discard').addEventListener('click',() => { if (busy) return; dirty = false; if (selected) select(selected); else { $('editor').hidden = true; $('welcome').hidden = false; } });
async function save(action) {
  if (busy) return;
  let entry;
  if (action === 'save') {
    const contrasts = lines($('contrasts').value).map(x => { const p = x.indexOf('|'); if (p < 1) throw new Error('辨析请按“易混词名称或 tag | 区别说明”填写。'); return [x.slice(0,p).trim(),x.slice(p+1).trim()]; });
    entry = {aliases:lines($('aliases').value),definition:$('definition').value.trim(),trap:$('trap').value.trim(),contrasts,categories:lines($('entry-categories').value),disabled:$('entry-disabled').checked};
  }
  const payload = {tag:$('tag').value.trim(),base_revision:revision,action,entry};
  freeze(true); ++loading; ++detailRequest; status('正在保存本地维护记录…');
  try { const result = await bridge.apiPost('toolbox/save',payload); dirty = false; revision = result.revision; await load(); if (result.exists) await select(result.tag); else { selected = ''; $('editor').hidden = true; $('welcome').hidden = false; } status(result.message); }
  catch (error) { status(`保存失败：${error.message || error}。编辑内容仍保留。`,true); }
  finally { freeze(false); $('tag').readOnly = Boolean(current?.tag); }
}
$('editor').addEventListener('submit',event => { event.preventDefault(); save('save').catch(error => status(error.message,true)); });
$('restore').addEventListener('click',() => save('restore').catch(error => status(error.message,true)));
$('copy').addEventListener('click',async () => { try { await navigator.clipboard.writeText($('tag').value); status('已复制英文 tag。'); } catch { $('tag').focus(); $('tag').select(); status('浏览器未允许自动复制；已选中 tag，可手动复制。'); } });
$('probe').addEventListener('click',async () => { if (busy) return; const text = $('query').value.trim(); if (!text) { status('先在搜索框填写试查内容。',true); return; } $('probe').disabled = true; try { const data = await bridge.apiPost('toolbox/probe',{text}); const d = data.diagnostics; $('probe-result').hidden = false; $('probe-result').replaceChildren(make('p',data.note)); for (const c of d.candidates || []) { const b = make('button',`${c.tag} · ${c.score.toFixed(2)}`); b.type = 'button'; b.addEventListener('click',() => { if (canNavigate()) select(c.tag); }); $('probe-result').append(b); } $('probe-result').append(make('p',`${d.candidates?.length || 0} 个候选 · 匹配分数不是生成概率${d.state === 'ready' ? '' : ' · 数据状态：'+d.state}`)); } catch (error) { status(`试查失败：${error.message || error}`,true); } finally { $('probe').disabled = false; } });
window.addEventListener('beforeunload',event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
$('return-wardrobe')?.addEventListener('click',event => { if (!canNavigate()) event.preventDefault(); });
if (!bridge) { status('请从 AstrBot 管理页面打开 Toolbox 词库。',true); for (const e of document.querySelectorAll('button,input,select')) e.disabled = true; }
else load();
