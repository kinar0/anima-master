// Offline fixture generation only. Node is NOT a plugin runtime dependency.
import { readFileSync, writeFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { fileURLToPath } from 'node:url';
const root = new URL('../', import.meta.url);
const original = readFileSync(new URL('third_party/llm_nai_toolbox/reference/search.ts', root), 'utf8');
const source = stripTypeScriptTypes(original + '\nexport { calcSimilarity, searchOne };', { mode: 'strip' });
const upstream = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const entries = [
  {tag:'from_above',count:500,zh:['俯视','从上方拍摄']},
  {tag:'looking_down',count:400,zh:['俯视','低头']},
  {tag:'cowboy_shot',count:300,zh:['牛仔构图']},
  {tag:'wet_clothes',count:200,zh:['湿衣服','湿衣']},
  {tag:'dirty_clothes',count:100,zh:['脏衣服']},
  {tag:'ponytail',count:100,zh:['马尾'],zhFull:['马尾'],zhShort:['尾'],zhNick:['小马尾']},
].map(e => Object.fromEntries(Object.entries({zh:[],zhFull:[],zhShort:[],zhNick:[],ja:[],en:[],other:[],series:[],...e})));
const pairs = [
  ['cowboy_shot','cowboy shot'],['ponytial','ponytail'],['湿衣服','脏衣服'],
  ['湿衣服','湿衣'],['蒸汽身体','蒸汽'],['絵桜','绘樱'],['W','white dress'],
  ['舌头上的物件','物件在舌头上'],['😀hat','😀hut'],['unknown','unknown'],
];
const queries = ['俯视','牛仔构图','湿衣服','脏衣服','ponytial','马尾','小马尾','W','unrelated'];
const index = upstream.buildIndex(entries);
const fixture = {
  source: 'llm-nai-toolbox 0.9.8 search.ts; synthetic entries, no copied vocabulary dataset',
  entries,
  pairs: pairs.map(([q,t])=>({q,t,expected:upstream.calcSimilarity(q,t)})),
  matches: queries.map(q=>({q,expected:entries.map(e=>upstream.matchEntry(q,e))})),
  searches: queries.map(q=>({q,expected:upstream.searchOne(entries,q,'概念',index).matches.slice(0,2).map(x=>({tag:x.tag,score:x.score}))})),
};
const out = new URL('tests/toolbox_upstream_fixture.json', root);
writeFileSync(out, JSON.stringify(fixture,null,2)+'\n');
process.stdout.write(fileURLToPath(out)+'\n');
