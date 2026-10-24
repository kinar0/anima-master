from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolbox_vocabulary import ToolboxVocabulary, match_entry, similarity, _general_rows


def data_dir(tmp_path: Path) -> Path:
    fixture = json.loads((ROOT/'tests/toolbox_upstream_fixture.json').read_text(encoding='utf-8'))
    entries = fixture['entries'] + [dict(tag='smile',count=100,zh=['微笑'])]
    (tmp_path/'tags_index_v2.json').write_text(json.dumps({'general':entries, 'characters':[dict(tag='fake_character',zh=['俯视'])]},ensure_ascii=False),encoding='utf-8')
    gloss = {
        'from_above':dict(g='相机从上方向下拍摄',vs=[['looking_down','相机方向不同于人物视线']]),
        'looking_down':dict(g='人物视线向下',vs=[['from_above','人物视线不同于相机方向']]),
        'cowboy_shot':dict(g='包含头部的大腿以上构图',trap='与牛仔服饰无关'),
        'wet_clothes':dict(g='衣物被水浸湿'), 'dirty_clothes':dict(g='衣物沾有污渍'),
        'smile':dict(g='微笑'), 'ponytail':dict(g='马尾发型'),
    }
    (tmp_path/'tag_gloss.json').write_text(json.dumps(gloss,ensure_ascii=False),encoding='utf-8')
    (tmp_path/'tag_browse.json').write_text(json.dumps({'镜头':[
        dict(t='from_above',g='相机从上方向下拍摄',c=500,m='do not pass this wiki text'),
        dict(t='cowboy_shot',g='包含头部的大腿以上构图',c=300)]},ensure_ascii=False),encoding='utf-8')
    return tmp_path


def packet(path, text, *, mode='reference', mappings=(), max_chars=1200, service=None, phrases=()):
    return asyncio.run((service or ToolboxVocabulary()).prepare(directory=str(path), mode=mode,
        text=text, mappings=mappings, phrases=phrases, max_chars=max_chars))


def test_scoring_and_ranking_match_real_upstream_code_on_synthetic_cases(tmp_path):
    x = json.loads((ROOT/'tests/toolbox_upstream_fixture.json').read_text(encoding='utf-8'))
    for row in x['pairs']:
        assert similarity(row['q'],row['t']) == pytest.approx(row['expected'],abs=1e-12)
    for row in x['matches']:
        for entry,expected in zip(x['entries'],row['expected']):
            score,field = match_entry(row['q'],entry)
            assert score == pytest.approx(expected['score'],abs=1e-12)
            assert field == expected['field']
    (tmp_path/'tags_index_v2.json').write_text(json.dumps({'general':x['entries']}),encoding='utf-8')
    svc = ToolboxVocabulary()
    for row in x['searches']:
        assert [{k:r[k] for k in ('tag','score')} for r in svc.search(tmp_path,row['q'])] == row['expected']


def test_ambiguous_terms_explain_camera_versus_gaze_without_choosing_owner(tmp_path):
    p = packet(data_dir(tmp_path),'俯视')
    assert 'from_above' in p.rows and 'looking_down' in p.rows
    assert '相机方向不同于人物视线' in p.rows
    assert 'fake_character' not in p.rows
    assert len(p.diagnostics['candidates']) == 2
    assert 'score' not in p.rows and 'wiki' not in p.rows


@pytest.mark.parametrize('text', ['不要俯视','甲不要低头','甲说：“俯视”','俯视这个词','心里的俯视','心里的湿衣服','俯视只是一种比喻'])
def test_negative_or_literal_mentions_do_not_become_visual_candidates(tmp_path,text):
    assert packet(data_dir(tmp_path),text).rows == ''


def test_positive_clause_after_negative_and_full_absence_concepts(tmp_path):
    path = data_dir(tmp_path)
    p = packet(path,'不要俯视，微笑')
    assert 'smile' in p.rows and 'from_above' not in p.rows


def test_existing_mapping_gets_one_necessary_explanation_no_second_mapping(tmp_path):
    p = packet(data_dir(tmp_path),'牛仔构图，牛仔构图',mappings=(('牛仔构图','cowboy_shot'),))
    assert p.rows.count('cowboy_shot') == 1
    assert '=>' not in p.rows
    assert '与牛仔服饰无关' in p.rows


def test_plain_known_mapping_adds_no_redundant_definition(tmp_path):
    assert packet(data_dir(tmp_path),'微笑',mappings=(('微笑','smile'),)).rows == ''


def test_budget_discards_whole_optional_rows_and_no_hits_add_nothing(tmp_path):
    path=data_dir(tmp_path)
    for budget in (0,200,600,650,700,1200,2400,10000):
        p=packet(path,'俯视，牛仔构图，湿衣服，微笑',max_chars=budget)
        assert p.diagnostics['added_chars_upper_bound'] <= min(2400,budget)
        if p.rows:
            assert all(' — ' in r or '释义：' in r for r in p.rows.splitlines())
    assert packet(path,'xxxxxxxxxxxxxxxxxxxxxx').diagnostics['added_chars_upper_bound']==0


def test_observe_preserves_inputs_but_reports_same_candidates(tmp_path):
    path=data_dir(tmp_path)
    active=packet(path,'俯视')
    observe=packet(path,'俯视',mode='observe')
    assert observe.rows==''
    assert observe.diagnostics['candidates']==active.diagnostics['candidates']


def test_missing_corrupt_and_repaired_files_invalidate_cached_snapshot(tmp_path):
    svc=ToolboxVocabulary()
    assert packet(tmp_path,'俯视',service=svc).diagnostics['state']=='unavailable'
    path=data_dir(tmp_path)
    assert packet(path,'俯视',service=svc).rows
    (path/'tag_gloss.json').write_text('{broken',encoding='utf-8')
    broken=packet(path,'俯视',service=svc)
    assert broken.rows=='' and broken.diagnostics['state']=='unavailable'
    data_dir(tmp_path)
    assert packet(path,'俯视',service=svc).rows


def test_browse_uses_category_with_keyword_ordering_not_filtering(tmp_path):
    svc=ToolboxVocabulary()
    rows=svc.browse(data_dir(tmp_path),'镜头','大腿',limit=8)
    assert [r['tag'] for r in rows]==['cowboy_shot','from_above']
    assert all('wiki' not in str(r) for r in rows)
    assert svc.browse(tmp_path,'nonexistent')==[]


def test_off_never_opens_data_or_runs_thread(tmp_path,monkeypatch):
    svc=ToolboxVocabulary()
    monkeypatch.setattr(svc,'_load',lambda *_: pytest.fail('off must not read data'))
    assert packet(tmp_path,'俯视',mode='off',service=svc).rows==''


def test_bad_optional_fields_and_long_explanations_fail_conservatively(tmp_path):
    path=data_dir(tmp_path)
    (path/'tag_gloss.json').write_text(json.dumps({'from_above':{'g':'x'*120},'looking_down':{'g':8}}),encoding='utf-8')
    p=packet(path,'俯视')
    # Browse fallback is still real local evidence; malformed gloss never crashes.
    assert p.diagnostics['state']=='ready'
    assert 'x'*120 not in p.rows


def test_weak_unseen_paraphrase_is_not_promoted_to_a_semantic_match(tmp_path):
    path=data_dir(tmp_path)
    # Search exposes the lexical result; automatic context requires a literal
    # local name for weak matches, so a shared fragment cannot choose a meaning.
    (path/'tags_index_v2.json').write_text(json.dumps({'general':[
        {'tag':'wet_clothes','count':100,'zh':['甲乙丙丁']}]}),encoding='utf-8')
    p=packet(path,'甲乙丙戊')
    assert p.rows=='' and p.diagnostics['weak_without_literal']>=1


def test_complete_negative_state_mapping_is_not_outside_negation(tmp_path):
    path=data_dir(tmp_path)
    (path/'tags_index_v2.json').write_text(json.dumps({'general':[
        {'tag':'barefoot','count':100,'zh':['没穿鞋']}]}),encoding='utf-8')
    (path/'tag_gloss.json').write_text(json.dumps({'barefoot':{'g':'赤脚，脚上未穿鞋'}}),encoding='utf-8')
    assert 'barefoot' in packet(path,'没穿鞋').rows
    assert packet(path,'不要没穿鞋').rows==''


def test_nai_wrapper_is_included_in_actual_increment_budget(tmp_path):
    from nai_character_mode import build_nai_character_plan_prompt
    from nai_character_mode import resolve_nai_canvas
    p=packet(data_dir(tmp_path),'俯视，牛仔构图')
    canvas=resolve_nai_canvas({'width':1024,'height':1536},(1024,1536))
    original=build_nai_character_plan_prompt('俯视，牛仔构图','',canvas)
    enhanced=build_nai_character_plan_prompt('俯视，牛仔构图','',canvas,vocabulary_context=p.rows)
    assert 0 < len(enhanced)-len(original) <= 1200
    assert 'This is not a required tag checklist' in enhanced


@pytest.mark.parametrize('source', [
    '{}', '[]', '{"general":{},"other":[]}', '{"general":[],}',
    '{"general":[]', '{"general":[]} trailing', '{"other":1 "general":[]}',
])
def test_category_reader_rejects_malformed_or_missing_general_index(tmp_path,source):
    path=tmp_path/'index.json'
    path.write_text(source,encoding='utf-8')
    with pytest.raises((ValueError,json.JSONDecodeError)):
        _general_rows(path)


def test_category_reader_handles_escaped_names_and_does_not_mix_other_categories(tmp_path):
    path=tmp_path/'index.json'
    expected=[{'tag':'smile','zh':['微笑']}]
    path.write_text(json.dumps({'characters':[{'tag':'hidden'}], 'escaped"key':[1],
                                'general':expected,'artists':[]}),encoding='utf-8')
    assert _general_rows(path)==expected


def test_config_defaults_are_opt_in_and_survive_group_roundtrip():
    from config_defaults import flatten_config,group_config,schema_defaults
    schema=ROOT/'_conf_schema.json'
    defaults=flatten_config(schema_defaults(schema))
    assert defaults['toolbox_vocabulary_mode']=='off'
    assert defaults['toolbox_vocabulary_path']==''
    cfg={'toolbox_vocabulary_mode':'observe','toolbox_vocabulary_path':'local/tagdb','toolbox_vocabulary_max_chars':700}
    restored=flatten_config(group_config(cfg,schema))
    assert all(restored[k]==v for k,v in cfg.items())


@pytest.mark.parametrize('text', [
    '千早爱音公主抱若叶睦',
    '执事爱音公主抱淡绿色礼服若叶睦（禁止使用校服）',
    '陌生角色甲稳稳公主抱另一个角色乙，同时角色丙微笑',
    '角色甲穿着湿衣服并抱着角色乙',
])
def test_embedded_local_alias_survives_long_names_and_modifiers(tmp_path, text):
    path = embedded_data_dir(tmp_path)
    p = packet(path, text)
    expected = 'wet_clothes' if '湿衣服' in text else 'princess_carry'
    assert expected in {row['tag'] for row in p.diagnostics['candidates']}
    assert 'reverse_princess_carry' not in {row['tag'] for row in p.diagnostics['candidates']}
    assert p.diagnostics['added_chars_upper_bound'] <= 1200


def embedded_data_dir(tmp_path):
    path = data_dir(tmp_path)
    index = json.loads((path/'tags_index_v2.json').read_text(encoding='utf-8'))
    index['general'] += [
        dict(tag='princess_carry', count=1000, zh=['公主抱']),
        dict(tag='reverse_princess_carry', count=100, zh=['反向公主抱']),
    ]
    (path/'tags_index_v2.json').write_text(json.dumps(index, ensure_ascii=False), encoding='utf-8')
    gloss = json.loads((path/'tag_gloss.json').read_text(encoding='utf-8'))
    gloss.update(princess_carry=dict(g='托住背部与腿弯横抱'),
                 reverse_princess_carry=dict(g='性别角色颠倒的横抱'))
    (path/'tag_gloss.json').write_text(json.dumps(gloss, ensure_ascii=False), encoding='utf-8')
    return path


@pytest.mark.parametrize('text', [
    '千早爱音不要公主抱若叶睦',
    '禁止使用公主抱',
    '角色甲说：“公主抱若叶睦”',
    '公主抱这个词写在纸上',
    '心里的公主抱',
    '不要反向公主抱',
    '公主宝若叶睦',
])
def test_embedded_alias_keeps_negative_literal_and_nearby_counterexamples(tmp_path, text):
    p = packet(embedded_data_dir(tmp_path), text)
    assert not {'princess_carry', 'reverse_princess_carry'} & {
        row['tag'] for row in p.diagnostics['candidates']}


def test_longest_alias_and_independent_positive_occurrence(tmp_path):
    path = embedded_data_dir(tmp_path)
    p = packet(path, '角色甲反向公主抱角色乙')
    assert {row['tag'] for row in p.diagnostics['candidates']} == {'reverse_princess_carry'}
    p = packet(path, '角色甲不要公主抱，角色乙公主抱角色丙')
    assert {row['tag'] for row in p.diagnostics['candidates']} == {'princess_carry'}


def test_known_identity_span_does_not_turn_name_into_clothing(tmp_path):
    path = embedded_data_dir(tmp_path)
    index = json.loads((path/'tags_index_v2.json').read_text(encoding='utf-8'))
    index['general'].append(dict(tag='chihaya_(clothing)', zh=['千早']))
    (path/'tags_index_v2.json').write_text(json.dumps(index, ensure_ascii=False), encoding='utf-8')
    gloss = json.loads((path/'tag_gloss.json').read_text(encoding='utf-8'))
    gloss['chihaya_(clothing)'] = dict(g='巫女礼仪外衣')
    (path/'tag_gloss.json').write_text(json.dumps(gloss, ensure_ascii=False), encoding='utf-8')
    p = asyncio.run(ToolboxVocabulary().prepare(directory=str(path), mode='reference',
        text='千早爱音公主抱若叶睦', protected_phrases=('千早爱音','若叶睦')))
    assert {row['tag'] for row in p.diagnostics['candidates']} == {'princess_carry'}
