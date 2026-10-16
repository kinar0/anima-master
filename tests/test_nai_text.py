from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_tools.comfyui_workflows import custom_t2i_workflow
from generation_task import insert_manual_prompt_suffix, split_manual_prompt_suffix
from nai_text import NaiTextRequest, extract_nai_text_request, split_nai_text_tail
from test_prompt_direct_writer import run, semantic_character, plan
from test_prompt_strategy import _pipeline


@pytest.mark.parametrize("description,expected", [
    ('女孩举牌，牌子上写着“你好，世界！(v2) [OK] C#”', ('你好，世界！(v2) [OK] C#',)),
    ('左边女孩说：“你好！”；右边女孩回答：“またね。”', ('你好！', 'またね。')),
    ('a sign reads "Hello,  world!"', ('Hello,  world!',)),
    ('漫画台词：甲：你好\n乙：再见\n正向提示词：两人站立', ('你好', '再见')),
    ('女孩穿“白色婚纱”，手里拿着“鲜花”', ()),
    ('印度女孩穿“白色婚纱”', ()),
    ('写实风格女孩穿“白色婚纱”', ()),
    ('招牌旁的女孩穿“白色婚纱”', ()),
    ('不要写“hello”，只有一块空白牌子', ()),
    ('牌子上不显示“red hair”，但女孩有红发', ()),
    ('无对话，仅惊吓音效气泡', ()),
    ('海报上写着“星光”，角色说“你好”', ('星光', '你好')),
])
def test_only_requested_literal_lettering_is_extracted(description, expected):
    assert extract_nai_text_request(description)[1].texts == expected


def test_existing_post_lettering_switch_avoids_duplicate_native_dialogue():
    assert extract_nai_text_request('甲说：“你好”', include_dialogue=False)[1].texts == ()
    assert extract_nai_text_request('漫画台词：甲：“你好”', include_dialogue=False)[1].texts == ()
    assert extract_nai_text_request('招牌写着“你好”', include_dialogue=False)[1].texts == ('你好',)


def test_tail_is_opaque_to_suffix_and_literal_hash_is_not_a_manual_switch():
    text = 'Hi,  (x:1.2), {A}, [b], artist:me :: C#\n\n你好！\n第二行'
    prompt = f'1girl, text Text: {text}'
    assert split_manual_prompt_suffix(prompt) == (prompt, '')
    assert split_manual_prompt_suffix('牌上写“C#”#backlighting') == ('牌上写“C#”', 'backlighting')
    result = insert_manual_prompt_suffix(prompt, '(watercolor:1.2)')
    assert split_nai_text_tail(result) == ('1girl, text, (watercolor:1.2)', text)
    assert split_nai_text_tail('a sign reads "Text: Hello"') == ('a sign reads "Text: Hello"', None)


def test_protection_cannot_grant_appearance_permissions_to_unquoted_tag_words():
    request = NaiTextRequest(('red',))
    assert request.protect('red hair, a sign reads "red"') == 'red hair, a sign reads "NAILITERAL0000TOKEN"'
    assert request.restore(request.protect('red hair, a sign reads "red"')) == 'red hair, a sign reads "red"'


def test_a_repeated_literal_text_marker_cannot_start_an_earlier_tail():
    request = NaiTextRequest(('Text: A, (B:2)',))
    result = request.finish('sign, a sign reads NAILITERAL0000TOKEN')
    assert split_nai_text_tail(result)[1] == 'Text: A, (B:2)'


def test_lettering_does_not_displace_a_dataset_selector():
    assert NaiTextRequest(('HELLO',)).finish('background dataset, sign').startswith('background dataset, sign')


@pytest.mark.parametrize("model,limit", [('NAI Diffusion V5 Full', 750), ('nai-diffusion-5-curated', 374), ('NAI Diffusion V4.5 Full', 118)])
def test_exact_model_limits_are_checked_without_truncation(model, limit):
    NaiTextRequest(('A' * limit,)).validate(model)
    with pytest.raises(ValueError, match='nai_text_too_long'):
        NaiTextRequest(('A' * (limit + 1),)).validate(model)


def test_old_model_and_non_english_fail_before_generation():
    with pytest.raises(ValueError, match='requires_v5'):
        NaiTextRequest(('你好',)).validate('NAI Diffusion V4.5 Full')
    with pytest.raises(ValueError, match='requires_v4'):
        NaiTextRequest(('Hello',)).validate('NAI Diffusion Anime V3')
    with pytest.raises(ValueError, match='nai_text_empty'):
        NaiTextRequest(explicit_payload='').validate('NAI Diffusion V5 Full')


def _nai_config():
    return dict(custom_workflow_enabled=True, custom_workflow_path='nai_api.json',
                debug_prompt_enabled=True, quality_prefix='best quality, no text')


def _with_regions(writer, source, assignments):
    value = json.loads(writer)
    value['text_regions'] = [dict(id=i, kind=kind, character_index=owner, panel=panel,
                                  placement=placement, source_text=[source])
                              for i, kind, owner, panel, placement in assignments]
    return json.dumps(value, ensure_ascii=False)


@pytest.mark.parametrize('regional', [False, True])
def test_real_pipeline_preserves_text_after_all_visual_filters(regional):
    request = '角色甲举牌，牌子写着“Hello,  (x:2), C#”；角色乙说：“你好！”'
    semantic = {'characters': [semantic_character('角色甲'), semantic_character('角色乙')]}
    writer = plan([
            ('character_a', 'girl, pink hair, holding sign, 角色甲举牌写着NAILITERAL0000TOKEN。', .3),
            ('character_b', 'girl, blue hair, speaking, 角色乙说NAILITERAL0001TOKEN。', .7),
        ], '2girls, text, speech bubble, 上半身合影，左边举牌，右边说话。')
    writer = _with_regions(writer, request, [(0, 'surface', 0, '', '左边角色甲的牌子写NAILITERAL0000TOKEN。'),
                                             (1, 'speech', 1, '', '右边角色乙的气泡写NAILITERAL0001TOKEN。')])
    result, calls = run(('-r ' if regional else '') + request, semantic, writer, _nai_config())
    assert result.final_prompt, result.summary
    visual, payload = split_nai_text_tail(result.final_prompt)
    assert payload is None
    assert 'no text' not in visual
    assert 'NAILITERAL' not in result.final_prompt
    assert len(calls) == 2
    assert 'NAI native lettering contract' in calls[1]['prompt']
    graph = custom_t2i_workflow({'custom_workflow_path': 'nai_api.json'}, result.final_prompt,
                              'lowres, text, chinese text, bad anatomy', 1024, 1024, 28, 6, 1,
                              nai_characters=result.summary.get('nai_characters'))
    assert graph['3']['inputs']['text'] == result.final_prompt
    assert graph['11']['inputs']['text'] == 'lowres, bad anatomy'
    rows = result.summary['nai_characters']
    assert split_nai_text_tail(rows[0]['prompt'])[1] == 'Hello,  (x:2), C#'
    assert split_nai_text_tail(rows[1]['prompt'])[1] == '你好！'
    assert 'Hello' not in rows[1]['prompt']
    assert result.summary['nai_text_auto_mode'] is (not regional)


def test_raw_and_disabled_modes_preserve_native_multiline_tail():
    payload = 'Hello,  world!\n\n(你好:2) # [OK]'
    for config, prefix in ((_nai_config(), '无优化 '), ({**_nai_config(), 'prompt_optimize_enabled': False}, '')):
        result = asyncio.run(_pipeline(config).build(object(), f'{prefix}text, 1girl Text: {payload}'))
        assert result.final_prompt == f'text, 1girl Text: {payload}'
    # Natural-language quotes do not override the raw bypass contract.
    result = asyncio.run(_pipeline(_nai_config()).build(object(), '无优化 1girl, sign reads "Hello"'))
    assert result.final_prompt == '1girl, sign reads "Hello"'


def test_text_from_reference_context_does_not_become_requested_lettering():
    semantic = {'characters': [semantic_character('角色甲')]}
    result, _ = run('角色甲站立。参考图：牌子写着“广告”', semantic,
                    '{Count: 1girl}{Characters: character_a}{Copyright: }'
                    '{Identity: character_a has pink hair}{Details: character_a stands}'
                    '{Tags: standing}{Nltags: A girl stands.}',
                    _nai_config(), original_prompt='角色甲站立')
    assert result.final_prompt
    assert split_nai_text_tail(result.final_prompt)[1] is None


def test_explicit_tail_keeps_augmented_reference_visual_context():
    original = '角色甲举着牌子 Text: 你好'
    writer = _with_regions(plan([('character_a', 'girl, pink hair, holding sign', .5)], '1girl, simple background'),
                           original, [(0, 'surface', 0, '', '牌子上写着NAILITERAL0000TOKEN。')])
    result, calls = run(original + '\n参考图服装：white shirt',
                        {'characters': [semantic_character('角色甲')]},
                        writer,
                        _nai_config(), original_prompt=original)
    assert split_nai_text_tail(result.summary['nai_characters'][0]['prompt'])[1] == '你好'
    assert '参考图服装：white shirt' in calls[-1]['prompt']


def test_submission_validates_raw_lettering_against_the_actual_graph_model(tmp_path):
    graph = json.loads((Path(__file__).resolve().parents[1] / 'nai_api.json').read_text())
    graph['1']['inputs']['model'] = 'NAI Diffusion V4.5 Full'
    path = tmp_path / 'nai_v4.json'
    path.write_text(json.dumps(graph))
    with pytest.raises(SystemExit, match='requires_v5'):
        custom_t2i_workflow({'custom_workflow_path': str(path)}, 'text Text: 中文', '', 1024, 1024, 28, 6, 1)


def _comic_fixture():
    source = ('左上A格，角色甲：“你好！”；角色乙：“等等！”；左上A格右上角，旁白：“夜晚。”\n'
              '右下B格，角色甲：“再见！”；右下B格左下角，旁白：“次日。”')
    writer = _with_regions(plan([
        ('character_a', 'girl, pink hair, speaking, 左上A格的角色甲说话。', .2),
        ('character_b', 'girl, blue hair, speaking, 左上A格的角色乙回答。', .4),
        ('character_a', 'girl, pink hair, speaking, 右下B格的角色甲说话。', .75),
    ], '3girls, comic, 2panels, 上左和下右两个格子，角色甲在两格出现，角色乙仅在A格。'), source, [
        (0, 'speech', 0, '左上A格', '左上A格角色甲头顶左侧的气泡写NAILITERAL0000TOKEN。'),
        (1, 'speech', 1, '左上A格', '左上A格角色乙头顶右侧的气泡写NAILITERAL0001TOKEN。'),
        (2, 'narration', None, '左上A格', '左上A格右上角的旁白框写NAILITERAL0002TOKEN。'),
        (3, 'speech', 2, '右下B格', '右下B格角色甲头顶气泡写NAILITERAL0003TOKEN。'),
        (4, 'narration', None, '右下B格', '右下B格左下角的旁白框写NAILITERAL0004TOKEN。'),
    ])
    return source, writer


@pytest.mark.parametrize('regional', [False, True])
def test_comic_routes_by_panel_instance_and_keeps_narration_position(regional):
    source, writer = _comic_fixture()
    result, calls = run(('-r ' if regional else '') + source,
                        {'characters': [semantic_character('角色甲'), semantic_character('角色乙')]},
                        writer, _nai_config())
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert split_nai_text_tail(result.final_prompt)[1] == '夜晚。\n\n次日。'
    assert '左上A格右上角' in result.final_prompt and '右下B格左下角' in result.final_prompt
    rows = result.summary['nai_characters']
    assert [split_nai_text_tail(row['prompt'])[1] for row in rows] == ['你好！', '等等！', '再见！']
    assert len(rows) == 3 and rows[0]['name'] == rows[2]['name']
    graph = custom_t2i_workflow({'custom_workflow_path': 'nai_api.json'}, result.final_prompt,
                              'text, bad anatomy', 1024, 1024, 28, 6, 1, nai_characters=rows)
    selector = graph[graph['1']['inputs']['characterPrompts'][0]]['inputs']
    for index, row in enumerate(rows, 1):
        converter = graph[selector[f'character{index}'][0]]
        assert converter['inputs']['comfyui_prompt'] == row['prompt']


@pytest.mark.parametrize('repair_text', [False, True])
def test_direction_downgrade_does_not_consume_or_weaken_text_repair(repair_text):
    source, writer = _comic_fixture()
    data = json.loads(writer)
    data['characters'][2]['interaction_tags'] = ['source#talking']
    fixed = json.dumps(data, ensure_ascii=False)
    if repair_text:
        data['text_regions'][3]['character_index'] = 0  # wrong panel instance
    initial = json.dumps(data, ensure_ascii=False)
    result, calls = run(source, {'characters': [semantic_character('角色甲'), semantic_character('角色乙')]},
                        initial, _nai_config(), repair=fixed if repair_text else None)
    assert result.final_prompt, result.summary
    assert len(calls) == (3 if repair_text else 2)
    assert result.summary['nai_interaction_initial_validation_errors']
    assert bool(result.summary['nai_text_initial_validation_errors']) is repair_text
    if repair_text:
        assert result.summary['nai_interaction_retry_validation_errors']
        assert result.summary['nai_text_retry_validation_errors'] == []
    rows = result.summary['nai_characters']
    assert rows[2]['interaction_tags'] == []
    assert '右下B格的角色甲说话。' in rows[2]['prompt']
    assert [split_nai_text_tail(row['prompt'])[1] for row in rows] == ['你好！', '等等！', '再见！']


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'invalid_owner', 'cross_panel', 'narrator_owner', 'invented_panel', 'panel_mismatch', 'ignore_speaker'])
def test_invalid_comic_allocation_repairs_once_then_stops(defect):
    source, writer = _comic_fixture()
    data = json.loads(writer)
    if defect == 'missing': data.pop('text_regions')
    if defect == 'duplicate': data['text_regions'].append(data['text_regions'][0])
    if defect == 'invalid_owner': data['text_regions'][0]['character_index'] = 10
    if defect == 'cross_panel': data['text_regions'][3]['character_index'] = 0
    if defect == 'narrator_owner': data['text_regions'][2]['character_index'] = 0
    if defect == 'invented_panel': data['text_regions'][0]['panel'] = '中间C格'
    if defect == 'panel_mismatch': data['characters'][0]['prompt'] = 'girl, speaking, 右下B格的角色甲。'
    if defect == 'ignore_speaker': data['text_regions'][0].update(kind='ignore', character_index=None, placement='只是引用')
    data['characters'][2]['interaction_tags'] = ['source#talking']
    broken = json.dumps(data, ensure_ascii=False)
    result, calls = run(source, {'characters': [semantic_character('角色甲'), semantic_character('角色乙')]},
                        broken, _nai_config(), repair=broken)
    assert not result.final_prompt
    assert len(calls) == 3
    assert result.summary['skipped_reason'] == 'nai_text_plan_failed'
    assert 'nai_text_' in calls[2]['prompt']


def test_model_cannot_replace_the_immutable_source_span():
    source, writer = _comic_fixture()
    data = json.loads(writer)
    data['text_regions'][0]['source_text'] = ['未在原文中的台词与分格']
    result, _ = run(source, {'characters': [semantic_character('角色甲'), semantic_character('角色乙')]},
                    json.dumps(data, ensure_ascii=False), _nai_config())
    assert result.final_prompt, result.summary
    region = result.summary['nai_text_regions'][0]
    assert region['source_text'] in source
    start, end = region['source_span']
    assert source[start:end] == '你好！'


def test_stray_literal_copies_are_projected_using_the_validated_map():
    source, writer = _comic_fixture()
    data = json.loads(writer)
    data['global_prompt'] += '，角色乙说NAILITERAL0001TOKEN。'
    data['characters'][2]['prompt'] += '，右下B格背景中保留一棵树，并说“NAILITERAL0001TOKEN”。'
    result, calls = run(source, {'characters': [semantic_character('角色甲'), semantic_character('角色乙')]},
                        json.dumps(data, ensure_ascii=False), _nai_config())
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert '等等！' not in result.final_prompt
    third = result.summary['nai_characters'][2]['prompt']
    assert '等等！' not in third and '一棵树' in third
    assert split_nai_text_tail(third)[1] == '再见！'
    assert result.summary['nai_dropped_text_references']


def test_identical_lines_keep_distinct_occurrence_ids():
    source = '左上A格角色甲：“好。”\n右下B格角色甲：“好。”'
    _, request = extract_nai_text_request(source)
    assert request.texts == ('好。', '好。')
    writer = _with_regions(plan([('character_a', 'girl, speaking, 左上A格的角色甲。', .2), ('character_a', 'girl, speaking, 右下B格的角色甲。', .8)],
                                '2girls, comic'), source, [
        (0, 'speech', 0, '左上A格', '左上A格的气泡写NAILITERAL0000TOKEN。'),
        (1, 'speech', 1, '右下B格', '右下B格的气泡写NAILITERAL0001TOKEN。'),
    ])
    result, _ = run(source, {'characters': [semantic_character('角色甲')]}, writer, _nai_config())
    assert result.final_prompt, result.summary
    assert [row['id'] for row in result.summary['nai_text_regions']] == [0, 1]
    assert [split_nai_text_tail(row['prompt'])[1] for row in result.summary['nai_characters']] == ['好。', '好。']


def test_no_humans_caption_does_not_invent_a_narrator_character():
    source = '无人风景画，右上角标题：“开幕”'
    writer = _with_regions(plan([], 'no humans, landscape, 无人的风景，标题位于右上角。'), source,
                           [(0, 'narration', None, '', '右上角标题框写NAILITERAL0000TOKEN。')])
    result, _ = run(source, {'characters': []}, writer, _nai_config())
    assert result.final_prompt, result.summary
    assert result.summary['nai_characters'] == []
    assert split_nai_text_tail(result.final_prompt)[1] == '开幕'


def test_empty_lettering_scene_clears_exported_example_character_boxes(tmp_path):
    graph = json.loads((Path(__file__).resolve().parents[1] / 'nai_api.json').read_text())
    graph['1']['inputs']['characterPrompts'] = ['20', 0]
    graph['20'] = {'class_type': 'CharacterPromptSelect', 'inputs': {'character1': 'example girl'}}
    path = tmp_path / 'with_example.json'
    path.write_text(json.dumps(graph))
    result = custom_t2i_workflow({'custom_workflow_path': str(path)}, 'no humans Text: HELLO', '',
                                1024, 1024, 28, 6, 1, nai_characters=[])
    assert 'characterPrompts' not in result['1']['inputs']


def test_character_only_tail_still_removes_conflicting_negative_text_tags():
    rows = [dict(name='a', prompt='girl, speech bubble Text: HELLO', x=.5, y=.5)]
    graph = custom_t2i_workflow({'custom_workflow_path': 'nai_api.json'}, '1girl, comic',
                              'text, lowres', 1024, 1024, 28, 6, 1, nai_characters=rows)
    assert graph['11']['inputs']['text'] == 'lowres'


def test_model_limit_applies_to_all_tails_combined(tmp_path):
    graph = json.loads((Path(__file__).resolve().parents[1] / 'nai_api.json').read_text())
    graph['1']['inputs']['model'] = 'NAI Diffusion V4.5 Full'
    path = tmp_path / 'nai_v4.json'
    path.write_text(json.dumps(graph))
    rows = [dict(name=f'p{i}', prompt='girl Text: ' + 'A' * 70, x=.5, y=.5) for i in range(2)]
    with pytest.raises(SystemExit, match='nai_text_too_long'):
        custom_t2i_workflow({'custom_workflow_path': str(path)}, '2girls', '', 1024, 1024, 28, 6, 1,
                            nai_characters=rows)
