"""Real writer evidence; does not submit images or alter runtime configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from test_deepseek_prompt_e2e import live_pipeline, _build
from nai_text import split_nai_text_tail


pytestmark = [pytest.mark.live_llm, pytest.mark.skipif(
    os.getenv('ANIMA_RUN_LIVE_LLM_E2E') != '1', reason='explicit live provider opt-in required',
)]


@pytest.mark.parametrize('case,prompt,texts', [
    ('sign', '一个粉发女孩穿白色连衣裙，举着一块木牌，木牌上写着“欢迎光临，C#社！”；文字只在木牌上。', ['欢迎光临，C#社！']),
    ('dialogue', '-r 千早爱音穿白衬衫站在左边说：“你好，祥子！”；丰川祥子穿黑色连衣裙站在右边回答：“Hello, Anon! (OK)”；两个人的短台词各放在自己的气泡中。', ['你好，祥子！', 'Hello, Anon! (OK)']),
    ('negative', '-r 千早爱音穿“白色连衣裙”，丰川祥子穿黑色连衣裙站在她旁边；两人身旁放着空白木牌，不要写“欢迎”，无对白。', []),
])
def test_live_native_text(live_pipeline, tmp_path, case, prompt, texts):
    pipeline, context = live_pipeline
    pipeline.config.update(custom_workflow_enabled=True, custom_workflow_path='nai_api.json',
                           comic_dialogue_enabled=False, debug_prompt_enabled=True)
    result, calls = _build(live_pipeline, prompt)
    directory = Path(os.getenv('ANIMA_LIVE_REPORT_DIR') or tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'nai_text_{case}.json').write_text(json.dumps(
        {'request': prompt, 'calls': calls, 'summary': result.summary, 'final_prompt': result.final_prompt},
        ensure_ascii=False, indent=2,
    ), encoding='utf-8')
    assert result.final_prompt, result.summary
    visual, payload = split_nai_text_tail(result.final_prompt)
    assert payload is None  # these examples bind all positive text to visible owners
    assert 'NAILITERAL' not in result.final_prompt
    if texts:
        writer = next(call for call in calls if 'NAI native lettering contract' in call['prompt'])
        # Exact quoted originals are also protected before visual filtering.
        for index, text in enumerate(texts):
            assert f'NAILITERAL{index:04d}TOKEN' in writer['response'] or text in writer['response']
        assert len(result.summary['nai_text_regions']) == len(texts)
        for region in result.summary['nai_text_regions']:
            owner = result.summary['nai_characters'][region['character_index']]
            assert texts[region['id']] in split_nai_text_tail(owner['prompt'])[1]
    if case == 'dialogue':
        rows = result.summary['nai_characters']
        anon = next(row for row in rows if row['name'] == 'chihaya_anon')
        sakiko = next(row for row in rows if row['name'] == 'togawa_sakiko')
        assert texts[0] in anon['prompt'] and texts[1] not in anon['prompt'], rows
        assert texts[1] in sakiko['prompt'] and texts[0] not in sakiko['prompt'], rows


@pytest.mark.parametrize('explicit_r', [False, True])
def test_live_comic_text_scopes(live_pipeline, tmp_path, explicit_r):
    pipeline, _context = live_pipeline
    pipeline.config.update(custom_workflow_enabled=True, custom_workflow_path='nai_api.json',
                           comic_dialogue_enabled=False, debug_prompt_enabled=True)
    source = (
        '两格漫画，左上A格在上半幅，右下B格在下半幅。'
        '左上A格：千早爱音穿白衬衫，丰川祥子穿黑色连衣裙，两人站立；'
        '千早爱音：“你好，祥子！”；丰川祥子：“等等！”；左上A格右上角旁白：“放学后。”\n'
        '右下B格：千早爱音仍穿白衬衫，独自站立；千早爱音：“再见！”；右下B格左下角旁白：“第二天。”'
    )
    result, calls = _build(live_pipeline, ('-r ' if explicit_r else '') + source)
    directory = Path(os.getenv('ANIMA_LIVE_REPORT_DIR') or tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'nai_text_comic_{explicit_r}.json').write_text(json.dumps(
        {'request': source, 'calls': calls, 'summary': result.summary, 'final_prompt': result.final_prompt},
        ensure_ascii=False, indent=2,
    ), encoding='utf-8')
    assert result.final_prompt, result.summary.get('llm_error')
    assert result.summary['nai_text_auto_mode'] is not explicit_r
    assert split_nai_text_tail(result.final_prompt)[1] == '放学后。\n\n第二天。'
    regions = result.summary['nai_text_regions']
    assert len(regions) == 5
    rows = result.summary['nai_characters']
    assert len(rows) == 3
    expected = {0: ('chihaya_anon', '你好，祥子！', 'A格'),
                1: ('togawa_sakiko', '等等！', 'A格'),
                3: ('chihaya_anon', '再见！', 'B格')}
    owners = {}
    for index, (name, text, panel) in expected.items():
        region = next(region for region in regions if region['id'] == index)
        owner = rows[region['character_index']]
        owners[index] = region['character_index']
        assert owner['name'] == name
        assert panel in region['panel'] and panel in owner['prompt']
        assert split_nai_text_tail(owner['prompt'])[1] == text
    assert owners[0] != owners[3]
    assert regions[2]['character_index'] is None and regions[4]['character_index'] is None
    assert '右上' in regions[2]['placement'] and '左下' in regions[4]['placement']
    assert 'NAILITERAL' not in result.final_prompt
    assert all('NAILITERAL' not in row['prompt'] for row in rows)
