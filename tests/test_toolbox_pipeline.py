"""Real prompt builders with controlled model output: host boundaries, not model quality."""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from types import SimpleNamespace

import pytest

from test_prompt_direct_writer import run, plan, Resolver, semantic_character
from test_toolbox_vocabulary import data_dir, embedded_data_dir


NORMAL = ('{Count: 1girl}{Characters: character_a}{Copyright: }'
          '{Identity: character_a: pink hair, grey eyes}'
          '{Details: character_a: standing}{Tags: white background}{Nltags: }')


def generate(path, *, mode='reference', nai=False, prompt='角色甲，俯视', resolver=None, config=None, output=None):
    cfg={'toolbox_vocabulary_path':str(path),'toolbox_vocabulary_mode':mode,'debug_prompt_enabled':True,**(config or {})}
    writer = output or (plan([('character_a','girl, standing, 站立。',.5)],'1girl, white background') if nai else NORMAL)
    return run(('-r ' if nai else '')+prompt,{'characters':[semantic_character('角色甲')]},writer,cfg,resolver=resolver)


@pytest.mark.parametrize('nai',[False,True])
def test_observe_is_input_identical_and_reference_adds_only_bounded_second_stage_evidence(tmp_path,nai):
    path=data_dir(tmp_path)
    base,base_calls=generate(path,mode='off',nai=nai)
    observed,observe_calls=generate(path,mode='observe',nai=nai)
    active,calls=generate(path,nai=nai)
    assert base.final_prompt==observed.final_prompt==active.final_prompt
    assert len(base_calls)==len(observe_calls)==len(calls)==2
    assert base_calls==observe_calls
    assert base_calls[0]==calls[0]
    assert {k:v for k,v in base_calls[1].items() if k!='prompt'}=={k:v for k,v in calls[1].items() if k!='prompt'}
    assert all(op in {'equal','insert'} for op,*_ in SequenceMatcher(
        None,base_calls[1]['prompt'],calls[1]['prompt'],autojunk=False).get_opcodes())
    assert '相机从上方向下拍摄' in calls[1]['prompt']
    assert 0 < len(calls[1]['prompt'])-len(base_calls[1]['prompt']) <= 1200
    assert active.summary['toolbox_vocabulary']['state']=='ready'
    assert observed.summary['toolbox_vocabulary']['candidates']
    assert 'from above' not in active.final_prompt
    assert active.summary.get('danbooru_term_fallback_tags',[])==[]


@pytest.mark.parametrize('nai',[False,True])
def test_no_hits_or_missing_data_keep_writer_input_identical(tmp_path,nai):
    path=data_dir(tmp_path)
    base,base_calls=generate(path,nai=nai,mode='off',prompt='角色甲，xxxxxxxxxxxxxxxx')
    active,calls=generate(path,nai=nai,prompt='角色甲，xxxxxxxxxxxxxxxx')
    missing,missing_calls=generate(path/'missing',nai=nai,prompt='角色甲，xxxxxxxxxxxxxxxx')
    assert calls==base_calls==missing_calls
    assert active.final_prompt==base.final_prompt==missing.final_prompt
    assert missing.summary['toolbox_vocabulary']['state']=='unavailable'


@pytest.mark.parametrize('nai',[False,True])
def test_term_switch_disables_new_retrieval_without_touching_character_authority(tmp_path,nai):
    result,calls=generate(data_dir(tmp_path),nai=nai,config={'danbooru_term_mappings_enabled':False})
    assert result.summary['toolbox_vocabulary']['mode']=='off'
    assert '相机从上方向下拍摄' not in calls[-1]['prompt']
    baseline,baseline_calls=generate(tmp_path,nai=nai,mode='off',config={'danbooru_term_mappings_enabled':False})
    assert calls==baseline_calls and result.final_prompt==baseline.final_prompt
    assert 'character_a' in calls[-1]['prompt']


@pytest.mark.parametrize('nai',[False,True])
def test_manual_mapping_is_retained_and_explanation_is_merged_once(tmp_path,nai):
    class Mapped(Resolver):
        def matched_term_mappings_for_prompt(self,_text): return (('牛仔构图','cowboy_shot'),)
        def cached_term_mappings_for_prompt(self,_text):
            return SimpleNamespace(candidate_tags=(),confirmed_tags=('cowboy_shot',))
    active,calls=generate(data_dir(tmp_path),nai=nai,prompt='角色甲，牛仔构图',resolver=Mapped())
    text=calls[-1]['prompt']
    assert text.count('牛仔构图 => cowboy_shot')==1
    assert text.count('cowboy_shot 释义：')==1
    assert '与牛仔服饰无关' in text
    assert len(calls)==2
    # A writer ignoring this optional vocabulary remains valid.
    assert active.final_prompt


def test_raw_and_optimization_disabled_do_not_even_load_the_index(tmp_path):
    path=data_dir(tmp_path)
    for prompt,config in [('原样 1girl, cowboy_shot',{}),('1girl, cowboy_shot',{'prompt_optimize_enabled':False})]:
        result,calls=generate(path,prompt=prompt,config=config)
        assert calls==[]
        assert 'toolbox_vocabulary' not in result.summary


def test_nai_structure_repair_reuses_same_bounded_packet_once(tmp_path):
    path=data_dir(tmp_path)
    valid=plan([('character_a','girl, standing, 站立。',.5)],'1girl, white background')
    config={'toolbox_vocabulary_mode':'reference','toolbox_vocabulary_path':str(path),'debug_prompt_enabled':True}
    result,calls=run('-r 角色甲，俯视',{'characters':[semantic_character('角色甲')]},'{bad json',config,repair=valid)
    assert result.final_prompt and len(calls)==3
    for call in calls[1:]:
        assert call['prompt'].count('from_above — 相机从上方向下拍摄')==1
    assert result.summary['nai_character_plan_attempt_count']==2


def test_optional_candidates_do_not_change_multi_instance_wardrobe_or_population(tmp_path):
    path=data_dir(tmp_path)
    output=plan([('character_a','girl, white_shirt, 穿着白衬衫站立。',.25),
                 ('character_a','girl, black_coat, 穿着黑色外套站立。',.75)],'2girls, white background')
    result,calls=generate(path,nai=True,prompt='左格角色甲穿白衬衫，右格角色甲穿黑外套，俯视',output=output)
    assert len(calls)==2 and result.final_prompt
    characters=result.summary['nai_characters']
    assert len(characters)==2
    assert 'white' in characters[0]['prompt'] and 'black' in characters[1]['prompt']
    assert all('from_above' not in r['prompt'] for r in characters)


@pytest.mark.parametrize('nai', [False, True])
def test_embedded_action_reaches_existing_writer_without_new_call_or_forced_tag(tmp_path, nai):
    path = embedded_data_dir(tmp_path)
    prompt = '角色甲稳稳公主抱角色乙（禁止使用校服）'
    output = (plan([('character_a','girl, standing, 站立。',.4),
                    ('character_b','girl, lying, 躺着。',.6)],'2girls, white background')
              if nai else '{Count: 2girls}{Characters: character_a, character_b}{Copyright: }'
              '{Identity: character_a: pink hair | character_b: blue hair}'
              '{Details: character_a: standing | character_b: lying}'
              '{Tags: white background}{Nltags: }')
    result, calls = run(('-r ' if nai else '')+prompt,
        {'characters':[semantic_character('角色甲'), semantic_character('角色乙')]},
        output, {'toolbox_vocabulary_path':str(path), 'toolbox_vocabulary_mode':'reference'})
    assert len(calls) == 2
    assert '- 公主抱 => princess_carry — 托住背部与腿弯横抱' in calls[-1]['prompt']
    assert 'reverse_princess_carry' not in calls[-1]['prompt']
    assert 'princess' not in result.final_prompt
    assert result.summary['toolbox_vocabulary']['added_chars_upper_bound'] <= 1200
