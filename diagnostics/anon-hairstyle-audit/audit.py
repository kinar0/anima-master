"""Replay historical prompts against current code; preserve baseline evidence.json."""
from __future__ import annotations

import ast
import json
import re
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from generation_task import insert_manual_prompt_suffix, split_manual_prompt_suffix
from prompt_background import framing_hidden_outfit_slots
from outfit_transfer import outfit_tag_slot

# Use the current implementation and its real helper dependencies. Importing the
# pipeline does not call any providers or generation endpoints.
from prompt_pipeline import appearance_dimensions, merge_authoritative_identity_block, WardrobeAuthority


def png_metadata(path):
    """Read PNG text chunks with the standard library; never decode pixels."""
    info = {}
    size = None
    with path.open('rb') as stream:
        assert stream.read(8) == b'\x89PNG\r\n\x1a\n'
        while header := stream.read(8):
            length, kind = struct.unpack('>I4s', header)
            payload = stream.read(length)
            stream.read(4)  # CRC; metadata audit does not edit the file.
            if kind == b'IHDR':
                size = list(struct.unpack('>II', payload[:8]))
            elif kind == b'tEXt':
                key, value = payload.split(b'\0', 1)
                info[key.decode('latin1')] = value.decode('latin1')
            elif kind == b'zTXt':
                key, value = payload.split(b'\0', 1)
                info[key.decode('latin1')] = zlib.decompress(value[1:]).decode('latin1')
            elif kind == b'iTXt':
                key, value = payload.split(b'\0', 1)
                compressed, method = value[:2]
                language, translated, text = value[2:].split(b'\0', 2)
                info[key.decode('latin1')] = (zlib.decompress(text) if compressed else text).decode('utf-8')
            if kind == b'IEND':
                break
    return size, info


def text_input(workflow, value):
    if isinstance(value, str):
        return value
    node = workflow[str(value[0])]
    if node['class_type'] == 'Textbox':
        return text_input(workflow, node['inputs']['text'])
    if node['class_type'] == 'ComfyUIToNovelAIV4':
        return text_input(workflow, node['inputs']['comfyui_prompt'])
    raise ValueError(f"Unrecognized text node: {node['class_type']}")


records = []
for path in (ROOT.parents[1] / 'plugin_data/astrbot_plugin_anima_master/tasks').glob('*.json'):
    task = json.loads(path.read_text(encoding='utf-8-sig'))
    request = task.get('original_prompt', '')
    if not task.get('time', '').startswith('2026-10-19'):
        continue
    if not ('千早爱音大头照' in request or '千早爱音超级大头照' in request
            or ('无优化' in request and 'chihaya anon' in request and 'two side' in request)):
        continue
    summary = task['prompt_summary']
    row = {k: task.get(k) for k in ('task_id', 'time', 'original_prompt', 'ok')}
    keys = ('raw_mode', 'semantic_plan_raw', 'prompt_llm_initial_content', 'final_prompt',
            'llm_prompt', 'semantic_character_outfits', 'wardrobe_authority',
            'required_core_tags', 'prompt_builder_template_customized',
            'semantic_plan_attempt_count', 'prompt_llm_attempt_count',
            'global_outfit_reinforcement_tags', 'low_cfg_harness',
            'danbooru_semantic_character_appearance_profiles')
    row['summary'] = {k: summary[k] for k in keys if k in summary}
    if not task.get('outputs'):
        row['submission_check'] = 'No saved output PNG'
        records.append(row)
        continue
    image_path = Path(task['outputs'][0])
    row['image_size'], metadata = png_metadata(image_path)
    workflow = json.loads(metadata['prompt'])
    generator = next(n for n in workflow.values() if n['class_type'] == 'NovelAIGenerator')
    # Whitelist text and model only. Do not save generator credential inputs.
    actual = text_input(workflow, generator['inputs']['prompt'])
    row['submitted_text_before_conversion'] = actual
    row['generator_model'] = generator['inputs'].get('model')
    if summary.get('raw_mode'):
        supplied = re.sub(r'^无优化\s+', '', request)
        row['submission_matches_supplied_raw'] = actual == supplied
        assert row['submission_matches_supplied_raw'], task['task_id']
    else:
        _, suffix = split_manual_prompt_suffix(request)
        expected = insert_manual_prompt_suffix(summary['final_prompt'], suffix)
        row['submission_matches_final_plus_suffix'] = actual == expected
        assert row['submission_matches_final_plus_suffix'], task['task_id']
    records.append(row)
records.sort(key=lambda row: row['time'])

profile = ('long_hair', 'pink_hair', 'grey_eyes', '(flat_chest:1.4)')
replays = []
for row in records:
    raw = row['summary'].get('prompt_llm_initial_content', '')
    found = re.search(r'\{Identity:\s*(.*?)\}', raw)
    if not found:
        continue
    changes = json.loads(row['summary']['semantic_plan_raw'])['characters'][0]['appearance_changes']
    opened = frozenset(c['dimension'] for c in changes if c.get('operation') != 'omit')
    replacements = frozenset(c['dimension'] for c in changes if c.get('operation') == 'replace')
    identity = found.group(1)
    merged = merge_authoritative_identity_block(
        'chihaya_anon', identity, profile, frozenset(),
        advisory_writer_dimensions=opened, replacement_writer_dimensions=replacements,
    )
    replays.append({'task_id': row['task_id'], 'raw_identity': identity,
                    'classified_dimensions': sorted(appearance_dimensions(identity)),
                    'replayed_identity': merged})

primary = next(r for r in records if r['task_id'] == '1368e69ee3e542d482e9078c86460599')
authority_data = primary['summary']['wardrobe_authority']
authority = WardrobeAuthority(**{
    key: tuple(value) if isinstance(value, list) else value
    for key, value in authority_data.items()
})
checks = {
    'hairstyle_tail_false_positive': sorted(appearance_dimensions('half-up twin-tail-like silhouette')),
    'unrelated_real_tail_control': sorted(appearance_dimensions('fox tail')),
    'chinese_headshot_hidden_slots': sorted(framing_hidden_outfit_slots('千早爱音超级大头照')),
    'english_headshot_hidden_slots': sorted(framing_hidden_outfit_slots('headshot')),
    'english_headshot_with_upper_hidden_slots': sorted(framing_hidden_outfit_slots('headshot, upper body')),
    'necktie_slot': outfit_tag_slot('green_necktie'),
    'long_sleeves_slot': outfit_tag_slot('long_sleeves'),
    'hair_ribbon_wardrobe_filter_result': authority.filter_tags(('hair_ribbon',)),
}
assert 'hair_style' in checks['hairstyle_tail_false_positive']
assert 'tail' not in checks['hairstyle_tail_false_positive']
assert 'tail' in checks['unrelated_real_tail_control']
assert checks['hair_ribbon_wardrobe_filter_result'] == ()
assert 'upper_body.neckwear' in checks['chinese_headshot_hidden_slots']
baseline = json.loads((OUT / 'evidence.json').read_text(encoding='utf-8'))

evidence = {'date': '2026-10-19', 'records': records,
            'identity_replays_current_source': replays, 'deterministic_checks': checks,
            'baseline_deterministic_checks': baseline['deterministic_checks'],
            'wardrobe_replay_note': 'Historical request-wide authority has no new wearer-scoped hairstyle accessory permissions; that new binding is covered by tests/test_appearance_reference_scope.py.',
            'scope': 'Historical records and PNG metadata only; no pixel assessment, API calls, configuration changes or production source changes.'}
(OUT / 'evidence-current.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'records': len(records), 'optimized': sum(not r['summary'].get('raw_mode') for r in records),
                  'raw': sum(bool(r['summary'].get('raw_mode')) for r in records),
                  'verified_png_submissions': sum('submitted_text_before_conversion' in r for r in records),
                  'all_available_submissions_match': True, 'checks': checks}, ensure_ascii=False, indent=2))
