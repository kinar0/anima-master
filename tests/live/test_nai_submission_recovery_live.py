"""Real neutral writer outputs for source-named and writer-designed panels."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nai_text import split_nai_text_tail
from test_deepseek_prompt_e2e import live_pipeline, _build


pytestmark = [pytest.mark.live_llm, pytest.mark.skipif(
    os.getenv("ANIMA_RUN_LIVE_LLM_E2E") != "1", reason="explicit live provider opt-in required",
)]


@pytest.mark.parametrize("named", [False, True])
def test_live_source_panel_contract(live_pipeline, tmp_path, named):
    pipeline, _context = live_pipeline
    pipeline.config.update(custom_workflow_enabled=True, custom_workflow_path="nai_api.json",
                           comic_dialogue_enabled=False, debug_prompt_enabled=True)
    if named:
        source = ('-r 两格漫画。左上A格：丰川祥子穿黑色外套挥手说“你好”。\n'
                  '右下B格：丰川祥子仍穿黑色外套，坐下说“再见”。')
        identity = "togawa_sakiko"
    else:
        source = ('-r 两格漫画，千早爱音穿白衬衫先挥手说“你好”，随后坐下说“再见”。'
                  '上下排列，每格各有同一人物的一次出场，台词各在所属人物头顶的气泡内。')
        identity = "chihaya_anon"
    result, calls = _build(live_pipeline, source)
    out = Path(os.getenv("ANIMA_LIVE_REPORT_DIR") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"source_panels_{named}.json").write_text(json.dumps(
        dict(request=source, calls=calls, summary=result.summary, final_prompt=result.final_prompt),
        ensure_ascii=False, indent=2), encoding="utf-8")
    assert result.final_prompt, result.summary.get("llm_error")
    rows = result.summary["nai_characters"]
    assert len(rows) == 2 and all(row["name"] == identity for row in rows)
    regions = result.summary["nai_text_regions"]
    assert len(regions) == 2
    assert regions[0]["character_index"] != regions[1]["character_index"]
    for text, region in zip(("你好", "再见"), regions):
        assert split_nai_text_tail(rows[region["character_index"]]["prompt"])[1] == text
        if named:
            assert ("左上A格" if region["id"] == 0 else "右下B格") in region["panel"]
        else:
            assert region["panel"] == ""
    assert all("NAILITERAL" not in row["prompt"] for row in rows)
    assert all(any("\u4e00" <= ch <= "\u9fff" for ch in row["prompt"]) for row in rows)
    for call in calls:
        if "NAI native lettering contract" in call["prompt"]:
            assert "no requested panel name" in call["prompt"]
