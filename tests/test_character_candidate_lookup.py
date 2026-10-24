"""Candidate failure regressions, including real captured provider replies."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from danbooru_semantic import SemanticAnchor, SemanticLookupResult, lookup_semantic_anchors, resolve_local_cli_path
from test_llm_diagnostics import Context, Logger, pipeline


def candidate_call(payload, *, target="乙", prompt="甲和乙站在一起", debug=False):
    context = Context(NS(completion_text=json.dumps(payload, ensure_ascii=False)))
    builder = pipeline({"debug_prompt_enabled": debug}, context, Logger())
    diagnostics = []
    result = asyncio.run(builder._generate_character_candidates_with_llm(
        provider_id="provider", user_prompt=prompt, target_name=target,
        rejected_content="", diagnostics=diagnostics))
    return result, diagnostics[0], context.calls[0]


@pytest.mark.parametrize("tag", ["bocchi_(bocchi_the_rock!)", "s:p_little_knight",
                                  "kasane_teto_(sv)", "test-name_(work)",
                                  "oblivionis_(bang_dream!)", "mortis_(bang_dream!)"])
def test_valid_canonical_punctuation_survives_without_stripping_versions(tag):
    result, evidence, request = candidate_call({"source_name": "乙", "tag_candidates": [tag]})
    assert result == (tag,)
    assert evidence["status"] == "candidates_ready"
    assert "copyright" not in evidence  # No guessed copyright is promoted to identity.
    assert "knowing the work does NOT mean" in request["prompt"]
    assert "do not substitute their base character" in request["prompt"]


def test_real_single_candidate_reply_is_not_silently_discarded():
    result, evidence, _ = candidate_call({"source_name": "后藤独", "copyright": "Bocchi the Rock!",
        "tag_candidates": ["gotoh_hitori_(bocchi_the_rock!)"]}, target="后藤独",
        prompt="千早爱音和后藤独并排站在一起", debug=True)
    assert result == ("gotoh_hitori_(bocchi_the_rock!)",)
    assert evidence["raw"]
    assert "gotoh_hitori" not in result  # Never invent a base by stripping a suffix.


def test_another_visible_person_cannot_supply_this_targets_candidates():
    result, evidence, _ = candidate_call({"source_name": "甲", "tag_candidates": ["character_a"]})
    assert result == ()
    assert evidence["status"] == "wrong_target"


@pytest.mark.parametrize(("payload", "reason"), [
    ({"source_name": "不存在", "tag_candidates": ["character_a"]}, "ungrounded_source_name"),
    ({"source_name": "乙", "tag_candidates": "character_b"}, "invalid_candidates"),
    ({"source_name": "乙", "tag_candidates": ["中文", "bad/tag"]}, "empty_candidates"),
    ([], "invalid_object"),
])
def test_rejected_output_has_a_request_scoped_reason(payload, reason):
    result, evidence, _ = candidate_call(payload)
    assert result == ()
    assert evidence["status"] == reason
    assert "raw" not in evidence


def test_invalid_json_and_provider_errors_remain_distinguishable():
    async def run(response):
        builder = pipeline({}, Context(response), Logger())
        evidence = []
        if isinstance(response, BaseException):
            with pytest.raises(type(response)):
                await builder._generate_character_candidates_with_llm(provider_id="provider",
                    user_prompt="乙", target_name="乙", rejected_content="", diagnostics=evidence)
        else:
            assert await builder._generate_character_candidates_with_llm(provider_id="provider",
                user_prompt="乙", target_name="乙", rejected_content="", diagnostics=evidence) == ()
        return evidence[0]
    assert asyncio.run(run(NS(completion_text="not JSON")))["status"] == "invalid_json"
    failure = asyncio.run(run(TimeoutError("offline")))
    assert failure["status"] == "provider_error"
    assert failure["error_type"] == "TimeoutError"


def test_known_person_does_not_suppress_unknown_person_refinement():
    context = Context(NS(completion_text='{"source_name":"乙","tag_candidates":["character_b"]}'))
    builder = pipeline({}, context, Logger())
    anchors = (SemanticAnchor("a", "target_character", "character", "甲", "甲", ("character_a",)),
               SemanticAnchor("b", "target_character", "character", "乙", "乙", ()))
    evidence = []
    refined = asyncio.run(builder._refine_unresolved_semantic_characters(provider_id="provider",
        user_prompt="甲和乙站在一起", anchors=anchors,
        result=SemanticLookupResult(confirmed_tags=("character_a",), anchor_tags=(("a", "character_a"),)),
        diagnostics=evidence))
    assert len(context.calls) == 1
    assert refined[0] == anchors[0]
    assert refined[1].candidates == ("character_b",)
    assert evidence[0]["target_name"] == "乙"


def test_captured_bad_scope_still_cannot_become_a_confirmed_identity():
    cli = resolve_local_cli_path()
    if cli is None:
        pytest.skip("local tag CLI unavailable")
    # Production parser preserves legal punctuation, but exact evidence is still
    # required. A fabricated scope cannot gain the base identity's authority.
    anchors = tuple(SemanticAnchor(key, "target_character", "character", "后藤独", "后藤独", (tag,))
                    for key, tag in (("bad", "gotoh_hitori_(bocchi_the_rock!)"),
                                     ("alias", "bocchi_(bocchi_the_rock!)"),
                                     ("base", "gotoh_hitori")))
    result = lookup_semantic_anchors(anchors, cli_path=cli, timeout=6)
    assert "bad" not in dict(result.anchor_tags)
    assert dict(result.anchor_tags)["alias"] == "gotoh_hitori"
    assert dict(result.anchor_tags)["base"] == "gotoh_hitori"


@pytest.mark.parametrize("tag", ["oblivionis_(bang_dream!)", "mortis_(bang_dream!)"])
def test_real_copyright_suffix_is_preserved_for_character_and_clothing_source(tag):
    cli = resolve_local_cli_path()
    if cli is None:
        pytest.skip("local tag CLI unavailable")
    anchors = tuple(SemanticAnchor(role, role, "character", "来源", "来源", (tag,))
                    for role in ("target_character", "outfit_source"))
    result = lookup_semantic_anchors(anchors, cli_path=cli, timeout=6)
    assert dict(result.anchor_tags) == {"target_character": tag, "outfit_source": tag}
    assert result.outfit_source_tags == (tag,)
