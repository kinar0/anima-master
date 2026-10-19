"""Opt-in observation of existing LLM responses, without changing requests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from time import perf_counter
from typing import Any
from uuid import uuid4


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_count(*values: Any) -> int | None:
    return next((count for value in values if (count := _count(value)) is not None), None)


def _response_usage(response: Any) -> dict[str, Any]:
    """Keep unavailable counts null; reasoning is a subset of output, not extra."""
    usage = _field(response, "usage")
    raw = _field(response, "raw_completion")
    raw_usage = _field(raw, "usage")
    if raw is not None and raw_usage is None and (
        "usage" in raw if isinstance(raw, Mapping) else hasattr(raw, "usage")
    ):
        # AstrBot may synthesize TokenUsage(0, 0, 0) when an OpenAI response
        # explicitly omits usage. Those adapter defaults are not provider counts.
        usage = None
    if raw_usage is None:
        raw_usage = usage
    input_tokens = _first_count(_field(raw_usage, "prompt_tokens"), _field(usage, "input"))
    cached = _first_count(
        _field(_field(raw_usage, "prompt_tokens_details"), "cached_tokens"),
        _field(raw_usage, "prompt_cache_hit_tokens"),
        _field(usage, "input_cached"),
    )
    uncached = _first_count(_field(raw_usage, "prompt_cache_miss_tokens"), _field(usage, "input_other"))
    if input_tokens is None and cached is not None and uncached is not None:
        input_tokens = cached + uncached
    output_tokens = _first_count(_field(raw_usage, "completion_tokens"), _field(usage, "output"))
    total = _first_count(_field(raw_usage, "total_tokens"))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    choices = _field(raw, "choices")
    finish = _field(choices[0], "finish_reason") if isinstance(choices, (list, tuple)) and choices else None
    return {
        "usage_available": usage is not None or raw_usage is not None,
        "input_tokens": input_tokens,
        "input_cached_tokens": cached,
        "input_uncached_tokens": uncached,
        "output_tokens": output_tokens,
        "reasoning_tokens": _first_count(
            _field(_field(raw_usage, "completion_tokens_details"), "reasoning_tokens"),
        ),
        "total_tokens": total,
        "finish_reason": finish if isinstance(finish, str) else None,
    }


class LLMDiagnostics:
    """Log only selected response fields; never dump requests or raw responses."""

    def __init__(self, logger: Any, get_bool: Callable[[str, bool], bool]):
        self._logger = logger
        self._bool = get_bool
        self._run_id: ContextVar[str] = ContextVar(f"anima_llm_run_{id(self)}", default="")

    def start_run(self) -> str:
        """Group one prompt build's calls independently of concurrent builds."""
        run_id = uuid4().hex if (
            self._bool("debug_llm_reasoning_enabled", False)
            or self._bool("debug_llm_usage_enabled", False)
        ) else ""
        self._run_id.set(run_id)
        return run_id

    async def generate(self, context: Any, *, stage: str, attempt: str = "initial", **kwargs: Any) -> Any:
        reasoning_enabled = self._bool("debug_llm_reasoning_enabled", False)
        usage_enabled = self._bool("debug_llm_usage_enabled", False)
        if not reasoning_enabled and not usage_enabled:
            return await context.llm_generate(**kwargs)

        call_id = uuid4().hex
        metadata = {
            "run_id": self._run_id.get() or call_id,
            "call_id": call_id,
            "stage": stage,
            "attempt": attempt,
            "provider_id": kwargs.get("chat_provider_id"),
            "requested_max_tokens": kwargs.get("max_tokens"),
            "requested_thinking": _field(kwargs.get("thinking"), "type"),
            "requested_reasoning_effort": kwargs.get("reasoning_effort"),
        }
        started = perf_counter()
        try:
            response = await context.llm_generate(**kwargs)
        except BaseException as exc:
            # Preserve errors and cancellation exactly; avoid logging exception
            # messages, which can contain provider request bodies or credentials.
            if usage_enabled:
                self._emit("usage", {**metadata, "status": "error", "error_type": type(exc).__name__,
                                     "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                                     "usage_available": False})
            raise

        try:
            metadata.update(status="ok", elapsed_ms=round((perf_counter() - started) * 1000, 2))
            if usage_enabled:
                self._emit("usage", {**metadata, **_response_usage(response)})
            if reasoning_enabled:
                reasoning = _field(response, "reasoning_content")
                self._emit("reasoning", {
                    **metadata,
                    "reasoning_status": "returned" if isinstance(reasoning, str) and reasoning.strip() else "not_returned",
                }, reasoning if isinstance(reasoning, str) and reasoning.strip() else "未返回推理内容")
        except Exception:
            # Diagnostics must never turn a usable generation into a failure.
            pass
        return response

    def _emit(self, kind: str, metadata: dict[str, Any], text: str | None = None) -> None:
        try:
            header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            if text is None:
                self._logger.info("[comfyui_agent] LLM %s %s", kind, header)
            else:
                self._logger.info("[comfyui_agent] LLM %s %s\n%s", kind, header, text)
        except Exception:
            pass
