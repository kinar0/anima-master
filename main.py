import asyncio
import json
import os
import sys
import threading
from copy import deepcopy
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:
    from .toolbox_library import ToolboxLibrary
    from .vocabulary_review import VocabularyReview
    from .agent_tools.comfyui_sizes import MAX_NON_WHITELIST_EXPLICIT_PIXELS
    from .autofilter_settings import AutofilterSettings
    from .artist_session_settings import ArtistSessionSettings
    from .command_router import parse_hard_route
    from .config_defaults import (
        flatten_config,
        group_config,
        maybe_migrate_to_grouped_config,
        maybe_reset_to_defaults,
        migrate_block_rule_groups,
        migrate_prompt_defaults,
    )
    from .prompt_presets import (
        apply_config_preset,
        maybe_materialize_chiyo_preset,
        resolve_chiyo_profile,
    )
    from .service_container import build_services
    from .danbooru_resolver import WardrobeValidationError
    from .usage_limiter import DailyUsageLimiter
    from .opus_usage_monitor import OpusUsageMonitor
    from .generation_request_guard import generation_request_guard
    from .noun_translation_batch import (
        build_import_plan, parse_existing_mappings, review_rows,
        select_general_tags, translate_with_astrbot, translation_review_rows, verify_with_astrbot,
    )
    from .prompt_block_rules import blocked_prompt_message
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from toolbox_library import ToolboxLibrary
    from vocabulary_review import VocabularyReview
    from agent_tools.comfyui_sizes import MAX_NON_WHITELIST_EXPLICIT_PIXELS
    from autofilter_settings import AutofilterSettings
    from artist_session_settings import ArtistSessionSettings
    from command_router import parse_hard_route
    from config_defaults import (
        flatten_config,
        group_config,
        maybe_migrate_to_grouped_config,
        maybe_reset_to_defaults,
        migrate_block_rule_groups,
        migrate_prompt_defaults,
    )
    from prompt_presets import (
        apply_config_preset,
        maybe_materialize_chiyo_preset,
        resolve_chiyo_profile,
    )
    from service_container import build_services
    from danbooru_resolver import WardrobeValidationError
    from usage_limiter import DailyUsageLimiter
    from opus_usage_monitor import OpusUsageMonitor
    from generation_request_guard import generation_request_guard
    from noun_translation_batch import (
        build_import_plan, parse_existing_mappings, review_rows,
        select_general_tags, translate_with_astrbot, translation_review_rows, verify_with_astrbot,
    )
    from prompt_block_rules import blocked_prompt_message


class ComfyUIAgentPlugin(Star):
    """Basic local ComfyUI backend for AstrBot."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context, config)
        schema_path = Path(__file__).with_name("_conf_schema.json")
        plugin_data_path = (
            Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_anima_master"
        )
        self._plugin_data_path = plugin_data_path
        chiyo_snapshot_path = plugin_data_path / "chiyo_preset_base.json"
        if bool(flatten_config(config or {}).get("reset_to_defaults", False)):
            chiyo_snapshot_path.unlink(missing_ok=True)
        grouped_or_reset = maybe_migrate_to_grouped_config(config or {}, schema_path)
        raw_or_reset = maybe_reset_to_defaults(
            config or {},
            schema_path,
        )
        raw_or_reset = migrate_prompt_defaults(
            migrate_block_rule_groups(flatten_config(raw_or_reset or grouped_or_reset))
        )
        raw_config = maybe_materialize_chiyo_preset(
            None,
            base_config=raw_or_reset,
            snapshot_path=chiyo_snapshot_path,
        )
        if config is not None and group_config(raw_config, schema_path) != dict(
            config or {}
        ):
            config.save_config(replace_config=group_config(raw_config, schema_path))
        self.config = apply_config_preset(raw_config)
        self._config_store = config
        self._schema_path = schema_path
        self._wardrobe_lock = asyncio.Lock()
        self._noun_translation_lock = asyncio.Lock()
        self._noun_web_job: dict[str, Any] = {"state": "idle", "message": ""}
        self._noun_web_task: asyncio.Task | None = None
        self._noun_snapshot_lock = threading.RLock()
        self._noun_snapshot_cache = None
        self._vocabulary_review = VocabularyReview(
            Path(__file__).parent / "third_party/llm_nai_toolbox/vocabulary_review_catalog.json",
            plugin_data_path / "vocabulary_review_decisions.json",
        )
        self._multi_generation_semaphore = asyncio.Semaphore(
            max(1, self._int("multi_max_concurrent_generations", 1))
        )
        self._generation_queue_lock = threading.Lock()
        self._unfinished_generation_requests = 0
        self._generation_request_guard = generation_request_guard
        self._daily_usage = DailyUsageLimiter(
            plugin_data_path / "daily_generation_usage.json", logger,
            tasks_dir=plugin_data_path / "tasks",
        )
        self._opus_monitor = OpusUsageMonitor(
            self.config, plugin_data_path / "opus_usage_cache.json"
        )
        self._autofilter_settings = AutofilterSettings(
            plugin_data_path / "autofilter_sessions.json"
        )
        self._artist_session_settings = ArtistSessionSettings(
            plugin_data_path / "artist_sessions.json"
        )
        self._danbooru_tag_cache: dict[str, Any] = {}
        self._last_prompt_summary: ContextVar[dict[str, Any]] = ContextVar(
            f"anima_prompt_summary_{id(self)}",
            default={},
        )
        self._generation_usage_summary: ContextVar[dict[str, Any]] = ContextVar(
            f"anima_generation_usage_{id(self)}",
            default={},
        )
        self._services = build_services(
            context=self.context,
            config=self.config,
            config_store=config,
            logger=logger,
            danbooru_tag_cache=self._danbooru_tag_cache,
            get_bool=self._bool,
            get_int=self._int,
            get_float=self._float,
            get_str=self._str,
            shorten=self._shorten,
            is_allowed=self._is_allowed,
            build_prompt=self._build_anima_prompt,
            prompt_summary=lambda: dict(self._last_prompt_summary.get()),
            generate=self._generate,
            edit=self._edit,
            remove_bg=self._remove_bg,
            spell=self._spell,
            reverse=self._reverse,
            generation_usage_text=self._generation_usage_text,
        )
        self._runtime = self._services.runtime
        self._runtime.autofilter_settings = self._autofilter_settings
        self._prompt_pipeline = self._services.prompt_pipeline
        self._toolbox_library = ToolboxLibrary(self._prompt_pipeline._toolbox_vocabulary)
        self._prompt_pipeline.artist_session_settings = self._artist_session_settings
        self._generation_task = self._services.generation_task
        self._action_handler = self._services.action_handler
        self._action_handler.artist_session_settings = self._artist_session_settings
        self._llm_tool_bridge = self._services.llm_tool_bridge
        context.register_web_api(
            "/astrbot_plugin_anima_master/wardrobe",
            self.get_wardrobe,
            ["GET"],
            "查看 Anima 服装词库",
        )
        context.register_web_api(
            "/astrbot_plugin_anima_master/opus-usage", self.get_opus_usage,
            ["GET"], "查看 Opus 剩余额度缓存",
        )
        context.register_web_api(
            "/astrbot_plugin_anima_master/wardrobe/save",
            self.save_wardrobe,
            ["POST"],
            "保存 Anima 服装词库",
        )
        for path, handler, methods, label in (
            ("toolbox/library", self.get_toolbox_library, ["GET"], "翻阅 Toolbox 词库"),
            ("toolbox/entry", self.get_toolbox_entry, ["GET"], "查看 Toolbox 词条"),
            ("toolbox/save", self.save_toolbox_entry, ["POST"], "维护 Toolbox 词条"),
            ("toolbox/probe", self.probe_toolbox, ["POST"], "本地试查 Toolbox"),
            ("noun-translation", self.get_noun_translation, ["GET"], "查看名词翻译进度"),
            ("noun-translation/status", self.get_noun_translation_status, ["GET"], "查看名词翻译任务状态"),
            ("vocabulary-review", self.get_vocabulary_review, ["GET"], "查看词汇候选与释义"),
            ("vocabulary-review/decision", self.save_vocabulary_review, ["POST"], "保存词汇审查记录"),
            ("noun-translation/enabled", self.set_noun_translation_enabled, ["POST"], "启用或停用名词翻译参与生图"),
            ("noun-translation/run", self.run_noun_translation, ["POST"], "启动名词翻译或复核"),
            ("noun-translation/import", self.import_noun_translation, ["POST"], "批量导入名词翻译"),
        ):
            context.register_web_api(
                f"/astrbot_plugin_anima_master/{path}", handler, methods, label,
            )
        context.register_web_api(
            "/astrbot_plugin_anima_master/autofilter",
            self.get_autofilter,
            ["GET"],
            "查看各会话图片打码设置",
        )
        context.register_web_api(
            "/astrbot_plugin_anima_master/autofilter/save",
            self.save_autofilter,
            ["POST"],
            "保存会话图片打码设置",
        )

    async def _toolbox_api(self, operation):
        try:
            self._ensure_dashboard_user()
            return json_response(await operation())
        except PermissionError:
            return error_response("请先登录 AstrBot 管理页面。", status_code=403)
        except (ValueError, TypeError) as exc:
            return error_response(str(exc), status_code=409 if "刷新" in str(exc) else 400)
        except OSError:
            logger.exception("Toolbox 本地词库操作失败")
            return error_response("无法读取或保存词库，请检查本地文件。", status_code=500)

    def _toolbox_directory(self):
        return str(self.config.get("toolbox_vocabulary_path", ""))

    async def get_toolbox_library(self):
        async def operation():
            p = request.query
            value = await asyncio.to_thread(self._toolbox_library.list, self._toolbox_directory(),
                query=p.get("query", ""), category=p.get("category", ""), state=p.get("state", "all"),
                page=int(p.get("page", 1)), page_size=int(p.get("page_size", 30)))
            value["mode"] = str(self.config.get("toolbox_vocabulary_mode", "off"))
            value["max_chars"] = int(self.config.get("toolbox_vocabulary_max_chars", 1200))
            return value
        return await self._toolbox_api(operation)

    async def get_toolbox_entry(self):
        async def operation():
            return await asyncio.to_thread(self._toolbox_library.detail, self._toolbox_directory(), request.query.get("tag", ""))
        return await self._toolbox_api(operation)

    async def save_toolbox_entry(self):
        async def operation():
            payload = await request.json(default={})
            return await asyncio.to_thread(self._toolbox_library.save, self._toolbox_directory(), payload)
        return await self._toolbox_api(operation)

    async def probe_toolbox(self):
        async def operation():
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求格式无效")
            return await asyncio.to_thread(self._toolbox_library.probe, self._toolbox_directory(), payload.get("text", ""))
        return await self._toolbox_api(operation)

    async def get_opus_usage(self):
        """Read the monitor cache; opening pages never triggers a NovelAI request."""
        try:
            self._ensure_dashboard_user()
            return json_response(self._opus_monitor.snapshot())
        except PermissionError:
            return error_response("请先登录 AstrBot 管理页面。", status_code=403)

    async def get_autofilter(self):
        try:
            self._ensure_dashboard_user()
            return json_response(self._autofilter_settings.snapshot())
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)

    async def save_autofilter(self):
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            self._autofilter_settings.set_enabled(payload.get("session", ""), payload.get("enabled"))
            return json_response(self._autofilter_settings.snapshot())
        except ValueError as exc:
            return error_response(str(exc), status_code=400)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)

    @staticmethod
    def _ensure_dashboard_user() -> None:
        if not getattr(request, "username", None):
            raise PermissionError("需要登录 AstrBot WebUI 后才能编辑服装词库。")

    async def get_wardrobe(self):
        """Return all editable wardrobe collections to the Plugin Page."""
        try:
            self._ensure_dashboard_user()
            return json_response(self._services.danbooru_resolver.wardrobe_snapshot())
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except Exception:
            logger.exception("服装词库：读取失败")
            return error_response(
                "读取服装词库失败，请查看 AstrBot 日志。", status_code=500
            )

    async def save_wardrobe(self):
        """Validate, persist, and hot-apply edits from the Plugin Page."""
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            async with self._wardrobe_lock:
                result = self._services.danbooru_resolver.save_wardrobe(payload)
                resolver_config = self._services.danbooru_resolver._config
                for key in (
                    "danbooru_named_outfit_mappings",
                    "danbooru_term_mappings",
                ):
                    self.config[key] = list(resolver_config.get(key, []))
                if self._config_store is not None:
                    self._config_store.save_config(
                        replace_config=group_config(self.config, self._schema_path)
                    )
            return json_response({**result, "saved": True})
        except WardrobeValidationError as exc:
            status = 409 if "刷新后重试" in str(exc) else 400
            return error_response(str(exc), status_code=status)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except Exception:
            logger.exception("服装词库：保存失败")
            return error_response(
                "保存服装词库失败，请查看 AstrBot 日志。", status_code=500
            )

    async def initialize(self):
        self._opus_monitor.start()
        img2img_enabled = self._bool("img2img_enabled", False)
        if img2img_enabled:
            edit_tool_changed = self.context.activate_llm_tool("comfyui_edit")
        else:
            edit_tool_changed = self.context.deactivate_llm_tool("comfyui_edit")
        remove_bg_tool_changed = self.context.deactivate_llm_tool("comfyui_remove_bg")
        logger.info(
            "[comfyui_agent] chiyo_preset=%s img2img_enabled=%s edit_tool_changed=%s remove_bg_tool_changed=%s base_url=%s workflow=%s",
            resolve_chiyo_profile(self.config) or "off",
            img2img_enabled,
            edit_tool_changed,
            remove_bg_tool_changed,
            self._str("comfyui_base_url", "http://127.0.0.1:8188"),
            self._str("workflow", "anima_t2i"),
        )

    async def terminate(self):
        await self._opus_monitor.close()

    def _bool(self, key: str, default: bool) -> bool:
        return bool(self.config.get(key, default))

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _str(self, key: str, default: str = "") -> str:
        value = self.config.get(key, default)
        return str(value if value is not None else default)

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        sender_id = str(event.get_sender_id() or "").strip()
        if sender_id in self._sender_id_set("generation_blacklist_sender_ids"):
            return False
        if self._bool("admin_only", False) and not event.is_admin():
            return False
        if sender_id in self._sender_id_set("generation_whitelist_sender_ids"):
            return True
        allowed = self.config.get("allowed_sender_ids", [])
        if isinstance(allowed, str):
            allowed = [allowed]
        allowed_set = {str(item).strip() for item in allowed or [] if str(item).strip()}
        if allowed_set and sender_id not in allowed_set:
            return False
        return True

    def _sender_id_set(self, key: str) -> set[str]:
        values = self.config.get(key, [])
        if isinstance(values, str):
            values = [values]
        return {str(item).strip() for item in values or [] if str(item).strip()}

    def _enter_generation_queue(self) -> int:
        """Register a request and return the unfinished requests ahead of it."""
        with self._generation_queue_lock:
            ahead = self._unfinished_generation_requests
            self._unfinished_generation_requests += 1
            return ahead

    def _leave_generation_queue(self) -> None:
        """Remove a completed, failed, or cancelled request from the count."""
        with self._generation_queue_lock:
            if self._unfinished_generation_requests <= 0:
                logger.warning(
                    "[comfyui_agent] generation queue count was already empty"
                )
                return
            self._unfinished_generation_requests -= 1

    def _reserve_generation_call(self, event: AstrMessageEvent) -> str:
        """Reserve one daily call and return an optional rejection message."""
        sender_id = str(event.get_sender_id() or "").strip()
        if not self._is_allowed(event):
            if sender_id in self._sender_id_set("generation_blacklist_sender_ids"):
                return "你已被加入生图黑名单，当前无法使用生图功能。"
            return "ComfyUI 助手已关闭，或当前用户没有使用权限。"
        decision = self._daily_usage.acquire(
            sender_id,
            self._int("daily_generation_limit", 0),
            whitelisted=(
                sender_id in self._sender_id_set("generation_whitelist_sender_ids")
            ),
            total_limit=self._int("daily_total_generation_limit", 0),
        )
        self._generation_usage_summary.set(
            {
                "used": decision.used,
                "limit": decision.limit,
                "remaining": decision.remaining,
                "whitelisted": decision.whitelisted,
                "day": decision.day,
                "total_used": decision.total_used,
                "total_limit": decision.total_limit,
                "total_remaining": decision.total_remaining,
            }
        )
        if not decision.allowed:
            if decision.rejection == "total":
                return (
                    f"今日总生图次数已达上限（{decision.total_used}/{decision.total_limit}）。"
                    "请明天再试。"
                )
            return (
                f"你今天的生图次数已用完（{decision.used}/{decision.limit}）。"
                "请明天再试。"
            )
        return ""

    def _generation_usage_text(self, event: AstrMessageEvent) -> str:
        """Show persisted daily totals without invoking models or ComfyUI."""
        state = self._daily_usage.snapshot()
        total = state["total_used"]
        total_limit = max(0, self._int("daily_total_generation_limit", 0))
        sender = str(event.get_sender_id() or "").strip()
        used = state["counts"].get(sender or "unknown", 0)
        personal_limit = max(0, self._int("daily_generation_limit", 0))
        whitelisted = sender in self._sender_id_set("generation_whitelist_sender_ids")
        total_text = f"{total}/{total_limit}" if total_limit else f"{total}（不限额）"
        personal_text = (
            f"{used}（不限额）" if whitelisted or not personal_limit
            else f"{used}/{personal_limit}"
        )
        return (
            f"生图统计（{state['date']}）：\n"
            f"今日总生图次数：{total_text}\n"
            f"你的今日次数：{personal_text}\n"
            "按已接受的文生图请求计数，失败不返还，自动重试不重复计数。"
        )

    async def _run_tool(self, args: list[str]) -> dict[str, Any]:
        return await self._runtime.run_tool(args)

    def _shorten(self, text: str, limit: int = 1800) -> str:
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n...[已截断]"

    async def _build_anima_prompt(
        self,
        event: AstrMessageEvent,
        user_prompt: str,
        mode: str = "txt2img",
        *,
        multi_person: bool = False,
        original_user_prompt: str = "",
        canvas_size: tuple[int, int] | None = None,
        canvas_size_explicit: bool = False,
    ) -> str:
        result = await self._prompt_pipeline.build(
            event,
            user_prompt,
            mode,
            multi_person=multi_person,
            original_user_prompt=original_user_prompt,
            canvas_size=canvas_size,
            canvas_size_explicit=canvas_size_explicit,
        )
        self._last_prompt_summary.set(dict(result.summary))
        return result.final_prompt

    async def _ensure_comfyui_ready(self, event: AstrMessageEvent) -> dict[str, Any]:
        return await self._runtime.ensure_ready(event)

    async def _send_payload(
        self, event: AstrMessageEvent, payload: dict[str, Any]
    ) -> str:
        return await self._runtime.send_payload(event, payload)

    async def _generate_payload(
        self,
        event: AstrMessageEvent,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        negative_prompt: str | None = None,
        multi_person: bool = False,
        prepared_prompt: str | None = None,
        prepared_prompt_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = await self._generation_task.generate_payload(
            event,
            prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            negative_prompt=negative_prompt,
            multi_person=multi_person,
            prepared_prompt=prepared_prompt,
            prepared_prompt_summary=prepared_prompt_summary,
        )
        verified = await self._services.generation_verifier.verify_and_maybe_retry(
            event,
            payload,
            user_request=prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            negative_prompt=negative_prompt,
            multi_person=multi_person,
        )
        return verified.payload

    async def _generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        negative_prompt: str | None = None,
        multi_person: bool = False,
        size_explicit: bool = True,
    ) -> str:
        guard = getattr(self, "_generation_request_guard", None)
        if guard is not None and not guard.claim(event):
            logger.info(
                "[comfyui_agent] duplicate generation message ignored: platform=%s message_id=%s",
                event.get_platform_id(),
                getattr(getattr(event, "message_obj", None), "message_id", ""),
            )
            return ""
        message = blocked_prompt_message(prompt, self.config)
        if message:
            await event.send(event.plain_result(message))
            return message
        if (
            size_explicit
            and width is not None
            and height is not None
            and int(width) * int(height) > MAX_NON_WHITELIST_EXPLICIT_PIXELS
            and str(event.get_sender_id() or "").strip()
            not in self._sender_id_set("generation_whitelist_sender_ids")
        ):
            message = "非白名单用户指定的分辨率不能超过 1MP（1024×1024 像素）。"
            await event.send(event.plain_result(message))
            return message
        quota_error = self._reserve_generation_call(event)
        if quota_error:
            await event.send(event.plain_result(quota_error))
            return quota_error
        requests_ahead = self._enter_generation_queue()
        multi_semaphore_acquired = False
        try:
            if self._bool("notify_drawing_and_at_sender", False):
                notice = (
                    "正在绘画中"
                    if requests_ahead == 0
                    else f"正在绘画中，前面还有{requests_ahead}人"
                )
                try:
                    await event.send(event.plain_result(notice))
                except Exception as exc:
                    logger.warning(
                        "[comfyui_agent] failed to send drawing progress notice: %s",
                        str(exc)[:500],
                    )
            usage_summary = (
                dict(self._generation_usage_summary.get())
                if hasattr(self, "_generation_usage_summary")
                else {}
            )
            if multi_person:
                await self._multi_generation_semaphore.acquire()
                multi_semaphore_acquired = True
            payload = await self._generate_payload(
                event,
                prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                negative_prompt=negative_prompt,
                multi_person=multi_person,
            )
            if usage_summary:
                payload["daily_usage"] = usage_summary
            if payload.get("prompt_degraded"):
                try:
                    await event.send(
                        event.plain_result(
                            "提示词优化服务不可用，本次已使用原始提示词继续生成；"
                            "结果可能不符合 Danbooru Tag 预期。"
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "[comfyui_agent] failed to send prompt degradation notice: %s",
                        str(exc)[:500],
                    )
            if payload.get("ok") and payload.get("character_resolution_warning"):
                try:
                    await event.send(
                        event.plain_result(
                            "未能可靠确认部分现有作品角色的 Danbooru Tag；"
                            "本次仍会发送结果，但角色外观可能偏离设定。"
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "[comfyui_agent] failed to send character resolution notice: %s",
                        str(exc)[:500],
                    )
            if (
                payload.get("ok")
                and payload.get("verification_warning")
                and self._bool("multi_show_degraded_notice", False)
            ):
                try:
                    await event.send(
                        event.plain_result(
                            "多人候选均未完全通过结构检查，已发送其中最接近要求的一张。"
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "[comfyui_agent] failed to send multi-person degradation notice: %s",
                        str(exc)[:500],
                    )
            result = await self._send_payload(event, payload)
            self._generation_task.record_delivery(
                payload.get("task_id"),
                payload.get("delivery"),
            )
            return result
        finally:
            if multi_semaphore_acquired:
                self._multi_generation_semaphore.release()
            self._leave_generation_queue()

    async def _edit(self, event: AstrMessageEvent, prompt: str) -> str:
        return await self._action_handler.edit(event, prompt)

    async def _spell(self, event: AstrMessageEvent) -> str:
        return await self._action_handler.spell(event)

    async def _reverse(self, event: AstrMessageEvent) -> str:
        return await self._action_handler.reverse(event)

    async def _upscale(self, event: AstrMessageEvent) -> str:
        return await self._action_handler.upscale(event)

    async def _remove_bg(self, event: AstrMessageEvent) -> str:
        return await self._action_handler.remove_bg(event)

    def _status_text(self, payload: dict[str, Any]) -> str:
        return self._action_handler.status_text(payload)

    async def _handle_action(
        self, event: AstrMessageEvent, action: str, prompt: str
    ) -> str | None:
        return await self._action_handler.handle_action(event, action, prompt)

    @filter.command_group("anm", alias={"comfyui", "anima"})
    def comfyui_group(self):
        pass

    @filter.event_message_type(EventMessageType.ALL, priority=sys.maxsize - 2)
    async def hard_route_comfyui(self, event: AstrMessageEvent):
        route = parse_hard_route(event.get_message_str())
        if not route:
            route = parse_hard_route(event.get_message_outline())
        if not route:
            return
        action, prompt = route
        if action in {"term_preview", "term_translate", "term_verify", "term_import"}:
            event.stop_event()
            if action == "term_preview":
                message = self._term_preview_text(event)
            elif action == "term_translate":
                message = await self._term_translate_text(event, prompt)
            elif action == "term_verify":
                message = await self._term_verify_text(event, prompt)
            else:
                message = await self._term_import_text(event, prompt)
            await event.send(event.plain_result(message))
            return
        if not self._is_allowed(event):
            await event.send(
                event.plain_result("ComfyUI 助手已关闭，或当前用户没有使用权限。")
            )
            event.stop_event()
            return

        logger.info(
            "[comfyui_agent] hard route action=%s sender=%s",
            action,
            event.get_sender_id(),
        )
        event.stop_event()

        message = await self._handle_action(event, action, prompt)
        if message:
            await event.send(event.plain_result(message))

    @comfyui_group.command("status", alias={"状态"})
    async def cmd_status(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "status", ""))

    def _noun_catalog_path(self) -> Path:
        skills_root = str(os.environ.get("COMFYUI_GOOD_ANIMA_SKILLS_DIR") or "").strip()
        root = Path(skills_root) if skills_root else Path.home() / ".codex" / "skills"
        return root / "danbooru-tags" / "anima-1.0.csv"

    def _existing_term_mappings(self) -> dict[str, str]:
        return parse_existing_mappings(self.config.get("danbooru_term_mappings", []))

    def _noun_translation_snapshot(self) -> dict[str, Any]:
        translated_path = self._plugin_data_path / "noun_translation_review.json"
        verified_path = self._plugin_data_path / "noun_translation_verified.json"
        catalog_path = self._noun_catalog_path()
        existing = self._existing_term_mappings()
        def signature(path):
            stat = path.stat() if path.exists() else None
            return (str(path), stat.st_mtime_ns, stat.st_size) if stat else (str(path), None)
        key = (signature(catalog_path), signature(translated_path), signature(verified_path), tuple(existing.items()))
        lock = getattr(self, "_noun_snapshot_lock", None)
        if lock is None:
            self._noun_snapshot_lock = lock = threading.RLock()
        with lock:
            cached = getattr(self, "_noun_snapshot_cache", None)
            if cached and cached[0] == key:
                return {**deepcopy(cached[1]), "job": dict(self._noun_web_job),
                        "enabled": bool(self.config.get("danbooru_term_mappings_enabled", True))}
            # Keep computation in the same lock; an overlapping refresh need not
            # duplicate expensive local work. The HTTP handler runs it off-loop.
            value = ComfyUIAgentPlugin._compute_noun_translation_snapshot(
                self, catalog_path, translated_path, verified_path, existing)
            self._noun_snapshot_cache = (key, value)
            return deepcopy(value)

    def _compute_noun_translation_snapshot(self, catalog_path, translated_path, verified_path, existing):
        candidates = select_general_tags(catalog_path)
        translated = json.loads(translated_path.read_text(encoding="utf-8")) if translated_path.exists() else []
        verified = json.loads(verified_path.read_text(encoding="utf-8")) if verified_path.exists() else []
        if not isinstance(translated, list) or not isinstance(verified, list):
            raise ValueError("翻译断点文件格式无效")
        reviewed = review_rows(translated, existing_mappings=existing)
        plan = build_import_plan(translated, verified, existing)
        rows = translation_review_rows(reviewed, verified, existing, plan)
        return {
            "total": len(candidates), "translated": len(translated), "verified": len(verified),
            "flagged": sum(bool(row["review_flags"]) for row in reviewed),
            "importable": len(plan["additions"]), "skipped": len(plan["skipped"]),
            "manual_review": sum(row["needs_manual_review"] for row in rows),
            "existing": sum(item["state"] == "existing" for row in rows
                            for item in row["alias_statuses"]),
            "enabled": bool(self.config.get("danbooru_term_mappings_enabled", True)),
            "additions": plan["additions"], "skipped_aliases": plan["skipped"],
            "rows": rows, "job": dict(self._noun_web_job),
        }

    async def get_noun_translation(self):
        try:
            self._ensure_dashboard_user()
            snapshot = await asyncio.to_thread(self._noun_translation_snapshot)
            return json_response(snapshot)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return error_response(str(exc), status_code=400)

    async def get_noun_translation_status(self):
        try:
            self._ensure_dashboard_user()
            return json_response({"job": dict(self._noun_web_job),
                                  "enabled": bool(self.config.get("danbooru_term_mappings_enabled", True))})
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)

    async def get_vocabulary_review(self):
        try:
            self._ensure_dashboard_user()
            params = request.query
            existing = self._existing_term_mappings()
            value = await asyncio.to_thread(self._vocabulary_review.snapshot,
                existing=existing, query=params.get("query", ""), state=params.get("state", "focus"),
                page=int(params.get("page", 1)), page_size=int(params.get("page_size", 30)))
            return json_response(value)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except (OSError, ValueError, TypeError) as exc:
            return error_response(str(exc), status_code=400)

    async def save_vocabulary_review(self):
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            value = await asyncio.to_thread(self._vocabulary_review.decide, payload)
            return json_response(value)
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except (ValueError, TypeError) as exc:
            return error_response(str(exc), status_code=409 if "刷新" in str(exc) else 400)
        except OSError:
            logger.exception("词汇审查记录保存失败")
            return error_response("审查记录保存失败，未应用词库", status_code=500)

    async def set_noun_translation_enabled(self):
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            if not isinstance(payload, dict) or type(payload.get("enabled")) is not bool:
                raise ValueError("enabled 必须是布尔值")
            if self._config_store is None:
                raise RuntimeError("配置存储不可用，开关未修改。")
            enabled = payload["enabled"]
            async with self._wardrobe_lock:
                updated = {**self.config, "danbooru_term_mappings_enabled": enabled}
                self._config_store.save_config(
                    replace_config=group_config(updated, self._schema_path)
                )
                self.config["danbooru_term_mappings_enabled"] = enabled
                self._services.danbooru_resolver._config[
                    "danbooru_term_mappings_enabled"
                ] = enabled
            return json_response({"enabled": enabled})
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except ValueError as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("名词翻译开关保存失败")
            return error_response("保存开关失败，请查看 AstrBot 日志。", status_code=500)

    async def _run_noun_web_job(self, action: str, provider_id: str) -> None:
        try:
            async with self._noun_translation_lock:
                if action == "translate":
                    candidates = select_general_tags(self._noun_catalog_path())
                    checkpoint = self._plugin_data_path / "noun_translation_review.json"
                    result = await translate_with_astrbot(self.context, provider_id, candidates, checkpoint)
                    rows = review_rows(json.loads(checkpoint.read_text(encoding="utf-8")),
                                       existing_mappings=self._existing_term_mappings())
                    (self._plugin_data_path / "noun_translation_flagged.json").write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    verification = await verify_with_astrbot(
                        self.context, provider_id, checkpoint,
                        self._plugin_data_path / "noun_translation_verified.json",
                        self._existing_term_mappings())
                    message = (f"本轮新译 {result['newly_translated']} 条、新复核 "
                               f"{verification['newly_verified']} 条；累计翻译 "
                               f"{result['completed']}/{len(candidates)}、复核 "
                               f"{verification['verified']} 条。本轮模型调用 "
                               f"{result['calls'] + verification['calls']} 次。")
                else:
                    result = await verify_with_astrbot(
                        self.context, provider_id,
                        self._plugin_data_path / "noun_translation_review.json",
                        self._plugin_data_path / "noun_translation_verified.json",
                        self._existing_term_mappings())
                    message = (f"本轮新复核 {result['newly_verified']} 条；"
                               f"累计复核 {result['verified']} 条，剩余 {result['remaining']} 条；"
                               f"本轮模型调用 {result['calls']} 次。")
            self._noun_web_job = {"state": "done", "message": message}
        except Exception as exc:
            logger.exception("名词翻译网页任务失败")
            self._noun_web_job = {"state": "error", "message": str(exc)}

    async def run_noun_translation(self):
        try:
            self._ensure_dashboard_user()
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是对象")
            action = payload.get("action")
            provider_id = str(payload.get("provider_id") or "").strip()
            if action not in {"translate", "verify"} or not provider_id or any(
                char.isspace() for char in provider_id
            ):
                raise ValueError("请选择翻译或复核，并填写有效的 provider ID。")
            if self._noun_translation_lock.locked() or self._noun_web_job["state"] == "running":
                return error_response("已有名词翻译任务正在运行。", status_code=409)
            self._noun_web_job = {"state": "running", "message": "翻译中…" if action == "translate" else "复核中…"}
            self._noun_web_task = asyncio.create_task(self._run_noun_web_job(action, provider_id))
            return json_response({"job": dict(self._noun_web_job)})
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except ValueError as exc:
            return error_response(str(exc), status_code=400)

    async def import_noun_translation(self):
        try:
            self._ensure_dashboard_user()
            if self._noun_translation_lock.locked() or self._noun_web_job["state"] == "running":
                return error_response("翻译任务运行中，请完成后再导入。", status_code=409)
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是对象")
            if payload.get("mode") != "apply":
                raise ValueError("导入模式无效")
            async with self._wardrobe_lock:
                translated = json.loads((self._plugin_data_path / "noun_translation_review.json").read_text(encoding="utf-8"))
                verified = json.loads((self._plugin_data_path / "noun_translation_verified.json").read_text(encoding="utf-8"))
                plan = build_import_plan(translated, verified, self._existing_term_mappings())
                additions = plan["additions"]
                if additions:
                    if self._config_store is None:
                        raise RuntimeError("配置存储不可用，未执行导入。")
                    updated = list(self.config.get("danbooru_term_mappings", [])) + [
                        f"{alias}={tag}" for alias, tag in additions.items()]
                    self._config_store.save_config(replace_config=group_config(
                        {**self.config, "danbooru_term_mappings": updated}, self._schema_path))
                    self.config["danbooru_term_mappings"] = updated
                    self._services.danbooru_resolver._config["danbooru_term_mappings"] = updated
            return json_response({"imported": len(additions), "skipped": len(plan["skipped"]),
                                  "existing": len(plan["existing"])})
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("名词翻译网页导入失败")
            return error_response("导入失败，请查看 AstrBot 日志。", status_code=500)

    def _term_preview_text(self, event: AstrMessageEvent) -> str:
        if not event.is_admin():
            return "仅管理员可查看批量名词翻译任务。"
        try:
            candidates = select_general_tags(self._noun_catalog_path())
        except (OSError, ValueError) as exc:
            return str(exc)
        sample = "、".join(row["tag"] for row in candidates[:12])
        progress = ""
        checkpoint = self._plugin_data_path / "noun_translation_review.json"
        if checkpoint.exists():
            try:
                reviewed = review_rows(
                    json.loads(checkpoint.read_text(encoding="utf-8")),
                    existing_mappings=self._existing_term_mappings(),
                )
                progress = (
                    f"已试译 {len(reviewed)}/{len(candidates)}；"
                    f"待审标记 {sum(bool(row['review_flags']) for row in reviewed)} 条，"
                    f"空别名 {sum(not row['aliases'] for row in reviewed)} 条。"
                )
            except (OSError, ValueError, KeyError, TypeError):
                progress = "已有断点文件，但格式需要检查。"
        return (
            f"全部候选 {len(candidates)} 个通用 tag（帖子数至少 1000，按热度排序）。"
            f"{progress}"
            f"前 12 个：{sample}。运行 /anm term_translate <DeepSeek provider ID> "
            "每次最多翻译 50 个；只写待审核断点，不导入现有词库。"
        )

    @comfyui_group.command("term_preview", alias={"名词翻译预览"})
    async def cmd_term_preview(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(self._term_preview_text(event))

    async def _term_translate_text(self, event: AstrMessageEvent, provider_id: str) -> str:
        if not event.is_admin():
            return "仅管理员可运行批量名词翻译。"
        if not provider_id or any(char.isspace() for char in provider_id):
            return "用法：/anm term_translate <DeepSeek provider ID>"
        if self._noun_translation_lock.locked():
            return "名词翻译已有任务正在运行，请稍后重试。"
        try:
            async with self._noun_translation_lock:
                candidates = select_general_tags(self._noun_catalog_path())
                checkpoint = self._plugin_data_path / "noun_translation_review.json"
                result = await translate_with_astrbot(
                    self.context, provider_id, candidates, checkpoint,
                )
                rows = review_rows(
                    json.loads(checkpoint.read_text(encoding="utf-8")),
                    existing_mappings=self._existing_term_mappings(),
                )
                review_path = self._plugin_data_path / "noun_translation_flagged.json"
                review_path.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                verification = await verify_with_astrbot(
                    self.context, provider_id, checkpoint,
                    self._plugin_data_path / "noun_translation_verified.json",
                    self._existing_term_mappings(),
                )
        except (OSError, ValueError, RuntimeError) as exc:
            return f"名词翻译暂停：{exc}"
        return (
            f"本次调用 {result['calls']} 次、新译 {result['newly_translated']} 条；"
            f"累计已译 {result['completed']}/{len(candidates)}，"
            f"待译 {result['remaining']}。本次 token：输入 {result['input_tokens']}、"
            f"缓存输入 {result['cached_tokens']}、输出 {result['output_tokens']}。"
            f"本轮新复核 {verification['newly_verified']} 条，"
            f"累计已复核 {verification['verified']}/{result['completed']}，"
            f"复核调用 {verification['calls']} 次、输入 {verification['input_tokens']}、"
            f"缓存输入 {verification['cached_tokens']}、输出 {verification['output_tokens']} token。"
            f"待审核文件：{review_path}。"
            "结果不会自动进入生图词库。"
        )

    @comfyui_group.command("term_translate", alias={"批量名词翻译"})
    async def cmd_term_translate(self, event: AstrMessageEvent, provider_id: str):
        event.stop_event()
        yield event.plain_result(await self._term_translate_text(event, str(provider_id)))

    async def _term_verify_text(self, event: AstrMessageEvent, provider_id: str) -> str:
        if not event.is_admin():
            return "仅管理员可审核批量译文。"
        if not provider_id or any(char.isspace() for char in provider_id):
            return "用法：/anm term_verify <DeepSeek provider ID>"
        if self._noun_translation_lock.locked():
            return "名词翻译已有任务正在运行，请稍后重试。"
        try:
            async with self._noun_translation_lock:
                result = await verify_with_astrbot(
                    self.context, provider_id,
                    self._plugin_data_path / "noun_translation_review.json",
                    self._plugin_data_path / "noun_translation_verified.json",
                    self._existing_term_mappings(),
                )
        except (OSError, ValueError, RuntimeError) as exc:
            return f"名词翻译审核暂停：{exc}"
        return (f"本轮新复核 {result['newly_verified']} 条，"
                f"累计已复核 {result['verified']} 条，剩余 {result['remaining']} 条；"
                f"本次模型调用 {result['calls']} 次，输入 {result['input_tokens']}、"
                f"缓存输入 {result['cached_tokens']}、输出 {result['output_tokens']} token。"
                "运行 /anm term_import preview 查看可导入数。")

    @comfyui_group.command("term_verify", alias={"名词翻译审核"})
    async def cmd_term_verify(self, event: AstrMessageEvent, provider_id: str):
        event.stop_event()
        yield event.plain_result(await self._term_verify_text(event, str(provider_id)))

    async def _term_import_text(self, event: AstrMessageEvent, mode: str) -> str:
        if not event.is_admin():
            return "仅管理员可导入批量译文。"
        if mode not in {"preview", "apply"}:
            return "用法：/anm term_import preview 或 /anm term_import apply"
        try:
            async with self._wardrobe_lock:
                translated = json.loads((self._plugin_data_path / "noun_translation_review.json").read_text(encoding="utf-8"))
                verified = json.loads((self._plugin_data_path / "noun_translation_verified.json").read_text(encoding="utf-8"))
                plan = build_import_plan(translated, verified, self._existing_term_mappings())
                additions = plan["additions"]
                if mode == "apply" and additions:
                    if self._config_store is None:
                        return "配置存储不可用，未执行导入。"
                    current = self.config.get("danbooru_term_mappings", [])
                    updated = list(current) + [f"{alias}={tag}" for alias, tag in additions.items()]
                    new_config = {**self.config, "danbooru_term_mappings": updated}
                    self._config_store.save_config(
                        replace_config=group_config(new_config, self._schema_path)
                    )
                    self.config["danbooru_term_mappings"] = updated
                    self._services.danbooru_resolver._config["danbooru_term_mappings"] = updated
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return f"名词翻译导入暂停：{exc}"
        verb = "可导入" if mode == "preview" else "已导入"
        return (f"{verb} {len(additions)} 个中文别名；"
                f"已有 {len(plan['existing'])} 个相同映射，无需重复导入；"
                f"跳过 {len(plan['skipped'])} 个冲突或不安全别名。"
                "未通过自动复核的别名不会导入。")

    @comfyui_group.command("term_import", alias={"名词翻译导入"})
    async def cmd_term_import(self, event: AstrMessageEvent, mode: str):
        event.stop_event()
        yield event.plain_result(await self._term_import_text(event, str(mode)))

    @comfyui_group.command("diagnose", alias={"诊断", "部署诊断", "diagnosis"})
    async def cmd_diagnose(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "diagnose", ""))

    @comfyui_group.command("debug", alias={"调试状态", "调试", "debug_status"})
    async def cmd_debug_status(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "debug_status", ""))

    @comfyui_group.command("stats", alias={"生图统计", "今日生图", "生图次数"})
    async def cmd_generation_stats(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "generation_stats", ""))

    @comfyui_group.command("generate", alias={"生图", "画图"})
    async def cmd_generate(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        message = await self._handle_action(
            event, "generate", str(prompt or "").strip()
        )
        if message:
            yield event.plain_result(message)

    @comfyui_group.command("multi_person", alias={"多人"})
    async def cmd_multi_person(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        message = await self._handle_action(
            event, "multi_person", str(prompt or "").strip()
        )
        if message:
            yield event.plain_result(message)

    @comfyui_group.command("edit", alias={"改图", "图生图", "风格化", "重绘"})
    async def cmd_edit(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(event, "edit", str(prompt or "").strip())
        )

    @comfyui_group.command("upscale", alias={"放大", "高清", "高清修复"})
    async def cmd_upscale(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(event, "disabled_upscale", "")
        )

    @comfyui_group.command("remove_bg", alias={"抠图", "去背景", "去除背景"})
    async def cmd_remove_bg(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(event, "disabled_remove_bg", "")
        )

    @comfyui_group.command(
        "spell", alias={"解析法术", "法术解析", "读取法术", "提取提示词", "读取提示词"}
    )
    async def cmd_spell(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "spell", ""))

    @comfyui_group.command("reverse", alias={"反推", "图片反推", "反推提示词"})
    async def cmd_reverse(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "reverse", ""))

    @comfyui_group.command(
        "artist",
        alias={
            "创建画师组",
        },
    )
    async def cmd_set_artist_tags(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(
                event, "create_artist_preset", str(prompt or "").strip()
            )
        )

    @comfyui_group.command("append_artist", alias={"追加画师组"})
    async def cmd_append_artist_tags(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(
                event, "append_artist_tags", str(prompt or "").strip()
            )
        )

    @comfyui_group.command("use_artist", alias={"切换画师组"})
    async def cmd_use_artist_preset(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(
                event, "use_artist_preset", str(prompt or "").strip()
            )
        )

    @comfyui_group.command("list_artist", alias={"查看画师组"})
    async def cmd_list_artist_presets(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(event, "list_artist_presets", "")
        )

    @comfyui_group.command("delete_artist", alias={"删除画师组"})
    async def cmd_delete_artist_preset(
        self, event: AstrMessageEvent, prompt: GreedyStr
    ):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(
                event, "delete_artist_preset", str(prompt or "").strip()
            )
        )

    @comfyui_group.command(
        "character",
        alias={
            "添加角色",
        },
    )
    async def cmd_add_fixed_character(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        yield event.plain_result(
            await self._handle_action(
                event, "add_fixed_character", str(prompt or "").strip()
            )
        )

    @filter.llm_tool(name="comfyui_status")
    async def comfyui_status(self, event: AstrMessageEvent) -> str:
        """Check whether local ComfyUI is online and ready.

        Args:
        """
        return await self._llm_tool_bridge.status(event)

    @filter.llm_tool(name="comfyui_generate")
    async def comfyui_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        negative_prompt: str | None = None,
    ) -> str:
        """Generate an image with local ComfyUI from the provided prompt/tags.

        Args:
            prompt(string): Complete prompt or tags to send to ComfyUI unchanged.
            width(number): Optional width from 512 to 1756, divisible by 4.
            height(number): Optional height from 512 to 1756, divisible by 4.
            steps(number): Optional sampling steps.
            cfg(number): Optional CFG scale.
            negative_prompt(string): Optional negative prompt to use for this generation.
        """
        return await self._llm_tool_bridge.generate(
            event,
            prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            negative_prompt=negative_prompt,
        )

    @filter.llm_tool(name="comfyui_edit")
    async def comfyui_edit(self, event: AstrMessageEvent, prompt: str) -> str:
        """Edit the most recent chat image with local ComfyUI img2img.

        Args:
            prompt(string): Complete img2img prompt or tags.
        """
        return await self._llm_tool_bridge.edit(event, prompt)

    @filter.llm_tool(name="comfyui_remove_bg")
    async def comfyui_remove_bg(self, event: AstrMessageEvent) -> str:
        """Remove the background from the most recent chat image with local ComfyUI.

        Args:
        """
        return await self._llm_tool_bridge.remove_bg(event)

    @filter.llm_tool(name="comfyui_extract_prompt")
    async def comfyui_extract_prompt(self, event: AstrMessageEvent) -> str:
        """Extract embedded generation prompt/metadata from the most recent or quoted image.

        Args:
        """
        return await self._llm_tool_bridge.extract_prompt(event)

    @filter.llm_tool(name="comfyui_reverse_prompt")
    async def comfyui_reverse_prompt(self, event: AstrMessageEvent) -> str:
        """Reverse-engineer danbooru tags from the most recent or quoted image using a vision model.

        Args:
        """
        return await self._llm_tool_bridge.reverse_prompt(event)
