# Codex 本地修改记录（Anima 插件）

> 数据来源：仓库 git 提交历史
> 分支：`codex/local-anima-custom`
> 提交作者：`Local AstrBot Customization <local-astrbot@localhost>`
> 提交总数：14 次（含本次提交）

## 提交概览

| 时间 | 提交 | 说明 |
|------|------|------|
| 2026-08-14 03:27 | `21f9bf2` | chore: preserve local Anima customization（初始快照） |
| 2026-08-14 14:12 | `e0b143a` | chore: snapshot structured prompt protocol changes |
| 2026-08-14 14:15 | `0e82f0b` | fix: strengthen directed character interactions |
| 2026-08-14 17:59 | `2b8b43f` | chore: snapshot current working version |
| 2026-08-14 22:38 | `1f7adb7` | feat: enhance prompt pipeline and add usage limits |
| 2026-08-15 02:42 | `cd8c329` | checkpoint: save prompt pipeline before Danbooru resolver refactor |
| 2026-08-15 16:46 | `23d7b0d` | feat: add validated Danbooru prompt pipeline |
| 2026-08-16 01:51 | `e170475` | feat: refine prompt pipeline and generation queue |
| 2026-08-18 18:20 | `5921c14` | checkpoint: save current prompt and workflow changes |
| 2026-08-19 21:24 | `0f6829a` | feat: 更新了大量内容，包括插件自己的标签学习管理页面和人数检测、网络连接等等 |
| 2026-08-19 21:42 | `f3b06db` | docs: 记录 LLM 空内容修复与其他 zoo 会话改动，清理误创建文件 |
| 2026-08-21 03:35 | `b886ca2` | fix: 防止 ComfyUI API 短暂超时被误报为未启动 |
| 2026-08-21 | （本次提交） | fix: 统一服装套组 canonical tag 与中英文别名并修复 UI 状态 |
| 2026-08-21 | （本次提交） | fix: 使用轻量 ComfyUI 健康检查，避免重复读取节点清单 |

## 各提交详细改动

### 1. `21f9bf2` — 初始快照

- 统计：92 个文件，+20560 行
- 项目整体首次入库，包含全部核心模块：
  - ComfyUI 运行时与 agent 工具
  - prompt 管线（`prompt_pipeline.py`、`prompt_builder.py`、`prompt_presets.py` 等）
  - Danbooru 标签解析
  - 生成任务 / 校验
  - 命令路由
  - 测试套件
  - `variants/` 变体

### 2. `e0b143a` — 结构化 prompt 协议快照

- 重构 `prompt_pipeline.py`（+196/-）
- 重构 `prompt_templates.py`（+58/-）
- 调整 `tests/test_prompt_strategy.py`

### 3. `0e82f0b` — 强化定向角色交互

- 修改 `prompt_pipeline.py`
- 修改 `prompt_templates.py`
- 修改 `multi_person_prompt.py`
- 同步更新 `_conf_schema.json` 及相关测试

### 4. `2b8b43f` — 工作版本快照

- 新增 `tests/test_model_refusal.py`（+184）
- 调整 `prompt_pipeline.py`
- 调整 `comfyui_runtime.py`
- 调整 `generation_task.py`

### 5. `1f7adb7` — 增强 prompt 管线 + 用量限制

- 新增 `usage_limiter.py`（+113）
- 新增 `tests/test_usage_limiter.py`
- 大幅改动 `prompt_pipeline.py`
- 大幅改动 `main.py`
- 改动 `comfyui_runtime.py`

### 6. `cd8c329` — Danbooru resolver 重构前检查点

- 改动 `danbooru_tags.py`（+56）
- 改动 `danbooru_resolver.py`（+12）
- 改动 `prompt_pipeline.py`（+96/-）
- 改动命令路由相关文件

### 7. `23d7b0d` — 验证过的 Danbooru prompt 管线

- 统计：17 个文件，+1596 行
- 核心提交：
  - 新增 `danbooru_semantic.py`（+481）
  - 大幅扩展 `danbooru_resolver.py`（+180）
  - 大幅扩展 `danbooru_tags.py`（+322）

### 8. `e170475` — 精化 prompt 管线与生成队列

- 统计：25 个文件，+1020 行
- 重点修改：
  - `prompt_builder.py`（+113/-）
  - `prompt_pipeline.py`
  - `main.py`（+61/-）
- 新增多个测试：
  - `test_manual_prompt_suffix.py`
  - `test_prompt_background.py`
  - `test_prompt_cleaning.py`
  - `test_prompt_default_migration.py`

### 9. `5921c14` — 最新检查点

- 统计：18 个文件，+895 行
- 新增：
  - `anima_api.json`（+211）
  - `prompt_background.py`（+111）
  - `prompt_keyword_rules.py`（+87）
- 继续增强 `prompt_pipeline.py`（+168/-）
- 更新 `agent_tools/comfyui_workflows.py` 及工作流测试

### 10. `0f6829a` — 最新检查点（LLM 空内容修复 + Count 模板 + futa 计数规则）

- 统计：37 个文件，+4815/-188 行
- 背景：用户反馈 LLM 经常返回空内容。经日志定位，根因是**深度思考（`reasoning_effort="high"`）与输出共享 `max_tokens` 预算，模型在 Count 决策上反复核算耗尽 token**，导致思考链被截断（日志末尾"如果我们写 Characters: togawa sakiko, togawa sakiko 就重复了。但"）、`completion_text` 为空。
- 按用户查证 danbooru 后确认的 futa 计数语义落地（**futa 计入 girls 人数**）：
  - 1女+1扶她 = `2girls, futa with female`（禁止 `2girls, futanari`，会被 Anima 误读为三人）
  - 2女+1扶她 = `3girls, futa with female`；单独扶她 = `1girl, futanari`；1男+1扶她 = `futa with male`
  - 混合场景：`2girls, 1boy, futanari`（1纯女+1扶她+1男）→ `2girls, 1boy, futa with female`，`2girls` 已含扶她不再 +1
- 三层修复：
  1. **模板层**：`prompt_templates.py` 的 `DEFAULT_LLM_PROMPT_TEMPLATE` Count 指令改为"一步查表"式（Count 人数 = Characters 项数，禁止反复核算/自我怀疑）；`_conf_schema.json`、`docs/advanced-config.example.jsonc` 默认模板同步更新，`LEGACY_BUILTIN_TEMPLATE_HASHES` 加入新模板 hash 以便自动迁移
  2. **提示词层**：`prompt_pipeline.py` system_prompt、`multi_person_prompt.py` 多人 plan count_tags 规则同步 futa 语义
  3. **确定性兜底**：重写 `prompt_pipeline.py` 的 `normalize_anima_count_tags`（新增 `_FUTA_*_KEYS` 常量组），多人路径 `deterministic_count_tags` 按已知性别直接算 Count，单人结构化路径 Count 不匹配时前置 `Npeople` 并标记 `structured_count_repaired`
- 其他修复：
  - 新增 `_count_tag(count, noun)` 辅助函数，杜绝 `1boys` 这类非法复数（混合性别场景的复数 bug）
  - `tag_cleaner.py` `MULTI_CHARACTER_BLOCKLIST` 新增 6 个 futa pair tags：单人内容流过滤 `futa with female`/`futa with male` 等，但保留单人身份 tag `futanari`/`1futanari`
- 文档：`docs/troubleshooting.md` 新增"思考链卡在 Count 决策上"成因与兜底说明；`docs/中文提示词到英文提示词流程.md` 阶段 5 新增 futa 计数规则段落
- 测试：`tests/test_prompt_strategy.py`（+429，新增 futa 归一化、混合性别单数等用例）、`tests/test_prompt_cleaning.py`（+28，futa pair tags 过滤用例）、`tests/test_variant_layout.py`（+22）等；相关套件 95 passed
- 同一提交还包含多次其他 zoo 会话的工作（均已入库），按主题分类如下：

**A. LLM 空内容防护（本次会话，即上文）**
- 见上：`_extract_completion_text` 兼容多字段、空内容关闭深度思考重试一次、`generation_task.py` 中止短路（+15/-）、`variants/turbo/low_cfg_harness/prompt_pipeline.py` 同步（+126）、`tests/test_model_refusal.py` 空响应测试（+196/-）、`docs/troubleshooting.md`（+31）

**B. 服装词库管理页面（wardrobe Plugin Page）**
- 新增 `pages/wardrobe/app.js`（+91）/`index.html`（+66）/`style.css`（+92）：服装档案、服装套组、名词翻译三个 tab 的可视化编辑页面
- `main.py`（+65）：注册 GET/POST `/astrbot_plugin_anima_master/wardrobe(/save)` Web API，含 WebUI 登录校验、`asyncio.Lock` 防并发、保存后写回 `danbooru_named_outfit_mappings`/`danbooru_term_mappings` 并持久化配置
- `danbooru_resolver.py`：新增 `wardrobe_snapshot()`/`save_wardrobe()`（`WardrobeValidationError`、revision 乐观锁防覆盖）
- `_conf_schema.json`/`_conf_sections.json`：新增 `danbooru_named_outfit_mappings`（默认 `月之森校服=tsukinomori_school_uniform`）与 `danbooru_term_mappings` 配置
- `.astrbot-plugin/i18n/en-US.json`/`zh-CN.json`（各 +8）、`metadata.yaml`（+2）、`docs/advanced-config.example.jsonc`（+6/-）同步

**C. 服装迁移（outfit transfer）证据驱动增强**
- `outfit_transfer.py`（+606）：新增 `UserOutfitPatch`/`EffectiveOutfitPlan` 请求级服装修改模型（本次唯一有效服装 tags、removed_tags 禁止恢复）、`outfit_tag_slot()` 服装槽位识别（上衣/裙/外套/腿袜等 12 类）、`_extract_target_character()` 从指令绑定穿着者、`build_outfit_transfer_block()` 迁移指令块
- `danbooru_tags.py`（+286）：`fetch_variant_outfit_profile()` 证据驱动服装档案（`sample_mode` 采样模式、`tag_counts` 词频、`anchor_tag`、stage 演出服判定、anchor 二次精化采样）
- `danbooru_semantic.py`（+242）：`prompt_context()` 扩展 authoritative visible outfit tags、named outfit-set hard tags（禁止 LLM 擅自补充套组组成）、`source_outfit_profiles` 按角色隔离、removed tags 禁止恢复
- `danbooru_resolver.py`：`remember_outfit_summary()`/`remember_named_outfit()` 持久化、`cached_named_outfits_for_prompt()`/`cached_term_mappings_for_prompt()` 缓存命中、`outfit_source_refresh_needed()` 7 天过期刷新
- 测试：`tests/test_outfit_transfer.py`（+163）、`tests/test_danbooru_semantic.py`（+570）、`tests/test_danbooru_tags.py`（+133）

**D. 画师组切换 `-sN`**
- `prompt_presets.py`（+74/-）：新增 `artist_preset_list()`（默认画师串 + 已存画师组排序）、`active_artist_tags(config, preset_index)`、`extract_artist_preset_switch()`（解析 `-s1`~`-sN`，越界/空列表返回中文错误）
- `prompt_builder.py`（+15/-）与 `variants/turbo/low_cfg_harness/prompt_builder.py`（+15/-）：消费 `_artist_preset_index`，错误时回退当前启用画师串
- `command_actions.py`（+14）：generate/multi_person 入口前置校验 `-sN` 越界并直接返回错误
- 测试：`tests/test_artist_preset_switch.py`（+196）

**E. 已清理**：`_head_test_command_router.py`（原 +3）内容是 git 报错文本 `fatal: ambiguous argument ';'`，经查证仅存在于 `0f6829a` 且为新增（`A`），全项目无任何代码引用，是终端命令重定向误创建的意外文件；已用 `git rm` 删除（待下次提交生效）。

### 11. `f3b06db` — 修改记录补全与意外文件清理

- 补全 `0f6829a` 中 LLM 空内容、Count/futa 规则、服装词库、服装迁移与画师组切换等主题记录。
- 删除终端重定向误创建且无代码引用的 `_head_test_command_router.py`。

### 12. `b886ca2` — ComfyUI 短暂 API 超时容错

- 背景：ComfyUI 进程仍在运行时，插件偶发报告“ComfyUI 未启动或无法连接”；数分钟后无需任何操作又能继续生成。历史任务记录已捕捉到根因：`/object_info` 请求触发 `ReadTimeout`，但原逻辑把所有状态检查失败统一映射为 `comfyui_offline`，并丢失原始诊断。
- `comfyui_startup.py`：状态检查遇到 `api_read_timeout` 时等待后重试一次；若仍超时，则在默认 300 秒内复用最近一次已完整验证的模型能力，避免 API 短暂忙碌直接阻止生成。真实端口拒绝、HTTP 错误等非瞬态故障不走缓存。
- `agent_tools/comfyui_status.py`：将轻量 `/system_stats` 连通性与较重的 `/object_info` 能力查询拆开；后者超时时仍保留 `comfyui_api_reachable=true`，以区分“服务仍可达但繁忙”和“进程不可达”。
- `generation_task.py`：失败任务持久化连接问题、提示和原始异常，后续可直接在任务 JSON 中定位。
- `comfyui_runtime.py`：面向用户的提示将 `api_read_timeout` 表达为“ComfyUI API 暂时无响应”，不再误称为未启动。
- 配置：新增 `readiness_cache_seconds`（默认 300，可设 0 关闭）和 `readiness_retry_delay_seconds`（默认 2）。
- 测试：新增 `tests/test_comfyui_readiness.py`，覆盖最近成功状态缓存与 `/object_info` 超时仍保留 API 可达性；相关测试通过（7 passed）。

### 13. 本次提交 — 服装套组别名模型与 UI 状态修复

- 修复服装词库页面的未保存提示：页面加载、保存或放弃修改后，`hidden` 不再被 `.dirty-bar` 的 `display: flex` 覆盖。
- 服装套组改为以 canonical tag 唯一标识；相同套组重复自动学习时合并档案与别名，旧重复记录在页面中折叠，并在后续学习或保存时收敛。
- canonical 下划线 tag 只保存在 tag 字段；别名列表分别保存中文名和空格英文名。运行时任一别名只注入一次 canonical hard tag。
- 服装套组 UI 增加“触发别名（中文 / 英文）”多值编辑框，一条 canonical tag 对应多个可见、可编辑别名。
- `wardrobe_snapshot()` 合并手动配置与自动学习的同 tag 条目，手动配置优先，同时保留可用别名。
- 测试新增自动学习合并、旧数据折叠、中英文别名 UI 往返、重复 hard tag 去重和 `hidden` 样式检查；服装与提示词相关套件 `121 passed`。

### 14. 本次提交 — 轻量 ComfyUI 健康检查

- 复现与证据：用户在没有执行生成任务时仍收到“ComfyUI API 暂时无响应”。现场测得 `/system_stats` 约 0.63 秒、`/queue` 约 0.02 秒且队列为空，但 `/object_info` 返回约 3.7 MB，耗时约 17.96 秒；原来的 20 秒读取阈值会因轻微波动而超时。
- `agent_tools/comfyui_agent.py`：`status` 增加 `--quick`，用于只探测 API 存活而不下载节点/模型清单。
- `agent_tools/comfyui_status.py`：支持轻量状态模式；完整能力校验的 `/object_info` 超时提高到 60 秒，并明确其只用于首次配置/模型验证。
- `comfyui_startup.py` / `comfyui_runtime.py`：首次完整验证成功后，后续生图调用轻量 `/system_stats` 健康检查并复用已验证的模型能力，避免每次生成重复读取数 MB 的 `/object_info`。
- 测试：`tests/test_comfyui_readiness.py` 增加轻量健康检查用例；相关测试通过（8 passed）。

## 总结

codex 在本地对 **Anima（astrbot_plugin_anima_master）** 插件的定制主要围绕以下方向：

1. **Prompt 生成管线**：从结构化 prompt 协议、角色交互，到 Danbooru 语义标签解析与关键词规则的持续迭代，是最核心的工作主线。现在版本不使用“多人”工作流也能较稳定地产出多个角色互动图片。但稳定程度仍低于“多人”工作流
2. **生成队列与用量控制**：引入 `usage_limiter.py` 限额机制，精化 `main.py` 的生成队列与投递逻辑。
3. **ComfyUI 工作流**：运行时与工作流定义、启动/状态相关模块的配合调整。
4. **测试配套**：为每项功能同步补充了大量单元测试。
5. **深度思考参数传递的实现有bug**Codex说明这是astrbot源码中含有的问题，需要修改：“AstrBot 的插件接口明确承诺支持 OpenAI-compatible **kwargs，见 [context.py (line 195)](C:/Users/69493/.astrbot_launcher/instances/1617d9a4-b150-48c1-a04e-cc719eaf7a73/core/astrbot/core/star/context.py:195)，但 OpenAI Provider 原先接收到参数后，没有把它们放进最终 payload。这确实属于核心 Provider 适配层的漏传，而不是插件没找到开关。”修改目标是“日常提示词优化明确发送 thinking: disabled；关键词触发深度思考时仍会重新开启：[prompt_pipeline.py (line 568)](C:/Users/69493/.astrbot_launcher/instances/1617d9a4-b150-48c1-a04e-cc719eaf7a73/core/data/plugins/astrbot_plugin_anima_master/prompt_pipeline.py:568)
AstrBot 现在仅透传 max_tokens、thinking、reasoning_effort，没有放开其他参数：[openai_source.py (line 995)](C:/Users/69493/.astrbot_launcher/instances/1617d9a4-b150-48c1-a04e-cc719eaf7a73/core/astrbot/core/provider/sources/openai_source.py:995)”需要注意astrbot源码的修改不在此branch中。
修改后的文件993-1002行：

  model = model or self.get_model()

        payloads = {"messages": context_query, "model": model}
        for key in ("max_tokens", "thinking", "reasoning_effort"):
            if key in kwargs:
                payloads[key] = kwargs[key]

        self._finally_convert_payload(payloads)

        return payloads, context_query

整体提交风格以 `chore:`（快照/检查点）、`feat:`（功能）与 `fix:`（缺陷修复）交替出现，共 14 次提交、涉及 90+ 个文件的持续定制开发。`0f6829a` 除标签学习管理页面等新功能外，还针对用户反馈的 **LLM 空内容问题**完成了 Count 模板"一步查表"化、futa 计数规则统一与确定性 Count 兜底；后续提交补足了 ComfyUI API 短暂超时容错、轻量健康检查，并统一了服装套组 canonical tag 与中英文触发别名的存储和 UI 表现。
