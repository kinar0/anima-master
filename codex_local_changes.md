# Codex 本地修改记录（Anima 插件）

> 数据来源：仓库 git 提交历史
> 分支：`codex/local-anima-custom`
> 提交作者：`Local AstrBot Customization <local-astrbot@localhost>`
> 提交总数：9 次

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

## 总结

codex 在本地对 **Anima（astrbot_plugin_anima_master）** 插件的定制主要围绕以下方向：

1. **Prompt 生成管线**：从结构化 prompt 协议、角色交互，到 Danbooru 语义标签解析与关键词规则的持续迭代，是最核心的工作主线。
2. **生成队列与用量控制**：引入 `usage_limiter.py` 限额机制，精化 `main.py` 的生成队列与投递逻辑。
3. **ComfyUI 工作流**：运行时与工作流定义、启动/状态相关模块的配合调整。
4. **测试配套**：为每项功能同步补充了大量单元测试。

整体提交风格以 `chore:`（快照/检查点）与 `feat:`（功能）交替出现，共 9 次提交、涉及 90+ 个文件的持续定制开发。
