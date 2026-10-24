# AstrBot 4.29.1 更新兼容检查

检查日期：2026-10-22（Asia/Shanghai）。保留检查前工作区已有的未提交修改。

## 结论

发现并修复一处真实的模型选择兼容问题。修复后 **1215 项离线测试通过**，另有 **59 项真实模型测试未运行**；四个现有页面浏览器测试全部通过。ComfyUI、当前 NAI 工作流所需节点、打码节点/检测模型、本地 Danbooru 查询、Toolbox 检索和图片元数据提取均通过本机检查。

本次没有调用收费模型、提交生图、给聊天发消息、改动用户词库或重载运行中的插件。**代码修复需在 AstrBot 中重载本插件后生效。**

## 环境

- AstrBot 源码与实际 WebUI 均为 4.29.1；启动器实际 WebUI 地址是 `http://localhost:18575`。配置文件中的 6185 不代表启动器当前监听端口。
- Python 为实例目录的 `venv/Scripts/python.exe`，版本 3.12.13；旧 `core/.venv` 已不存在。插件辅助子进程已有 `sys.executable` 回退，本次真实子进程执行成功。
- ComfyUI 0.33.2，`http://127.0.0.1:8188/` 在线。当前配置使用 `nai_api.json` 自定义工作流。
- WebUI 未登录，因此没有直接读取运行中服务的受保护插件/日志 API。完整插件初始化和接口验证使用当前安装的真实 Context、PluginRequest 及请求绑定机制，数据与工具状态隔离在测试目录中；不能把这一检查说成生产页面已登录验收。
- AstrBot 文件日志未开启。更新迁移后的旧任务记录仍存在，最近五条为更新前的 19:28–19:42 成功记录，不能据此证明更新后的生成与回传已成功。

## 修复

新版 `ProviderManager` 从 `agent_runner.config.model.provider_id` 和会话偏好选择默认聊天模型。本机旧 `provider_settings.default_provider_id` 已不存在，专用 `default_image_caption_provider_id` 为空。原 `ReferenceContextBuilder` 和 `GenerationVerifier` 只读取后两者，会得到空 provider：影响反推、参考图理解和自动验图，不影响已显式指定 DeepSeek provider 的普通提示词写作。

新增 `provider_selection.image_provider_id`，共享以下顺序：专用图片理解模型 → AstrBot `get_current_chat_provider_id(umo)` → 旧默认字段。验图入口继续保留插件指定验图模型与多人专用验图模型的更高优先级。配置读取或 SDK 查询失败继续保守回退，不猜模型、不解析新版私有配置结构、不增加 LLM 调用。模型仍需支持图片输入，本次未测试其实际视觉能力。

回归覆盖真实 Context 的会话解析方法、反推实际转发的 provider/图片参数、显式模型优先级、旧宿主缺方法/报错/空结果，以及缺失或异常配置。另有完整插件构造、初始化、15 个 Web API 注册及三个只读入口的真实请求对象测试。

同时处理检查中暴露的维护偏差：

- `_conf_sections.json` 补齐已存在的 Toolbox 三项、每日总次数及两项 LLM 日志开关；不修改运行配置。
- schema 与高级示例的第一阶段默认 system 同步到当前内置原文，补齐已有的完整角色名单要求。运行时原本会迁移这一旧默认文本，未改写两阶段任务书或自定义提示词。
- 页面测试改用新版 `process_plugin_page_html` 和路径票据，验证衣柜到内嵌 Toolbox 再返回、CSS/JS 相对链接及 bridge 查询票据。
- 旧测试预期同步到已存在的双页面、tag 下划线显示为空格、空模板选用内置默认、模型判断背景的契约；没有为通过测试修改生图提示词。
- `AGENTS.md` 与配置工具示例更新实际 Python 路径，并记录测试临时目录的权限处理。

## 验证范围

| 功能 | 结果与证据 |
| --- | --- |
| 命令/别名/尺寸/屏蔽/权限/配额与统计 | 离线测试通过，含命令入口拒绝与并发隔离 |
| 普通优化、raw、手工尾缀、角色/衣柜/外貌权限 | 离线测试通过，未重新做真实模型语义验收 |
| NAI `-r`、多人实例、原生文字、方向标签与修复 | 离线测试通过；本机工作流可构造，9 个节点的类型全部存在，坐标 0.3427 保持精度 |
| 普通 NAI 工作流 | 6 个节点的类型全部存在，配置的 NAI Diffusion V5 Full 在节点模型列表中 |
| 内置 Anima 与实验改图 | 离线工作流测试通过；本机 UNet、CLIP、VAE 与图生图节点可用；未运行 GPU 生成 |
| 打码 | 离线交付边界测试通过；本机工作流节点齐全，`bbox/censor_detect_v1.0_s.pt` 在可选检测模型列表中 |
| 图片解析 | 从 core 目录启动真实图片 helper，测试 PNG 的正负提示词均正确读出 |
| 图片反推/参考图/验图 | provider 兼容修复及 SDK 测试通过；未调用真实视觉模型 |
| 画师/预设/每日次数/图片回传/对白排字 | 离线测试通过；未向 QQ/OneBot 实际发送图片 |
| 本地 Danbooru | 用插件真实 `lookup_semantic_anchors` 查询 clothing 组，状态 resolved，确认 white_dress |
| Toolbox | 当前 36893 条 general 数据可读，“公主抱”检索命中 princess_carry，有预算参考；首次加载约 16 秒，不代表每次请求耗时 |
| 衣柜/词库/审核/批量翻译页面 | 四组 Playwright + Chrome 浏览器测试通过，覆盖 30001 条分页、完整搜索、跨页保存、草稿/冲突与慢请求 |
| 批量名词翻译与复核 | 离线断点、导入、权限测试通过，未调用真实翻译模型或导入数据 |
| 放大/去背景 | 仍是文档中的开发/非稳定功能；本机没有 BiRefNetRMBG 节点，不属于此次更新导致的已确认回归 |

最终全量测试记录：`pytest-verified.log` / `pytest-verified.xml`，1215 passed、59 deselected、3 warnings。两条警告是仍可工作的 AstrBot 同步停用工具 API 弃用提示，一条是 core 的 audioop 弃用提示，未造成测试失败。页面记录：`ui.log`。本机服务详情：`local-services.json`。

复现离线检查时，从插件目录用实例 `venv` 的 Python，设置 `PYTHONPATH` 为 core，执行 `-m pytest -p no:cacheprovider tests -m 'not live_llm' -q`。若系统默认 pytest 临时目录拒绝访问，将 `TEMP`/`TMP` 与全新的 `--basetemp` 指向插件内专用目录。最初临时目录权限错误已在环境层解决，不计为插件功能故障。

## 尚未验收

本次未启用 59 项收费模型测试、未执行真实 ComfyUI/NAI 生图、未测试当前模型识图质量与成图表现、未实际通过平台发送图片。重载插件后的一次聊天生图仍可验证运行中实例的模型调用、工作流执行、下载、打码及聊天交付整条链路。本次的离线和只读结果不能替代该验证。
