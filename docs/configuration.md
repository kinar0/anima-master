# 配置说明

配置页按使用频率分组。首次使用时，先看“ComfyUI 连接”和“出图参数”。较少使用的项目会收进“更多配置”。

如果你想手动做高度自定义，请对照这些文件：

- `docs/advanced-config.example.jsonc`
- `data/config/astrbot_plugin_anima_master_config.example.jsonc`

真正生效的运行文件是 `data/config/astrbot_plugin_anima_master_config.json`。它是标准 JSON，不能直接写注释。

## 最低启动清单

能发起生图前，至少确认这些配置：

```text
comfyui_base_url = AstrBot 能访问到的 ComfyUI 地址
workflow = 插件支持的工作流预设
unet_name = ComfyUI 中的模型文件名
clip_name = ComfyUI 中的文本编码器文件名
vae_name = ComfyUI 中的 VAE 文件名
```

多数用户可以先用绘世启动器、ComfyUI portable 或自己的脚本手动启动 ComfyUI。如果开启 `auto_start`，还必须填写 `startup_command`。不开 `auto_start` 时，不需要填写启动命令。

这些项目主要分布在“ComfyUI 连接”和“出图参数”两个配置组里。

## 预设

- `reset_to_defaults`：一键恢复默认配置。打开并保存后，仅在下一次插件加载时执行一次，完成后会自动关闭。
- `chiyo_preset`：选择千代预设。可选 `未启用` / `千代base` / `千代aesthetic` / `千代turbo`（配置值分别为空、`base`、`aesthetic`、`turbo`）。

未启用时使用当前配置。选择任一千代预设后，都会把对应的千代画师组写入画师 tags，并把狐莉加入固定角色。狐莉不是默认角色，只有指令里明确提到“狐莉”时才会使用。

各预设差异：

| 预设 | UNet | CFG | 质量词 / 负面词 |
| --- | --- | --- | --- |
| 千代base | `anima_baseV10.safetensors` | 5 | 使用当前配置的质量词与负面词 |
| 千代aesthetic | `anima_aestheticV11.safetensors` | 3 | 都不注入 |
| 千代turbo | `anima_baseV10.safetensors` + `anima-turbo-lora-v0.2.safetensors` | 1 | 使用当前配置的质量词与负面词 |

千代turbo 使用 `variants/turbo/workflows/comfyui_00051_api.json`，固定为 10 步、`euler`、`simple`，并启用面向 CFG 1 的二次 LLM 约束规划。该规划会前置明确要求、删除冲突或稀释 Tag、为明确要求的画风加权，并可把内容段进一步限制在 20–80 个 Tag；存在明确约束时不会触发普通的长度重试。

首次启用预设时，插件会在插件数据目录保存模型、生成参数、正负面提示词和自定义工作流字段的基础快照，再把预设值回写到配置页；内部快照不会显示为配置项。保存并重载插件后刷新页面即可看到实际值。切换预设时会从同一份基础快照重新计算，选择“未启用”则恢复启用前的值。旧配置里的 `chiyo_preset_enabled` 与旧画师组名“千代风格”“千代画风”仍可识别，并会迁移到千代base。已有的“千代turbo”和“千代turbo2”画师组会保留；千代turbo 预设优先使用同名画师组，不会覆盖用户保存的内容。

## ComfyUI 连接

- `comfyui_base_url`：AstrBot 能访问到的 ComfyUI 地址。
- `workflow`：工作流预设，默认使用 `anima_t2i`。
- `custom_workflow_enabled`：是否改用自定义 ComfyUI API 工作流；普通版默认关闭。
- `custom_workflow_path`：自定义工作流 JSON 路径；千代turbo 会自动使用 `variants/turbo/` 中的工作流。
- `timeout`：等待生成完成的最长时间。
- `storage_retention_days`：输入图片、输出图片和逐任务状态文件的保留天数；设为 `0` 可关闭按时间清理。
- `manifest_max_records`：输入图片清单压缩后保留的最大有效记录数，默认 `5000`。
- `poll_interval`：查询 ComfyUI 生成状态的间隔。

`custom_workflow_override_parameters` 默认关闭。关闭时保留自定义工作流中的尺寸、采样步数、CFG、采样器和调度器；Turbo 工作流应保持关闭。需要统一使用插件出图参数时再开启。

### NovelAI（NAI）工作流

支持通过 ComfyUI 的 `ComfyUI_RS_NAI_API_Request` 中的 `NovelAIGenerator`
调用 NAI。账户认证仍由该 ComfyUI 扩展管理；AstrBot 继续连接 ComfyUI 地址。
无需在 AstrBot 中另外填写 NAI 密钥，也不使用本地 Anima 模型加载器。

本仓库的 `nai.json` 是可视化编辑格式；配套的 `nai_api.json` 是供插件提交的
API 格式，保留已连接的生成链和采样设置，以中性占位词替换示例提示词。
原图中未连接到生成器的 `CharacterPromptSelect` 分支不包含在 API 文件中。
在插件的“ComfyUI 连接”配置组设置：

```text
custom_workflow_enabled = true
custom_workflow_path = nai_api.json
custom_workflow_override_parameters = false
```

保存并重载插件后，继续使用 `/anm` 或 `/anm 无优化`。此接入复用现有的
中文优化、角色/服装证据和图片回传；没有新增 `/nai` 命令。普通请求沿用
整体提示词。正文中单独写 `-r` 时，先走现有角色/衣柜证据流程，再让模型
理解视角、人物互动和高低/前后关系，自主构思布局，再把每个可见角色实例的
身份、外貌、服装、动作与 XY 位置拆开；同一角色在多个格子出现
会占多个角色框（最多五个）。全局质量词、画师、背景、光影、布局等留在原正向
输入。`-r` 只在 `#` 前识别，`#` 后手工 tag 仍追加到全局输入。该模式要求
`NovelAIGenerator` 工作流与提示词优化；原样/关闭优化模式不支持。
规划失败时停止提交，不退回整体提示词。
配置中的正向质量词、画师组和负面词仍按原有规则生效，
不会保留 JSON 示例里的正负提示词。建议千代预设保持“未启用”，避免其 Anima
参数、质量词或 Turbo 工作流覆盖自己的设置。已有预设应先关闭、保存并重载，
然后再设置上述自定义工作流。

NAI 提示词支持生成节点上的直接字符串，以及经 `Textbox`、
`ComfyUIToNovelAIV4` 连接的字符串；插件沿连线替换原始文本，保留转换节点，
由扩展把 ComfyUI 权重语法转换为 NAI 格式。未知字符串处理节点、循环连线或
正负共用输入会在提交前报错。该接入不是 NAI 专用提示词优化器；生成质量仍需
结合 NAI 模型调整质量词、画师词等配置。

`-r` 模式会在提交的 API 图中连接 `CharacterPromptSelect`，为每个实例提供
独立的 `ComfyUIToNovelAIV4` 角色提示词与 0–1 坐标。仓库内的
`nai_api.json` 无需预先保留示例角色输入；不带 `-r` 时不增加角色节点。
规划结果还记录简短的整体构图分析和逐角色位置理由，供任务摘要排查；主机只
校验坐标范围，不按“坐椅子”“跪在地上”等词套固定坐标或改写模型所选位置。
位置规划模型还收到本次实际画布的宽、高、宽高比和横竖方向。未单独指定尺寸且
关闭参数覆盖时读取 NAI API 工作流中的画布尺寸；单次指定尺寸或开启参数覆盖时
使用本次提交的宽高。位置仍用相对于该画布的 0–1 坐标表示。

默认使用 API 文件中的 `NAI Diffusion V5 Full`、1024×1536、28 步、CFG 6、
`k_euler/karras`；种子每次由插件传入。开启参数覆盖只覆盖 NAI 的宽高、步数
和 CFG，始终保留 JSON 中的 NAI 专用采样器和调度器，避免注入不兼容的
`er_sde/normal`。单次明确尺寸也会覆盖宽高，但 NAI 宽高必须是 64 的倍数，
同时仍需满足聊天命令的尺寸范围；例如 `--尺寸 832x1216`。

`limit_opus_free` 保留原工作流的 `false`，这不保证免费额度；实际计费取决于
账户和生成参数。可在 `nai_api.json` 中按需调整此节点选项。接入测试默认仅
验证本地工作流和接口结构，不自动提交收费生图请求。

若要切回内置 Anima，关闭自定义工作流；若要切回原自定义 Anima，将路径改回
`anima_api.json`。本接入仅用于文本生图，实验性改图仍走原来的 Anima 路线。

`127.0.0.1` 指 AstrBot 所在机器。跨机器部署时，请填写局域网或 Tailscale 地址。

## 出图

- `width` / `height`：默认尺寸；宽和高都必须在 832–1756（含边界），且为 4 的倍数。
- `allowed_sizes`：只用于“竖图、横图、宽屏”等比例快捷词的候选列表，不是明确数字尺寸的白名单。
- `steps`：采样步数。
- `cfg`：CFG 强度。
- `sampler_name` / `scheduler`：采样器和调度器。
- `unet_name` / `clip_name` / `vae_name`：模型文件名。
- `quality_prefix`：固定拼接在正面提示词前面的质量词。
- `negative_prompt`：默认负面提示词。

模型文件名只填文件名，不填路径。

## 提示词

- `prompt_optimize_enabled`：是否让聊天模型优化自然语言提示词。
- `prompt_builder_provider_id`：指定用于优化提示词的模型。留空时使用当前会话主模型。
- `prompt_builder_max_tokens`：提示词优化模型最大输出长度，同时约束两次 LLM；第一次语义规划最多使用 900 tokens，以容纳复数角色的逐角色意图 JSON。
- `prompt_builder_max_content_tags`：LLM 内容段的硬上限，默认 65；不计算质量词、固定角色、画师组和画风。自然语言主题通常以 40–55 个内容 Tag 为目标，简单表情包或头像可以使用 30–45 个，复杂服装或构图约 60 个；已经是 Tag 串的输入不设最低数量。普通模式会在去除较多同义词后尝试一次按缺失画面槽位补全。
- `unspecified_wardrobe_policy`：按角色处理“没有实际服装基础”的策略。`scene_adaptive`（默认）把角色 default 作为软参考；中性场景更倾向沿用，涩气或强场景允许第二次 LLM 按情境替换或补全。也可固定为 `default_profile`（仍以软参考方式提供）或 `creative_fallback`（完全不加载 default）。
- `scene_adaptive_wardrobe_markers`：场景自适应的强触发词列表。默认包含泳池边、出浴、睡眠前后、比赛、运动/训练结束、健身、游泳结束和演出结束等；普通公园、在家、吃饭不会触发。
- `danbooru_semantic_system_prompt`：第一次 LLM 的意图抽取规则，只负责可见角色、明确服装或未知、服装修改和外貌修改。它不应生成 tag、candidate、lookup 或内部 ID；旧内置默认值会自动迁移，本地角色/衣柜匹配由主机随后完成。
- `prompt_builder_template`：第二次 LLM 的七段英文提示词模板。

通常不需要一开始就改这两个模板。第一次 LLM 不调用 Codex 的 `danbooru-tags` skill，也不负责猜 tag；插件的本地 resolver 在意图抽取后匹配档案并验证证据。`unspecified_wardrobe_policy` 按角色生效：别的角色有服装证据不会关闭当前角色的缺省参考；当前角色已有普通衣物、命名套组或来源服装时则不加载 default。第二次 LLM 始终读取完整原文，并结合 LLM1 关系建议、数据库证据和禁止恢复项完成最终语义。

## 联网与 tag 查询

- `prompt_builder_web_search_enabled`：允许在需要时联网搜索。
- `prompt_builder_search_query_template`：搜索查询模板。
- `danbooru_core_tag_lookup_enabled`：是否启用旧的联网角色核心 tag 校正，默认关闭；关闭时仍保留 LLM 候选与本地 `danbooru-tags` 语义校验，也会继续使用已经持久化的服装档案。
- `danbooru_tag_base_urls`：优先使用的 Donmai tag API 地址。

联网搜索需要 AstrBot 全局 Tavily key。搜索失败会自动降级，不会中断生图。

## 服装词库页面

重载插件后，可以从 AstrBot WebUI 的插件详情页进入“服装词库”。页面提供三个可搜索、可增删改的列表：

- **服装档案**：按角色或服装来源保存经过帖子样本验证的可见服饰结构。
- **服装套组**：保存独立命名的完整服装套组及其 canonical Danbooru tag。
- **名词翻译**：把中文服装名词稳定映射到 canonical Danbooru tag。

服装套组以“独立套组实体 + 变体”维护触发别名，每个实体指向一个 canonical tag。例如：

```text
Canonical Tag: haneoka_school_uniform
触发别名: 羽丘校服, haneoka school uniform
```

下划线形式 `haneoka_school_uniform` 只保存在 canonical tag 字段，不会重复写进别名列表；中文名和空格英文名会分别保存并参与匹配。不同套组实体允许指向同一个 canonical tag，不会仅因 tag 相同就强制合并；最终 hard tag 会去重，但各套组仍保留自己的别名、变体和角色绑定 anchor。手动配置的别名优先于冲突的自动学习别名。

页面保存使用 revision 检查，若生成期间词库已被其他请求更新，会要求刷新后重试，避免覆盖新数据。保存成功后配置与运行时缓存立即更新，不需要为每次词库编辑重载插件；只有安装或更新本页代码后需要先重载一次插件。

配置页中的 `danbooru_named_outfit_mappings` 和 `danbooru_term_mappings` 仍可直接编辑。前者是“服装套组”的兼容配置入口，格式为 `别名1 | 别名2=canonical_tag`；后者是没有角色归属的全局 hard tag 映射。推荐日常使用“服装词库”页面，减少格式错误。完整的数据源职责、角色绑定和最终注入规则见 [角色与服装配置工作流](角色与服装配置工作流.md)。

人物检测还支持两个上下文映射：`danbooru_series_alias_mappings` 把本地化标题、续作或外传名称归入角色 tag 实际使用的作品家族；`danbooru_character_alias_mappings` 把角色俗称映射到 canonical character tag。两者都使用 `别名1 | 别名2=canonical_tag`。带作品后缀的角色映射只有在同一请求也命中兼容作品家族时才会进入本地精确校验，因此单独出现的同名角色不会被强行消歧。

## 角色与画风

- `fixed_characters`：固定角色辅助信息；普通结构化路径主要把它作为身份提示，多人兼容路径和旧回退路径会更强地注入。不要在其中保存服装、动作、背景、质量词或画师词，详见 [角色与服装配置工作流](角色与服装配置工作流.md)。
- `default_artist_tags`：未启用画师组时使用的备用画师 tags。
- `style_tags`（画风）：独立于画师组的画风 tags，拼接在当前画师组之后。
- `style_presets`：已保存的画风列表。
- `active_style_preset`：当前启用的画风；留空时使用 `style_tags`。
- `sensual_mode_enabled`：涩气表现力优化。
- `sensual_mode_markers`：触发涩气表现力优化的关键词。
- `keyword_prompt_rules_enabled`：启用关键词强制规则。
- `keyword_prompt_rules`：命中用户原始文字后，向 LLM 追加高优先级硬性指令。

关键词强制规则每行使用以下格式，多个触发词用 `|` 分隔：

```text
黑丝|丝袜 => 必须在 Tags 中明确加入 black pantyhose；不要使用近义词代替
雨天|下雨 => 必须加入 rain、wet clothes，并把场景设置为雨中
```

同一行命中任一触发词即生效；多行可以同时生效。检测仅针对用户原始文字，参考图反推内容或搜索摘要不会误触发。格式错误的行会被忽略。

固定角色格式：

```text
角色名=danbooru tags
```

## 实验功能

图生图、放大和去背景仍在开发中。配置项保留给后续版本使用，不建议作为稳定功能依赖。

改图工作流会先用 `max_image_side` 限制输入图的基准最长边，再把基准宽高乘以
`upscale_factor`，并将结果写入 ComfyUI 的 `ImageScale`（“缩放图像”）节点。
例如基准尺寸为 `832x1216`、倍率为 `1.5` 时，改图尺寸为 `1248x1824`。

## 发送与权限

### 按会话发送前打码

插件详情页的“服装词库”页面内“图片打码设置”区域按 AstrBot 完整会话 ID（平台、群聊/私聊类型、会话 ID）分别保存开关。会话首次请求图片后会自动出现在页面，也可手工添加完整会话 ID。默认关闭。开启的会话在发送每张生成图片前，把该图片上传到 ComfyUI，运行插件目录中的 `autofilter2.json`，替换工作流 `LoadImage` 的图片输入，并把 `SaveImage` 的输出前缀临时换成 Windows 安全的唯一名称；其他节点和参数沿用文件中的值，原 `autofilter2.json` 不会被修改；聊天收到的是 `SaveImage` 输出。关闭时直接发送原图。打码工作流失败时不会发送原图，会提示发送失败。该功能依赖 ComfyUI 安装新工作流所需的 Impact Pack、Impact Subpack、Essentials 节点及 `bbox/censor_detect_v1.0_s.pt` 检测模型；嵌入的 Image Blur 子图由插件展开为 GLSLShader API 节点。

会话开关保存在插件数据目录的 `autofilter_sessions.json`，页面保存后立即生效，不需重载插件。

- `send_result_to_chat`：是否把图片发回聊天。
- `max_send_images`：最多发送几张。
- `notify_drawing_and_at_sender`：同一个开关控制两项行为：接受生图请求后发送动态绘图进度；前方没有未完成申请时显示“正在绘画中”，否则显示“正在绘画中，前面还有X人”（X 不包括申请人自己）；图片完成后在群聊中 At 原申请人并告知今日剩余次数。不限额和白名单用户显示“今日剩余次数：不限”；私聊不会 At。
- `admin_only`：是否仅管理员可用。
- `allowed_sender_ids`：允许使用的用户 ID 列表。
- `daily_generation_limit`：每个 QQ 号每天最多发起的生图次数；`0` 表示不限制。一次多人请求及其候选重试只计一次，请求被接受后即占用次数。
- `generation_whitelist_sender_ids`：免限额白名单。名单内 QQ 不受每日次数和 `allowed_sender_ids` 限制。
- `generation_blacklist_sender_ids`：黑名单，优先级最高；同时出现在黑白名单时按黑名单处理。

每日计数按 AstrBot 服务器本地日期统计，并保存在插件数据目录的 `daily_generation_usage.json`，重启 AstrBot 不会清空当天计数。

## 由 AstrBot 启动 ComfyUI

`auto_start` 不是开机自启。它只会在 ComfyUI 离线且收到绘图请求时，在 AstrBot 所在机器上执行 `startup_command`。

不开 `auto_start` 时，不需要填写 `startup_command`，但需要你先手动启动 ComfyUI。开启 `auto_start` 后，`startup_command` 就是必填项。

`startup_command` 必须是能直接启动 ComfyUI 服务的命令。ComfyUI portable 常见是 `run_nvidia_gpu.bat` 一类脚本；绘世启动器如果只是打开启动器界面，不等于 ComfyUI 服务已经启动。

如果 ComfyUI 在另一台机器，AstrBot 不会直接启动远端 ComfyUI。你需要自行配置 SSH、Tailscale SSH 或其它远程启动脚本。

## 调试

调试项默认关闭。只有排查问题时再打开：

- `debug_prompt_enabled`
- `debug_image_reference_enabled`
- `debug_send_payload_enabled`

开启后可能在日志和 `last_task.json` 中记录较长提示词，请注意隐私。

`debug_prompt_enabled` 会同时记录第一次语义规划 LLM 的任务书、system prompt、完整原始回复、解析后的 anchors、恢复前后的 `character_plans`，以及第二次提示词 LLM 的任务书、回复和最终 prompt。任务摘要还会保存 `semantic_plan_prompt` 与 `semantic_plan_raw`，便于判断模型没有输出关系，还是关系在严格解析时被丢弃。

排查服装串档时，优先查看最近任务摘要中的这些字段：

- `semantic_character_outfits`：第一次 LLM 给每名角色分配的服装。
- `explicit_wardrobe_evidence` / `wardrobe_source`：本次是否存在明确服装证据，以及最终采用哪个来源。
- `wardrobe_authority.cached_tags` / `explicit_tags` / `selected_tags` / `source_grounding_tags` / `stale_cached_tags`：缓存、明确要求、最终选中、允许 cosplay 补全参考的来源衣柜，以及应删除的穿着者陈旧衣柜。
- `outfit_summary_source`：存在逐角色计划时应为 `suppressed_by_character_authority`，避免请求级摘要重新注入旧衣服。

## 推荐做法

普通使用：

1. 先在 WebUI 里只改常用项。
2. 需要更细的控制时，再展开“更多配置”。

深度自定义：

1. 先打开带注释模板。
2. 对照修改真实运行文件 `data/config/astrbot_plugin_anima_master_config.json`。
3. 保存后重载插件。


角色衣橱的“服装名称 / 变体”现支持任意名称；原五类仅作建议。所属角色与学习来源分别保存为 `ownerTags` / `sourceTags`，同一角色的每套命名服装独立选择，旧数据保持兼容。详见 [自由命名衣橱](角色与服装配置工作流.md#自由命名衣橱与完整造型标签2026-09-12)。
