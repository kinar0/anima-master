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

`active_artist_preset` 是尚未在聊天中单独选择画师组的会话的初始值。聊天命令选择保存在插件数据目录的 `artist_sessions.json`，按群聊/私聊会话分别生效，并且重载后保留；画师组本身仍是共享配置。配置页改动初始值不会覆盖已有会话选择。`-sN` 只影响单次生成。

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

2026-10-15 原生文字补充：带明确文字的优化请求自动复用逐实例 writer，即使未写 `-r` 也按画格/角色实例分配台词。对白各进所属实例 `Text:` 尾段，旁白等无主文字保留画格与位置说明后进入 base；不是将全篇台词集中到 base。没有文字需求时仍按下文普通/`-r` 边界处理。raw/关闭优化只保留显式原生尾段，不自动绑定；已有发送前排字开关保持独立。详见 [NAI 原生文字生成](NAI原生文字生成.md)。

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
整体提示词。正文中单独写 `-r` 时，保留第一次意图规划与本地角色/衣柜证据，专用第二次 LLM 直接输出 JSON，让模型
理解视角、人物互动和高低/前后关系，自主构思布局，再把每个可见角色实例的
身份、外貌、服装、动作与 XY 位置拆开；同一角色在多个格子出现
会占多个角色框（V5 文生图最多 22 个，V4/V4.5 最多 6 个）。全局质量词、画师、背景、光影、布局等留在原正向
输入。`-r` 只在 `#` 前识别，`#` 后手工 tag 仍追加到全局输入。该模式要求
`NovelAIGenerator` 工作流与提示词优化；原样/关闭优化模式不支持。
规划失败时停止提交，不退回整体提示词。
配置中的正向质量词、画师组和负面词仍按原有规则生效，
不会保留 JSON 示例里的正负提示词。建议千代预设保持“未启用”，避免其 Anima
参数、质量词或 Turbo 工作流覆盖自己的设置。已有预设应先关闭、保存并重载，
然后再设置上述自定义工作流。

`-r` 布局模型必须返回全局提示词、每个可见角色实例的提示词与坐标，
以及简短的构图分析和位置理由。其输出上限取
`max(1200, min(7200, max(prompt_builder_max_tokens, 1200 + 260 × 模型角色上限)))`；
复杂多人请求可能因 JSON 不完整而停止提交。启用画师词时，主机在布局结果
解析后按当前画师组（含本次 `-sN` 选择）恢复全局画师串，并替换同名的错误副本；
`(yd_(orange_maru):1.1)` 等嵌套括号加权项保留原始写法。

重复身份涉及修改时，专用 JSON 用 `source_text` 的原文片段（或片段数组）声明每格适用的要求，衣柜不同再用 `anchor_id` 区分本地计划；这些辅助字段只用于权限和诊断，不写进 `ComfyUIToNovelAIV4` 的 prompt。无法定位实例或原文证据时停止，正常主链不追加第三次拆分或审查调用；方向配对结构错误时可额外修复一次。

按 NovelAI 的 base prompt / character prompt 边界，布局任务书把无数字性别、身份、
外貌、服装、姿势、表情、身体与头脸朝向、视线、动作、所持物以及人物与环境的
局部关系限定在对应角色框中；`sitting on table`、`lying on bed`、`leaning against
wall` 均属于角色实例条件。全局提示词承载总人数、画师、质量、版权、整幅背景或
场景、共享环境物件、整图镜头、光影、色彩和布局。背景里的 `bed` 可以继续全局
存在，即使角色框用 `lying on bed` 引用它；解析器不会仅因两处重复就删除背景物件。
`looking at viewer`、`school uniform`、`hug` 等明确角色条件即使适用于所有人物，
也必须分别写入角色框；解析器会确定性删除误放到 `global_prompt` 的明确角色项。

NAI 提示词支持生成节点上的直接字符串，以及经 `Textbox`、
`ComfyUIToNovelAIV4` 连接的字符串；插件沿连线替换原始文本，保留转换节点，
由扩展把 ComfyUI 权重语法转换为 NAI 格式。未知字符串处理节点、循环连线或
正负共用输入会在提交前报错。该接入不是 NAI 专用提示词优化器；生成质量仍需
结合 NAI 模型调整质量词、画师词等配置。

`-r` 模式会在提交的 API 图中连接 `CharacterPromptSelect`，为每个实例提供
独立的 `ComfyUIToNovelAIV4` 角色提示词与 0–1 坐标。仓库内的
`nai_api.json` 无需预先保留示例角色输入；不带 `-r` 时不增加角色节点。
布局模型从完整原文判断是否存在可见人物之间的明确动作，不依赖固定动作词组；有明确动作时，主动方的角色框使用
`source#hug`，被拥抱方使用 `target#hug`；若双方相互拥抱，则两边各用
`mutual#hug`。`#` 后只能是一项 Danbooru 动作 tag，不能接逗号分隔的多个
tag 或整句话；单人动作应留空。模型在同次直接写作中复核互动、方向、坐标和 tag 归属。主机丢弃格式错误的方向标签，并核对同动作 source/target 位于不同实例、mutual 至少两人且不与同实例的 directed 角色冲突；配对不完整时修复一次，仍失败停止。方向标签只进入各自角色框，不会放进全局提示词。此语法见
[NovelAI 多角色提示词文档](https://docs.novelai.net/en/image/multiplecharacters/#action-tags)；
官方也说明它并非每次都可靠。
规划结果还记录简短的整体构图分析和逐角色位置理由，供任务摘要排查；主机只
校验坐标范围；方向标签仅作可选项，不按“坐椅子”“跪在地上”等词套固定
坐标或改写模型所选位置。
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

- `width` / `height`：默认尺寸；宽和高都必须在 512–1756（含边界），且为 4 的倍数。用户本次明确指定尺寸时，非 `generation_whitelist_sender_ids` 白名单用户最多使用 1024×1024 像素（1MP）；比例快捷词也受此限制。未指定尺寸的默认配置不受此请求限制。
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
- `blocked_prompt_rules`：配置页的“屏蔽词组对规则”可视化组件可添加任意多条规则。每条有“左组词条”和“右组词条”两个可增删列表，分别填 A/B、C/D 等。任一规则中左组任意一个词条与右组任意一个词条同时出现，即拒绝生成；不同规则之间不会交叉配对。例如第一条为 A/B、第二条为 C/D，A+D 不会因这两条规则而命中。每条规则的两组都需要至少一个非空词条。同组内多个词条不要求同时出现。按子串匹配，不要求顺序或相邻，不区分英文字母大小写，也检查 `#` 后的手工尾缀。拒绝发生在额度扣减、模型调用和 ComfyUI 连接检查前；原样和多人模式同样生效。改图功能开启时也检查改图输入。上一版的固定 A/B 列表会在重载时迁移为一条组对规则。
- `blocked_prompt_combinations`：旧版 `&&` 组合仍在配置页保留并继续生效。任一旧规则或任一组对规则命中即拒绝；如需只使用组对规则，请清空旧规则列表。
- `prompt_builder_provider_id`：指定用于优化提示词的模型。留空时使用当前会话主模型。
- `prompt_builder_max_tokens`：普通提示词优化模型最大输出长度，同时约束两次 LLM；普通请求的第一次语义规划最多使用 900 tokens。`-r` 为容纳最多 22 个角色实例，将第一次语义规划的输出预算提高到 7200 tokens，专用第二阶段按 `max(1200, min(7200, max(prompt_builder_max_tokens, 1200 + 260 × 模型角色上限)))` 计算；正常没有第三次拆分或审查写作；方向配对结构错误可额外修复一次。
- `prompt_builder_max_content_tags`：LLM 内容段的硬上限，默认 65；不计算质量词、固定角色、画师组和画风。自然语言主题通常以 40–55 个内容 Tag 为目标，简单表情包或头像可以使用 30–45 个，复杂服装或构图约 60 个；已经是 Tag 串的输入不设最低数量。普通模式会在去除较多同义词后尝试一次按缺失画面槽位补全。
- `unspecified_wardrobe_policy`：按角色处理“没有实际服装基础”的策略。`scene_adaptive`（默认）把角色 default 作为软参考；中性场景更倾向沿用，涩气或强场景允许第二次 LLM 按情境替换或补全。也可固定为 `default_profile`（仍以软参考方式提供）或 `creative_fallback`（完全不加载 default）。
- `scene_adaptive_wardrobe_markers`：场景自适应的强触发词列表。默认包含泳池边、出浴、睡眠前后、比赛、运动/训练结束、健身、游泳结束和演出结束等；普通公园、在家、吃饭不会触发。
- `danbooru_semantic_system_prompt`：第一次 LLM 的意图抽取规则，只负责可见角色、明确服装或未知、服装修改和外貌修改。它不应生成 tag、candidate、lookup 或内部 ID；旧内置默认值会自动迁移，本地角色/衣柜匹配由主机随后完成。
- `prompt_builder_template`：普通第二次 LLM 的七段英文提示词模板；NAI `-r` 使用专用 JSON 任务书。

通常不需要一开始就改这两个模板。第一次 LLM 不调用 Codex 的 `danbooru-tags` skill，也不负责猜 tag；插件的本地 resolver 在意图抽取后匹配档案并验证证据。`unspecified_wardrobe_policy` 按角色生效：别的角色有服装证据不会关闭当前角色的缺省参考；当前角色已有普通衣物、命名套组或来源服装时则不加载 default。第二次 LLM 始终读取完整原文，并结合 LLM1 关系建议、数据库证据和禁止恢复项完成最终语义。

## 联网与 tag 查询

- `prompt_builder_web_search_enabled`：允许在需要时联网搜索。
- `prompt_builder_search_query_template`：搜索查询模板。
- `danbooru_core_tag_lookup_enabled`：是否启用旧的联网角色核心 tag 校正，默认关闭；关闭时仍保留 LLM 候选与本地 `danbooru-tags` 语义校验，也会继续使用已经持久化的服装档案。
- `danbooru_tag_base_urls`：优先使用的 Donmai tag API 地址。

联网搜索需要 AstrBot 全局 Tavily key。搜索失败会自动降级，不会中断生图。

## 服装词库页面

首屏读取完整词库与轻量翻译任务状态；完整批量翻译进度在展开面板或点击“刷新进度”后读取。目录按文件版本缓存，派生审查结果随翻译/复核断点或当前映射变化失效；重型进度计算不阻塞请求主循环。未保存编辑时刷新会提示先处理编辑；迟到读取不覆盖新修改，保存期间的新编辑也保留为待保存状态。词库仍完整保存，不把未加载分类当空数组提交。

新增“词汇候选与释义审核”，按需展开、每页 30 个概念，支持中文/tag/释义查找、字面相近候选及状态筛选。维护资料在 `third_party/llm_nai_toolbox/vocabulary_review_catalog.json`，人工决定另存插件数据目录的 `vocabulary_review_decisions.json`，带 revision 检查。资料与决定不是运行映射；采纳或修改只记录待验收候选，不直接调用模型或导入。正式词库仍为 `danbooru_term_mappings`。来源/MIT 说明见 `third_party/llm_nai_toolbox/`。新页面与接口须加载新插件代码并刷新页面，本轮未自动重载。

重载插件后，可以从 AstrBot WebUI 的插件详情页进入“服装词库”。页面提供三个可搜索、可增删改的列表：

- **服装档案**：按角色或服装来源保存经过帖子样本验证的可见服饰结构。
- **服装套组**：保存独立命名的完整服装套组及其 canonical Danbooru tag。
- **名词翻译**：把中文名词映射为 canonical Danbooru tag 参考；由模型决定采用、舍弃或自然语言表达，不做缺失 tag 补入。

三个列表默认每页显示 30 条，可切换为 10、50、100 或 200 条。列表上下均提供首页、上一页、下一页、末页和输入页码跳转（也可按 Enter）。各分类独立记住当前页和每页数量；搜索覆盖整个分类，先搜索／排序再分页。翻页保留未保存编辑，保存仍提交完整词库；删除末页最后一条会自动回到有效页，新增会清空搜索并定位到新条目。批量翻译断点预览也使用同样的分页控制，默认每页 30 条。

“名词翻译参与生图”开关对应 `danbooru_term_mappings_enabled`，默认开启。可在配置页或下方批量面板切换；通过面板切换会立即影响后续请求。关闭时普通优化和 NAI `-r` 都不读取 `danbooru_term_mappings` 来给模型提供参考，列表、翻译断点和批量管理仍保留。该开关也会停用手工添加的名词映射；角色、服装套组、作品别名和画师配置不受影响。

服装套组以“独立套组实体 + 变体”维护触发别名，每个实体指向一个 canonical tag。例如：

```text
Canonical Tag: haneoka_school_uniform
触发别名: 羽丘校服, haneoka school uniform
```

下划线形式 `haneoka_school_uniform` 只保存在 canonical tag 字段，不会重复写进别名列表；中文名和空格英文名会分别保存并参与匹配。不同套组实体允许指向同一个 canonical tag，不会仅因 tag 相同就强制合并；最终 hard tag 会去重，但各套组仍保留自己的别名、变体和角色绑定 anchor。手动配置的别名优先于冲突的自动学习别名。

页面保存使用 revision 检查，若生成期间词库已被其他请求更新，会要求刷新后重试，避免覆盖新数据。保存成功后配置与运行时缓存立即更新，不需要为每次词库编辑重载插件；只有安装或更新本页代码后需要先重载一次插件。
页面允许最多 50000 条名词翻译，支持后续扩展到数万条；浏览时仅渲染当前页条目。词库仍一次性读取，保存仍校验完整列表与 revision。角色衣柜和通用套组仍分别限制为 300、500 条。每条别名与 canonical tag 仍逐项校验，重复中文别名仍会拒绝保存。

配置页中的 `danbooru_named_outfit_mappings` 和 `danbooru_term_mappings` 仍可直接编辑。前者是“服装套组”的兼容配置入口，格式为 `别名1 | 别名2=canonical_tag`；后者没有角色归属：仅供第二阶段模型理解；模型未输出独立 canonical tag 时不自动补入。推荐日常使用“服装词库”页面，减少格式错误。完整的数据源职责、角色绑定和最终注入规则见 [角色与服装配置工作流](角色与服装配置工作流.md)。

批量初译使用 AstrBot 已配置的模型 provider：管理员运行 `/anm term_preview` 查看范围，随后运行 `/anm term_translate <DeepSeek provider ID>`。每次最多初译 50 个通用 tag，并让同一 provider 独立复核待审批次；结果断点保存在插件数据目录。旧的未审核初译可运行 `/anm term_verify <DeepSeek provider ID>`。`/anm term_import preview` 只预览数量，`/anm term_import apply` 才会把自动复核通过、且不与现有手工映射冲突的中文别名批量写入 `danbooru_term_mappings` 并立即更新运行中解析器。模型调用按对应 provider 的计费规则产生费用。

候选范围为本地 `anima-1.0.csv` 中帖子数至少 1000 的全部通用标签（category 0），按帖子数降序、同热度按 tag 名称排序，不设前 1000 个的总量上限。角色、作品、画师等其他分类以及低于门槛的标签不在本批范围内。扩展候选后继续沿用原翻译和复核断点，已处理的 tag 不会重复送译；更新代码后需重载插件，页面与命令会使用相同的完整候选范围。

也可在本页下方“批量名词翻译”面板完成同一流程。填入 AstrBot 的 DeepSeek provider ID，点击“翻译下一批并复核”；页面会轮询后台任务。统计卡显示累计翻译、累计模型复核、可导入别名、已在词库的别名和需人工复核的标签数；任务完成提示另列本轮新增数量与模型调用次数，已复核的断点不会重复送给模型。断点表可按状态筛选并分页查看。点击“导入可用别名”后，刷新页面即可在上方“名词翻译”列表继续编辑。导入前请先保存上方词库尚未保存的修改。页面安装或更新后先重载插件。

找问题条目时选择“需人工复核”：它包含未解决的精确映射冲突、格式问题、模型拒绝的别名，以及已复核但无中文译名的标签。“待模型复核”单独列出尚未完成自动复核的断点。同一标签的多个别名逐个显示状态，部分可导入、部分需人工复核时会同时出现在这两个筛选中。全部初译别名已在当前词库且指向相同 tag 的条目归入“已在词库（全部别名）”，不会混进人工复核；手动添加的相同映射也视为已解决，因此这里表示当前词库状态，不是历史导入记录。旧的子串包含提示不等于导入失败，不会单独触发人工复核。上方词库保存修改后点击“刷新进度”，分类会重新计算。导入结果中的“已有相同映射”与“冲突或格式问题未导入”分别计数。

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

`comic_dialogue_enabled` 位于“实验与附加功能”，默认关闭。开启后处理明确标记的对白／内心独白／旁白，以及带说话人的引号台词。先从用户原文复制台词，短台词优先在画面内排气泡，空间不足时在图片上方添加对白区；Anima 与 NAI 共用。原图保留在 ComfyUI 输出目录，排字版保存在 AstrBot `workspace/outputs/comic_dialogue/`，沿用 `storage_retention_days` 清理。若显式台词无法可靠提取或排字失败，本次不发送缺台词的原图。关闭开关或请求没有可识别台词时不改变现有提示词和回传流程。

图生图、放大和去背景仍在开发中。配置项保留给后续版本使用，不建议作为稳定功能依赖。

改图工作流会先用 `max_image_side` 限制输入图的基准最长边，再把基准宽高乘以
`upscale_factor`，并将结果写入 ComfyUI 的 `ImageScale`（“缩放图像”）节点。
例如基准尺寸为 `832x1216`、倍率为 `1.5` 时，改图尺寸为 `1248x1824`。

## 发送与权限

### 按会话发送前打码

插件详情页的“服装词库”页面内“图片打码设置”区域按 AstrBot 完整会话 ID（平台、群聊/私聊类型、会话 ID）分别保存开关。会话首次请求图片后会自动出现在页面，也可手工添加完整会话 ID。默认关闭。开启的会话在发送每张生成图片前，把该图片上传到 ComfyUI，运行插件目录中的 `autofilter2.json`，替换工作流 `LoadImage` 的图片输入，并把 `SaveImage` 的输出前缀临时换成 Windows 安全的唯一名称；其他节点和参数沿用文件中的值，原 `autofilter2.json` 不会被修改；聊天收到的是 `SaveImage` 输出。关闭时直接发送原图。打码工作流失败时不会发送原图，会提示发送失败。该功能依赖 ComfyUI 安装新工作流所需的 Impact Pack、Impact Subpack、Essentials 节点及 `bbox/censor_detect_v1.0_s.pt` 检测模型；嵌入的 Image Blur 子图由插件展开为 GLSLShader API 节点。

检测模型需要手工放入 **ComfyUI 的模型目录**：从
[deepghs/anime_censor_detection 的 `censor_detect_v1.0_s` 目录](https://huggingface.co/deepghs/anime_censor_detection/tree/main/censor_detect_v1.0_s)
下载其中的 `model.pt`，重命名为 `censor_detect_v1.0_s.pt`，放到
`ComfyUI/models/ultralytics/bbox/censor_detect_v1.0_s.pt`。工作流中的
`UltralyticsDetectorProvider` 选择的是 `bbox/censor_detect_v1.0_s.pt`；
文件名或目录不一致时，打码工作流无法加载该模型。

当前随附工作流的 `GrowMask` 遮罩扩张值为 `15`（此前为 `8`）；
它扩大被检测区域周围的打码遮罩。工作流参数在每次运行时从文件读取，
不会由会话开关另行覆盖。

会话开关保存在插件数据目录的 `autofilter_sessions.json`，页面保存后立即生效，不需重载插件。

- `send_result_to_chat`：是否把图片发回聊天。
- `max_send_images`：最多发送几张。
- `notify_drawing_and_at_sender`：同一个开关控制两项行为：接受生图请求后发送动态绘图进度；前方没有未完成申请时显示“正在绘画中”，否则显示“正在绘画中，前面还有X人”（X 不包括申请人自己）；图片完成后在群聊中 At 原申请人并告知今日剩余次数，私聊则在图片前直接显示剩余次数而不 At。不限额和白名单用户显示“今日剩余次数：不限”。额度用完时，群聊和私聊都会在生成前收到拒绝提示。
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

`debug_prompt_enabled` 会同时记录第一次语义规划 LLM 的任务书、system prompt、完整原始回复、解析后的 anchors、恢复前后的 `character_plans`，以及第二次提示词 LLM 的任务书、回复和最终 prompt。`-r` 请求以 `NAI -r direct writer` 记录专用第二阶段实际画布、位置规划任务书、模型原始回复，以及解析和后处理后的完整位置规划（构图分析、每个角色的位置理由、XY 坐标、角色提示词、被丢弃的互动方向 tag，以及从全局提示词删除的明确角色实例 tag）。解析失败时也会记录原始回复和错误。任务摘要另保存 `nai_character_plan_prompt`、`nai_character_plan_raw` 与 `nai_dropped_global_character_tags`；方向配对错误、初稿/修复稿及尝试次数分别记录在 `nai_interaction_initial_validation_errors` / `nai_interaction_retry_validation_errors`、`nai_character_plan_initial_raw` / `nai_character_plan_retry_raw` 和 `nai_character_plan_attempt_count` 中，便于复核规划结果。关闭该开关时不输出这些位置规划调试日志。

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

## Toolbox 词库导入与撤回记录

**当前状态（2026-10-13）：全量导入已按用户要求撤回，名词翻译恢复原有 1427 条。** 已核对原列表完全一致、其他配置未变，记录见 `toolbox_import/restored.json`。原因是来源别名存在口语覆盖不足、日语混写及语义不等价问题；导入数量和格式正确不能证明提示词质量改善。下述命令仅为工具用法与历史记录，不是重新全量导入的建议。后续见 [中文口语词库质量改进与新对话交接](中文口语词库质量改进与新对话交接.md)。

本地 `toolbox_terms.py` 直接读取 Toolbox Release 的 `resources/tagdb/tags_index_v2.json`，只使用 general 区的 `zhFull / zhShort / zhNick / zh`，不调用翻译模型。它把有效、含汉字、长度 2–80 的单值别名填入现有 `danbooru_term_mappings`；同词多 tag 跳过，人工映射优先，合并总数不得超过 50000。只向提示词模型发送本次命中的参考，不发送整库。

2026-10-13 历史导入（现已撤回）：原有 1427 行，新增 42264 行，745 个相同映射不重复添加，40 个人工冲突不覆盖，2808 个歧义别名跳过；当时合并 43691 行。4344 个非法/不适用来源项的具体字段、值和原因保存在清单。别名里的冒号、箭头不会先于正式 `=` 被解释为分隔符；页面、运行时、导入和批量翻译审查使用相同解析规则。

在插件目录用 AstrBot bundled Python 执行。下例不安装解释器或额外词库服务：

```powershell
$taskPython = Resolve-Path '..\..\..\.venv\Scripts\python.exe'
& $taskPython toolbox_terms.py --source 'D:\Users\69493\Desktop\llm-nai-toolbox-0.9.8-win-x64\resources\tagdb\tags_index_v2.json' --config '..\..\..\data\config\astrbot_plugin_anima_master_config.json' --report 'toolbox_import\下一次导入.json'
```

默认只预览；同一命令加 `--apply` 才写入。已应用清单不能覆盖，后续导入请换报告文件名。清单保存版本、SHA-256、原词库、新增、相同映射、冲突、歧义及无效原因，不备份 API key 等其他配置。写入前核对配置是否改变；文件以临时文件替换，其他配置项保留原值。写入前的待应用日志也可恢复，避免写配置后报告失败导致无法撤销。

本次实际恢复清单为 `toolbox_import/applied.json`。以下命令仅删除仍与导入值一致的新增别名，保留导入后用户改过的映射和其他配置，不恢复整份旧配置：

```powershell
& $taskPython toolbox_terms.py --config '..\..\..\data\config\astrbot_plugin_anima_master_config.json' --report 'toolbox_import\applied.json' --restore
```

这个离线入口修改配置文件，须重载插件并刷新词库页面后生效；不要让旧页面的未保存内容覆盖新词库。现有页面或 `/anm term_import` 的批量模型翻译导入仍使用自己的断点与热更新入口，未被替换。本次代码没有重启运行中的 AstrBot。

Toolbox 作者信息、原始 README/package 许可声明和 MIT 许可文本保存在 `third_party/llm_nai_toolbox/`。提供的源代码与 Release 没有独立作者版权/许可全文文件，因此未虚构版权年份或持有人；若上游补发具体声明，后续更新须一并保留。Electron 许可不能替代作者 MIT 声明。
