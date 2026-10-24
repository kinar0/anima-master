# 后藤独首次学习差异排查（2026-10-24）

本页先保留历史任务的只读证据；用户明确要求回归后，又执行了真实模型对照、本地词典验证、独立档案副本中的真实帖子采样，并修复了候选补查。没有修改运行档案或配置，没有提交图片或重载插件。时间均为北京时间。

## 真实回归与修复结果

已复现同类失败，确定候选补查存在两个相接的缺陷：

1. 旧任务书要求 `a work disambiguation suffix when known`，并用 `name_(work)` 作输出示例。模型因此可能把作品名强行拼进本来没有后缀的角色 canonical，返回 `gotoh_hitori_(bocchi_the_rock!)`，而本地实际 canonical 是 `gotoh_hitori`。
2. `_generate_character_candidates_with_llm()` 使用的候选字符白名单不接受 `!`，尽管语义锚点、本地词典及别名允许它。唯一候选带 `!` 时，在本地查询之前就被清空，正好形成历史任务中的 `candidates=[]` 表现。真实别名 `bocchi_(bocchi_the_rock!)` 也会被这个错误过滤拦截。

使用当前安装的真实 provider/adapter，捕获的实际 HTTP JSON 确认关闭思考和 350 token 预算送达。修复前，单人三次均得到有效候选；双人三次中有两次只给上述添加后缀的名字，返回 `finish_reason=stop`，解析结果为空。因此本次复现不是超时、拒答、JSON 损坏或 token 截断，而是具体的候选名生成与字符校验问题。原始请求与回复见 `live-current-baseline/calls.json`、`wire.json`。

历史参数透传方式也单独对照了两轮单人/双人，共四次，均成功取得候选，见 `live-network/` 的 `historical/*`。该方式实际未送达关闭思考/token 预算。**这说明不能把本次非思考模式下的复现伪装成已恢复 20:22 的历史原始回包，也不能断言当时必然走了同一条失败原因。** 历史证据缺口仍在；已确认的是能造成同样症状的具体缺陷及修复效果。

修复保持原调用链和权限：

- 作品信息独立保留，不因知道作品就编造角色后缀；真实 canonical 自带的消歧/persona 后缀保持。明确版本要求每一个候选都对应该版本，不能再额外给基础角色作替代。
- 接受已有语义锚点支持的合法 `!`、`:` 等标点。不会自动去掉后缀猜基础角色，所有候选继续交给本地词典验证，模糊结果仍不取得身份权限。
- 多人补查的 `source_name` 必须等于当前 Target name，避免返回另一个同样出现在原文的人物后被错误绑定。
- 请求级摘要新增 `semantic_character_candidate_attempts`：逐目标保存状态、接受/拒绝候选及异常类型；debug 时保存实际补查任务书与原始回复。不是共享的实例变量，不因并发覆盖其他请求。

### 验收证据

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 修复前原请求对照，当前透传方式 | 单人 3/3 有效，双人 1/3 有效，失败两次均为带 `!` 的错误后缀唯一候选 | `live-current-baseline/` |
| 修正后原请求重复 | 单人 5/5、双人 5/5 均由真实模型返回并经本地词典确认 `gotoh_hitori` | `live-fixed/` |
| 最终版本再次核对原请求 | 单人 2/2、双人 2/2 成功；与前五轮合计各 7/7 | `live-final-originals/` |
| 最终版本扩展 | 姓名反序、其他人物、明确 SV 版本、S:P 标点名称各两次均确认正确；匿名人物两次均不生成候选 | `live-fixed-extended-final/` |
| 用户指定的真实作品后缀 | Oblivionis、Mortis、借衣来源各两次，6/6 最终确认完整 `oblivionis_(bang_dream!)` / `mortis_(bang_dream!)`。模型偶尔只给 `oblivionis`，也由本地精确别名解析回完整 canonical；真实后缀没有删除 | `live-real-suffixes/` |
| 真正无后藤独档案的双人/单人副本 | 各自冷启动，真实候选补查 → 本地确认 → 网络帖子采样 → 新建独立档案；均 fetched=100、accepted=43、focused=24、rejected=57，无采样错误 | `cold-cache-learning/report.json` |
| 运行档案保护 | 两种冷启动学习结束后，运行档案 SHA-256 与测试前一致 | `cold-cache-learning/production-cache-unchanged.txt` |
| 本地回归 | 六个相关测试文件共 398 passed | 下列命令 |

首版扩展测试 `live-fixed-extended/` 曾发现独立版本补查把 `kasane_teto` 与 `kasane_teto_(sv)` 同时列为候选，直接查询会优先确认基础角色。该失败记录保留；最终任务书补上“所有候选均须为同一明确版本，不把基础角色作额外替代”，最终复测两次均只选 `kasane_teto_(sv)`。正常完整链路还保留原有本地/括号别名版本权限，未通过去后缀兜底。

`live-network/` 另保存了一次带日文回复的输出编码中断，该原始回复已先写盘，不能把脚本终端 GBK 错误当作模型失败。此前默认沙箱网络连接失败记录在 `live-baseline/`，未计入模型成功/失败比率。

本地测试范围：`test_character_candidate_lookup`、`test_prompt_strategy`、`test_prompt_direct_writer`、`test_danbooru_semantic`、`test_llm_diagnostics`、`test_llm_request_controls`。使用实例 Python 3.12、core 的 PYTHONPATH、`-p no:cacheprovider` 和任务内独立 TEMP/TMP/basetemp；398 项通过，仅有现有主机 audioop 弃用警告。`git diff --check` 通过。

随后为用户补充的真实后缀新增正例，`test_character_candidate_lookup` 与 `test_outfit_source_anchors` 合计 20 项通过（与前次测试有重叠，不将两次计数相加作独立用例数）。角色与 outfit_source 两种角色的本地查询均保留真实完整后缀，原有普通/NAI 借衣来源 tag 保留回归也通过。修复没有把“作品后缀”全局禁止，只纠正模型强加不存在的后缀及合法标点被错误过滤的问题。

### 候选任务书职责对照

| 原要求 | 最终对应 |
| --- | --- |
| 已有作品角色、最多六个候选、原语言名字 | 保留 |
| canonical 小写 ASCII 与下划线 | 保留，明确兼容 canonical 合法标点 |
| 作品单独推断与记录 | 保留；修正“知道作品就追加角色后缀”的错误要求 |
| 原文 source_name、没有命名人物不编造 | 保留，并强化指定目标时 source_name 必须等于该目标 |
| 地方化职业/称号可能是官方角色名，不能仅字面翻译 | 完整保留 |
| Target、作品 hints、完整原文、初始失败候选 | 完整保留 |
| JSON 三字段、独立 system、350 token、关闭思考 | 保留；候选示例不再强制 `name_(work)` 形态 |
| 候选只作查询线索 | 保留，仍精确验证，不改写 LLM1、LLM2、衣柜/外貌/出场权限 |

修复需要重载插件后用于新请求。真实回归只覆盖身份候选、查询与档案学习，未调用画面 writer 或提交 ComfyUI/NAI 图片，不能据此宣称成图外貌或服装已验收。

## 已确认结论

失败位于新角色 canonical 身份候选建立阶段，早于帖子采样。双人请求的 LLM1 没有遗漏后藤独；主机保留了她的 target anchor，但该 anchor 最终没有任何英文查询候选，也没有确认身份。因此没有建立她的角色档案。单人请求取得候选并确认 `gotoh_hitori` 后才成功学习。

不能据此断言程序禁止双人请求学习，或“只穿内衣”禁止学习默认服装。当前 `resolve_semantic_anchors()` 对已确认的每个目标独立取得角色采样，不以本次衣服是否为默认服装作为该采样循环的条件。生成请求中的人数也不是帖子采样的人数；单人归属筛选针对检索到的帖子。

## 历史证据

原始任务位于 `core/data/plugin_data/astrbot_plugin_anima_master/tasks/`，这里只摘录身份和学习元数据。

| 时间 | 任务 ID | LLM1 角色 | 后藤独最终 anchor candidates | 后藤独身份结果 |
| --- | --- | --- | --- | --- |
| 20:22:02 | `89a2f68d9dd142eb829c671f754d3157` | 千早爱音、后藤独，两条均存在，规划一次通过 | `[]` | 未确认；`danbooru_semantic_missing` 含后藤独；实际 NAI 角色框仍叫“后藤独” |
| 20:24:49 | `d7c81ee4cbc04d4d9ee6f18ca7df2f61` | 后藤独，规划一次通过 | `gotoh_hitori`, `gotou_hitori`, `hitori_gotoh` | `confirmed_character_bindings` 为后藤独 → `gotoh_hitori`；实际角色框使用 canonical |
| 20:30:25 | `9afccbb2d28d4aeca30f74d8283918ff` | 千早爱音、后藤独 | `gotoh_hitori` | 双人请求已能确认并使用后藤独 |

20:22 的已确认绑定、稳定外貌和来源服装资料都只有千早爱音。总体 `danbooru_semantic_status=resolved` 不能代表所有角色已解析：查询结果允许部分成功，仍要逐项检查 missing 和 bindings。

当前 `danbooru_outfit_profiles.json` 的“后藤独”记录：

- canonical/source：`gotoh_hitori`；别名“后藤独”和 `gotoh_hitori`。
- `evidence.created_at = evidence.updated_at = 1792844712.869464`，即 2026-10-24 20:25:12.869464 +08:00，位于单人请求之后。
- 查询 `gotoh_hitori`；抓取 100 条，接受 43 条，最终服装聚类 24 条，拒绝 57 条；`request_errors=[]`。
- 稳定外貌 `pink_hair, long_hair, blue_eyes`；保存六条默认服装组件。

该证据支持单人请求成功完成学习。20:30 的 canonical 候选与新档案一致，结合实现可解释为已保存别名提供了本地身份依据；不再需要首次中文名补查成功才能使用。

## 实现路径与证据缺口

1. `prompt_pipeline.py` 解析 LLM1 `characters`，主机生成 target anchors。中文名字没有本地身份或明确英文别名时，不会由 LLM1 直接猜 tag。
2. `DanbooruResolver.prefer_cached_character_anchors()` 用已保存角色资料的别名提供 canonical 候选。20:22 的爱音已有此依据，后藤独当次没有取得候选。
3. `danbooru_semantic.lookup_semantic_anchors()` 按 candidates 构造查询；原文兜底仅接受指定 ASCII 名称形态。中文“后藤独”加空 candidates 不能直接产生英文查询。
4. `PromptPipeline._refine_unresolved_semantic_characters()` 为未确认目标调用 `_generate_character_candidates_with_llm()`，取得有界 canonical 猜测，再精确复查。它逐目标处理，已有一人成功不会按设计关闭其他人的补查。
5. 候选解析可能因非法 JSON、非对象、缺少/无法在原文定位的 `source_name`、缺少数组 `tag_candidates`、候选字符不合规而返回空；调用异常同样被捕获并留下空候选。外围规划/查询异常也可能提前结束这段处理。
6. `resolve_semantic_anchors()` 只为已确认绑定的目标进入 `fetch_variant_outfit_profile()` 和 `remember_outfit_summary()`。20:22 的后藤独没有达到这个入口。NAI 第二阶段不会因自行认识该名字而重新执行角色学习，历史结果也未将其姓名归一到 canonical。

历史任务保存了 LLM1 和 NAI writer 原始结果，但没有保存候选补查调用的原始回复、解析拒绝原因或逐目标调用结果。当前 `_generate_character_candidates_with_llm()` 本身也没有记录该原文及拒绝原因。文件日志当时配置为关闭，本次没有取得可补足该阶段的历史控制台记录。因此目前能确认“有效 canonical 候选没有建立”，不能进一步确定当次究竟是模型拒答/空答、JSON/来源名校验失败、调用异常，还是外围异常；不能把其中任一可能写成已证实根因。

以上是初次只读检查时的边界。随后真实回归与修复已落实逐目标调用/解析状态、合法标点与目标对应校验，具体证据和职责对照见本页“真实回归与修复结果”。未给后藤独写入任何生产代码特判，未绕过本地精确验证。

## 验证范围

既有定向回归 `tests/test_prompt_strategy.py::test_unresolved_localized_character_is_refined_and_exactly_rechecked` 用本地替身验证“未解析中文角色 → 候选补查 → 精确复查”的一般流程。它不能还原本次未保存的真实候选回复，也不证明真实模型或成图表现。
