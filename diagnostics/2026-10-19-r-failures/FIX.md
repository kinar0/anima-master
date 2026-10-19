# 提交前失败修复记录

本次改动是在已有未提交工作上进行的局部修复，未重置其他修改、未提交 Git、未修改运行配置或重载 AstrBot，也未提交任何 ComfyUI/NAI 图片请求。新代码在重载插件后用于后续请求，不自动重发历史任务。

## 实现与契约变化

1. `prompt_pipeline.py` 的逐实例 LLM2 校验统一了 JSON/必填字段/坐标与身份/文字的恢复入口。初稿结构异常会作为具体错误发给原有一次修复；修复仍不合法则停止，保留具体错误码。每个请求至多一次修复，不能结构修完再获得一次身份或文字重试。方向标签降级不占预算。首次可解析布局即使来自修复稿，也不能跳过 donor 语义审查。
2. `nai_confirmed_identity_anchors` 只接受已确认 canonical 绑定且类别为角色的原文 anchor，并与既有档案别名共同检查歧义。名单和审核的 canonical 绑定都使用这些证据；无需先有外貌或衣柜档案，候选拼写和普通分类服装仍不能成为身份权威。
3. 对确认名单之外的额外未知审核行，只保存 `unconfirmed_identity_review` 诊断，不授予人物、服装、外貌或缺人解释权限。已确认人物缺失、已知来源升级为可见、矛盾 visibility、歧义别名、必要的原文依据继续校验。未知审核与已返回同名角色框矛盾时也仍拒绝。
4. `nai_text.py` 的文字 schema、初稿规则与修复规则一致说明：用户没有命名画格时 `panel=""`，模型设计的格子由实际实例 prompt、placement 和独立 character_index 表达；非空画格仍须来自该文字的原文上下文。不改变原文文字、出现 id、台词归属或命名画格的跨格检查。
5. 最终逐实例构建失败设置 `llm_failed=true`，使现有任务记录器正确写出 `llm_ok=false`。初稿错误被成功修复时不误标最终失败。

新增诊断键为 `nai_structure_initial_validation_errors / nai_structure_retry_validation_errors`。原始回复、修复任务书、最终采用轮次继续沿用已有 debug 记录。

## 提示词职责对照

| 原有职责 | 本次处理 |
| --- | --- |
| 七类画面语义、英文视觉 tags 与中文完整句子 | 原写作与同次复核职责保留；没有精简或替换任务书。 |
| 可见实例、人数、canonical、来源不自动出场 | 保留，补足可靠查询确认身份的名单来源。 |
| 逐角色衣柜、显式删改、稳定外貌与可见性 | 保留，不补衣柜/外貌、不从额外审核授予权限。 |
| 镜头、连续坐标、整组接触与 tag 身体所有者 | 保留；结构修复只要求模型补回合法坐标，不由主机猜测。 |
| 方向标签配对、降级与普通动作去重 | 保留，不新增硬拒绝或占用文字修复预算。 |
| 文字原文、出现 id、按画格实例索引分配 | 保留，明确区分用户命名画格与 writer 自行分镜。 |
| 完整 system/user/修复任务书继承 | 保留，修复任务书增补精确 schema 与未命名画格规则。 |
| raw、关闭优化、手工 `#` 尾缀、普通七字段、ComfyUI 提交与交付 | 处理边界保持。 |

## 验证结果

新增 `tests/test_nai_submission_failures.py`，覆盖六类结构缺陷成功恢复/仍失败、共用预算、首次可用 donor 审查、无档案但查询确认身份、未确认候选与普通分类服装反例、别名冲突、未知审核的无权限诊断、缺人与矛盾分类、未命名画格的逐实例文字恢复。已有两个测试的失败路径预期同步为“一次修复后仍停止”，没有把名词缺项变成重试条件。

本次新增缺陷回归最终单独运行结果为 **28 passed**，修改文件的语法检查通过。最后一次扩大本地回归为 **453 passed，3 failed**。三项未通过的检查来自同时变化的共享写作规则和外貌合并，本次未覆盖或回退它们：

- `test_restored_scene_responsibilities_stay_inside_the_direct_nai_writer`：另一处共享规则包含 `seven-field` 文本，与原有“任务书不得含该字面串”的断言冲突。
- `test_direct_writer_reviews_composition_tag_ownership_and_directions_in_one_call`：同一共享规则与字面断言冲突。
- `test_identity_merge_preserves_leading_is_with_attributes`：外貌合并结果不再保留该用例期待的 sidelocks / two side up buns；本次未修改相应合并函数。

修改初期同组 NAI 定向检查曾为 257 passed；随后扩大检查时工作区的 `prompt_templates.py / nai_character_mode.py / prompt_visibility.py` 等出现了其他改动。上述三项不能隐藏为“全部通过”，也没有为消除失败而回退其他工作。

历史离线重放使用 `audit.py --expect-fixed`，写入独立的 `fixed-evidence.json`，保留原 `evidence.json`。原先四条身份误拦在保存的真实 LLM2 修复稿上全部消失；两个原文画格错误仍按契约拒绝，未通过主机放宽原文校验来消除。原先五条结构错误仍被严格解析器拒绝，其恢复由新增的一次修复入口负责，使用中性测试覆盖。

## 经用户授权的真实模型回归

目的地：当前配置的 `https://api.deepseek.com/v1`，`deepseek-v4-flash`。用户明确允许发送两条中性漫画测试、任务书与相关角色资料。没有发送当天失败案例的原始请求，也没有调用图片生成。

`tests/live/test_nai_submission_recovery_live.py`：**2 passed**，实际耗时 21.44 秒。

| 场景 | 原始回复与最终结果 |
| --- | --- |
| 未命名的上下两格、白衬衫角色先挥手后坐下 | 初稿仍产生无法定位的画格名，触发一次文字修复；修复稿两条 panel 均为空，两个实例索引不同，各自最终尾段分别是“你好”和“再见”。第一次模型失败也保留在记录中。 |
| 明确命名“左上A格/右下B格”、黑色外套角色 | 一次成功，两条 panel 保留用户命名，实例和台词分别绑定。 |

原始模型回复、完整实际任务书、最终全局与角色框保存在 `live-fixed/source_panels_False.json` 和 `live-fixed/source_panels_True.json`。人工复核确认实际 raw/final 都保留英文衣物/姿态等 tags 与中文成句，台词没有在最终全局重复。模型仍可能在全局自然语言中重复个体衣物等已有语义问题；本次通过的是这些案例的结构、原文与实例传递，并不代表所有画面语义均已验证。

上述两次模型成功不保证所有初稿遵从、不保证结构修复必然成功，也没有验收生成图的字形、分格或肢体。无需为本次代码修复重启 ComfyUI；AstrBot 插件仍需重载后生效。
