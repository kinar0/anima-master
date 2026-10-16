# NAI -r 位置控制验证（2026-10-15）

改动：在既有 Composition、同次复核和 writer system 中追加空间推演与全部动作同时可行的要求。保留原双语、实例、身份、衣柜、可见性、身体所有权和方向规则，不增加模型阶段或输出字段。修复请求继承同一提示词。

配套 ComfyUI 节点：0–1 范围不变，UI 步长 0.001、关闭 UI 取整，后端取消 `round(..., 1)`；高精度输入直接进入正负 caption 的 centers。原有 22 槽与模型人数限制保留。

## 自动检查

- AstrBot：`test_nai_workflow.py`、`test_prompt_strategy.py`、`test_custom_workflow.py` 共 269 项通过。
- ComfyUI 扩展：当前工作区全量 unittest 共 51 项通过，包含其他任务同时维护的转换测试；本次精度回归覆盖 22 槽、正负 caption、V4.5/V5 payload、近邻/重叠中心、边界与越界裁剪，四位小数保持。
- 两仓库 diff 空白检查通过。

## 真实模型与人工核对

使用 AstrBot 当前 provider；保存实际 system/user 任务书、全部原始回复、摘要和最终字符串。未提交 ComfyUI、未调用 NAI 出图。

| 场景 | 最后一次空间检查 | 核对内容 |
| --- | --- | --- |
| `nai_spatial_joint_reach.json` | 通过 | 1216×832 横图；同一人物同时扶肩、握手；双方接触和手/肩所有权进入实际文本；左右位置透传。 |
| `nai_spatial_seated_support.json` | 通过 | 832×1216 侧视全身；坐姿/跪姿高低；握手与托肘均进入全局及逐人文本；坐标原值保留。 |
| `nai_spatial_separate_panels.json` | 通过 | 左右独立画格及留白；同身份不同服装、站/坐；没有互动标签或跨格接触；整幅画布坐标 0.25/0.75。 |
| `nai_bilingual_repeated_identity_recolor.json` | 最后一次失败，前次通过 | 三实例、多点扶肩、黑/白外套。最后一次模型把不连续的局部与共享原文拼成 `source_text`，被现有 `nai_character_plan_ungrounded_instance_source` 拒绝；没有放宽权限校验或猜测归属。失败原文另存 `_failed.json`。 |

加强实际文本复核前的记录保存在 `before-final-review/`：模型曾在分析与角色框写出动作、却在全局文本遗漏动作。因此追加“全部动作/接触必须进入实际全局与相关实例 prose”，并增加 raw/final 两端的动作检查。最后一轮三组空间检查通过，不代表其他写作契约或空间语义均稳定。

人工复核还发现模型输出仍有波动：最后一次双重接触案例的全局缺少英文镜头/场景 tags，分格案例同时输出 upper_body/full_body，以及全局中文复述个别衣物。原指令仍明确禁止这些行为，本次没有把它们改成可选，也没有增加硬补词或额外语义重试。故本次结论仅为：空间要求已加强、连续坐标的本地传输损失已修复；不能宣称真实模型所有原有语义契约完全通过。

尚未验证 NAI 成图的臂展、接触、遮挡和像素位置；坐标仍是构图引导。[官方文档](https://docs.novelai.net/en/image/multiplecharacters/)说明 V4/V4.5 自定义位置受 5×5 网格限制，传输精度不会突破模型本身限制。

生效：重载 AstrBot 插件；重启 ComfyUI 并刷新节点界面。验证期间没有重启现有服务、提交生图或修改运行配置。工作区中并行出现的 NAI 文字渲染及转换器改动保留，不属于本次位置控制改动。
