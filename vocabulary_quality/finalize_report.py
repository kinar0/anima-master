"""Report frozen results and withdrawn proposals without retuning the holdout."""
from collections import Counter
import json
from pathlib import Path
from run_evaluation import ROOT, PLUGIN, read, write, sha


def main():
    baseline=read(ROOT/"baseline.json")
    comparison=read(ROOT/"comparison.json")
    candidates=read(ROOT/"candidates.json")
    reviews=read(ROOT/"candidate_review.json")
    audit=read(ROOT/"original_audit.json")
    samples={r["id"]:r for r in (json.loads(x) for x in (ROOT/"evaluation.jsonl").read_text(encoding="utf-8").splitlines())}
    failed={tag for r in comparison["delta"] if r["split"]=="acceptance" for tag in r["new_wrong"]}
    dispositions=[]
    for r in reviews:
        tag=r["entry"].split("=",1)[1]
        d=dict(r)
        if r["decision"]=="retain_for_comparison":
            d["final_disposition"]="withdrawn_acceptance_error" if tag in failed else "provisional_requires_fresh_holdout"
        else:
            d["final_disposition"]=r["decision"]
        dispositions.append(d)
    provisional=[r["entry"] for r in dispositions if r["final_disposition"]=="provisional_requires_fresh_holdout"]
    write("final_dispositions.json",{
        "applied":False, "batch_gate":"failed; no publishable validated batch this run",
        "withdrawn_tags":sorted(failed),"aliases":dispositions,
        "provisional_rows":provisional,
        "semantic_deferred_concepts":[c["canonical_tag"] for c in candidates["concepts"] if c["review_decision"]!="proposed"],
        "holdout_policy":"Do not rerun the reduced set on this acceptance set to claim validation; use a fresh holdout next task"})
    manifest=read(ROOT/"baseline_manifest.json")
    config=PLUGIN/"../../../data/config/astrbot_plugin_anima_master_config.json"
    changed=[name for name,old in manifest["existing_file_hashes"].items() if not (PLUGIN/name).is_file() or sha(PLUGIN/name)!=old]
    integrity={"config_entire_file_unchanged":sha(config)==manifest["config_sha256"],
               "original_mappings_snapshot_unchanged":sha(ROOT/"original_mappings.json")==manifest["mappings_sha256"],
               "evaluation_unchanged":sha(ROOT/"evaluation.jsonl")==read(ROOT/"evaluation_lock.json")["sha256"],
               "existing_files_checked":len(manifest["existing_file_hashes"]),"existing_files_changed":changed,
               "candidate_applied":False,"runtime_sync":"unknown; no reload/restart/UI operation"}
    integrity["evaluated_matcher_unchanged"]=all(sha(PLUGIN/name)==manifest["existing_file_hashes"][name]
        for name in ("danbooru_resolver.py","mapping_syntax.py","danbooru_semantic.py","danbooru_tags.py"))
    write("integrity.json",integrity)
    performance=read(ROOT/"performance.json")
    lines=["# 中文口语词库阶段 1–3 离线差异报告", "",
        "日期：2026-10-13。阶段 1–3 的审查材料与离线对比已完成；首批候选未通过发布门槛，未应用。无模型/联网/生图调用，无运行配置写入，无页面保存或重启。", "",
        "## 基线与保留边界", "",
        f"- 磁盘名词映射为 **1427 行**，与 `toolbox_import/applied.json` 的 `previous_rows` 逐项完全一致。Git HEAD 为 `{manifest['git_head']}`，原有用户修改保留。只导出名词映射，完整配置仅计算散列，不复制密钥。",
        "- 本任务只新增 `vocabulary_quality/` 审查文件，没有改写既有普通七字段、Nltags、背景由模型判断、名词只作参考或 NAI `-r` 流程文件。期间其他工作区修改见文末完整性记录，不回退或覆盖。",
        "- 当前代码核对：命令分发 → generation_task 拆手工尾缀/旁路 → prompt_pipeline → LLM1 意图 → resolver 角色/本地词汇证据 → 衣柜与外貌权限 → 普通 writer 或 `-r` 专用实例 JSON → 同权限后处理/最终组装 → 工作流提交、history、download → 聊天回传。此次仅在 resolver 的 `matched_term_mappings_for_prompt` 上离线运行；从未进入提交和收费调用。",
        "- 词汇命中来自原文；最长重叠、独立位置、缓存随配置快照失效和紧贴否定过滤沿用现实现。名词匹配不证明人物归属；场景提及衣物和别人的属性可作词义参考，不能授予目标人物衣柜权限。raw/关闭优化及 `#` 尾缀不纳入词汇匹配评估，旧独立多人/Turbo 也未改动。",
        "- `事后=aftersex` 已按用户本轮明确的词库语境认可保留。未把日常语言多义本身当成必删理由。", "",
        "## 评估口径", "",
        "132 条合成请求：开发集 96，预先冻结的验收集 36（约 27%）。没有读取聊天记录；实际失败请求样本不足，不能代表真实使用频率。每条保存来源、原文、适合参考、禁止参考、允许弃权及语义依据。预期是词汇参考，不是最终提示词必含 tag。",
        "原文与分集 SHA-256 固定在 evaluation_lock.json；候选从开发集缺口编写，不用验收结果添词。semantic_adjudication.json 单独解释未预列的原库父概念/误命中，不改原文、不调整候选。输出 0 项未判定命中；人工语义判断仍可讨论。",
        "表中“目标参考”是样本明确评分的概念；“允许父概念”是额外合理但较粗的参考。漏匹配只统计需要覆盖的正例，允许不命中的反例不计漏词。主动弃权计为反例完全不命中；局部歧义不命中但有其他有效词的情况保留在逐样本明细中。", "",
        "| 集合 / 库 | 目标正确参考 | 允许父概念 | 错误参考 | 仍缺目标参考 | 反例完全弃权 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split,label in [("development","开发"),("acceptance","验收")]:
        for result,name in [(baseline,"原库"),(comparison,"原库 + 33 行候选")]:
            c=result["counts"][split]
            lines.append(f"| {label} / {name} | {c['focus_correct_references']} | {c['permissible_parent_references']} | {c['wrong_references']} | {c['missing_references']} | {c['acceptable_abstentions']} |")
    lines += ["", "## 原库审查", "",
        f"全 1427 行完成格式、本地目录与词面对照初筛：{audit['unique_alias_count']} 个有效独立别名、{audit['unique_tag_count']} 个 canonical 值；无解析无效行或重复别名。标记 **54 条**语义/口语/条件问题，**1 条**用户确认语境，**1372 条**初筛未发现显著问题。这 1372 条没有得到全语境或逐项 Wiki 正确认证。完整逐行记录见 original_audit.json，问题表见 original_issues.md。",
        f"本地存在性精确批查发现 **5 行** canonical 未命中：{', '.join('`'+r['canonical_tag']+'`' for r in audit['local_missing_rows'])}。缺失只说明当前 Anima 目录覆盖，不表示 NAI 不支持或应删。未按假名字符删词，假名仅作标记。",
        f"记录 {len(audit['substring_overlaps'])} 个异 tag 的长短包含对；包含本身不是错误。需要区分白衬衫/衬衫的合理父子关系、露出微笑/露出的误导，以及否定长词后剩余短词仍触发的问题。",
        "明确不等价例：鸭舌帽 → mob_cap；条件丢失例：短裙 → skirt；上下文风险例：白裙子 → white_skirt、背心 → tank_top、说出心声 → spoken_heart。原值均未删改。", "",
        "## 候选与对比结论", "",
        "人工制作 48 个概念的候选记录，每条有别名数组、中文释义、语义依据、来源、原映射关系、本地与 NAI 覆盖说明、正例、邻近反例及审查结论。48 个 canonical 均在本地目录 exact_tag 命中；目录存在不证明中文等价或图片效果。未复制 Toolbox 词条，现有来源/MIT 材料保留。",
        "14 个概念在语义/范围/冗余审查时先暂缓，其余别名在开发集分别评估。坐着、下着雨因开发反例造成新错误暂缓；未测得增益的自然变体继续候选待审，不凑别名数；露指手套为原库已有同值，跳过。无不同值精确冲突覆盖。33 行进入合并对比，原有前 1427 行逐项保留、重复合并幂等。",
    ]
    for split,label in [("development","开发集"),("acceptance","独立验收集")]:
        delta=[r for r in comparison["delta"] if r["split"]==split]
        lines.append(f"- {label}：新增目标正确参考 **{sum(len(r['new_focus_correct']) for r in delta)}** 个，新增明确错误参考 **{sum(len(r['new_wrong']) for r in delta)}** 个；减少原库错误 **{sum(len(r['removed_wrong']) for r in delta)}** 个。新增目标正确参考出现在 {sum(bool(r['new_focus_correct']) for r in delta)} 条请求中。减少旧错误有些仅源于最长别名遮住短词，不证明机制已修复。")
    lines += ["", "验收目标参考从 8 增至 35，净增 27；逐概念新增 29，同时 2 个旧的宽泛目标参考（book / cup）被持物长别名覆盖。新增、丢失和净变化分别记录，不能把新增 29 当成净增或整条请求成功数。"]
    lines += ["", "验收新增错误如下，归因于本次候选：", "", "| 样本 | 原文 | 新误导参考 | 根因 |", "| --- | --- | --- | --- |"]
    for r in comparison["delta"]:
        if r["split"]=="acceptance" and r["new_wrong"]:
            reason="否定与词条之间存在动机/动作文字；现有紧贴窗口覆盖不到" if r["id"] in ("A25","A26") else "引述/比喻不等于可见场景；字面匹配缺乏语境边界"
            lines.append(f"| {r['id']} | {samples[r['id']]['text']} | {', '.join(r['new_wrong'])} | {reason} |")
    lines += ["", "**因此 33 行候选批次验收失败，不提供可直接发布的验收通过清单。** 球鞋、半指手套、教室、公园 4 项从发布候选中撤回。其余 29 行仅保留为下一轮待审小批次（final_dispositions.json），没有再用同一验收集重新评估它们并宣称通过；下一次需要新的独立验收样本。候选维护记录中的语义等价判断与实际匹配可发布性分开。",
        "原库已有错误单独保留在 baseline.json：例如“趴”字面引用、针织背心、露出微笑、否定半指手套后短词手套漏入、裙子形制歧义与认真行为误作严肃表情；不把这些留存错误算成新候选成功。",
        "仍缺失详见 comparison.json 的 remaining_missing。摸头/亲吻的语境、仰卧条件、白裙子和黑丝形制等继续允许弃权。未列出的别名不会自动因为词义接近被匹配；本轮未添加模糊匹配或完整失败句。",
        "匹配规则未改：否定作用域、引述/非视觉语境属于一般机制问题，需要独立设计与新反例；不能用几个词特判来证明普遍解决。本轮先完成定位并暂缓相关候选，避免把未经过机制验证的数据写入运行库。", "",
        "## 成本与未验证项", "",
        f"同一 132 条短请求各跑 7 轮：原库首次冷匹配中位 {performance['baseline']['cold_median_ms']:.2f} ms，合并 {performance['with_candidates']['cold_median_ms']:.2f} ms；整集暖匹配中位 {performance['baseline']['warm_median_ms']:.2f} ms → {performance['with_candidates']['warm_median_ms']:.2f} ms。全部轮次见 performance.json；这是本机匹配耗时，不是服务端/大库/生图性能保证。",
        "当前模型本来可能理解这些词；无真实模型输出对比，不能推出提示词或图片质量提升。NAI 支持、在线定义、真实用户分布、页面当前编辑值和运行中词库均未验证。没有重载，不能宣称运行实例已同步。", "",
        "## 文件与下一步", "",
        "- evaluation.jsonl / evaluation_lock.json：冻结评估及分集。",
        "- original_mappings.json / baseline_manifest.json：仅词汇快照与完整性证据。",
        "- original_audit.json / original_issues.md：全表初筛、问题与重叠记录。",
        "- candidates.json / candidate_review.json / final_dispositions.json：语义候选、开发门控及撤回结果。",
        "- baseline.json / comparison.json / tag_validation.json / performance.json / integrity.json：可复核离线证据。",
        "- prepare_evaluation.py、prepare_candidates.py、run_evaluation.py、review_original.py、finalize_report.py：本地复现入口，只写本目录。冻结评估不覆盖；run_evaluation 不读运行配置，只读词汇快照。", "",
        "下一项具体工作：先独立审查否定范围与引述/比喻的通用匹配边界，或者继续保守跳过这些表达；为 29 行暂留候选准备新的独立验收样本，再决定是否申请阶段 4 应用。此次不写配置、不重启、不启动批量翻译或生图。",
        "文档差异登记：README 的 Toolbox 可导入说明不代表当前已应用；角色与服装工作流中的“已直接导入”段落已过时；configuration 中仍有“遗漏名词补全局 Tags”和旧固定数量写作说明，与本轮核对的现实现不符。以当前代码和交接边界为准，本轮只登记，避免混入既有流程修改。", "",
        "相关现有匹配回归执行 4 项通过（词汇仅参考、开关、紧贴否定、最长重叠与独立位置）；未重跑历史 488 项。此结果只验证实现边界，不代替中文语义质量。",
        f"完整性：原配置文件未变={integrity['config_entire_file_unchanged']}；已存词汇与评估未变；实际被评估的 resolver、映射解析、语义类型与本地 tag 模块散列未变={integrity['evaluated_matcher_unchanged']}。{integrity['existing_files_checked']} 个任务开始时既有文件核对，变化列表={changed}。这些文件本任务未写入，视为其他工作区改动并保留；其他会话新增文件不属于此初始散列清单。不声称运行态已热更新或当前完整流程已重新验收。",
    ]
    (ROOT/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Final provisional",len(provisional),"withdrawn",sorted(failed),"integrity",integrity)


if __name__=="__main__":
    main()
