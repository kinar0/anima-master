"""Whole-inventory screening with explicit semantic exceptions and limits."""
from collections import Counter
import json
from pathlib import Path
from run_evaluation import resolver, read, write, sha, PLUGIN, ROOT

# Categories distinguish definite non-equivalence from context sensitivity.
# Existing values remain untouched. Examples are diagnostic, never new aliases.
FLAGS = {
    "短裙": ("lost_modifier", "skirt 没编码短的条件", "短裙与长裙不同"),
    "丝袜": ("catalog_missing", "stocking 未在本地目录精确命中；袜型也不明确", "连裤与独立长袜需区别"),
    "中性": ("context_sensitive", "性别外观语境成立，配色或语气语境不成立", "中性灰配色"),
    "丰满": ("context_sensitive", "可指胸部或体型，plump 固定了体型解释", "胸部丰满但腰身纤细"),
    "低头": ("context_sensitive", "头姿不保证眼睛向下看", "低头但抬眼看观众"),
    "侧头": ("context_sensitive", "倾斜与转头侧向有别", "转头看侧面"),
    "侧身": ("context_sensitive", "on_side 是侧卧姿态，不是站着侧身", "侧身站立"),
    "变装": ("context_sensitive", "换装泛词不一定异性装扮", "从校服变装为同性交际礼服"),
    "另一人": ("context_sensitive", "1other 是性别计数，不是泛指又一位人物", "另一人是女孩"),
    "吊带": ("context_sensitive", "可指背心肩带，不一定背带 suspenders", "细吊带连衣裙"),
    "吐舌头": ("added_condition", ":p 带表情符号/嘴形语境，吐舌本身不保证该形状", "张大嘴伸出舌头"),
    "呼吸": ("context_sensitive", "一般呼吸不必有可见 breath", "静止肖像里平稳呼吸"),
    "外套": ("context_sensitive", "泛称包含大衣、披风等，不总是 jacket", "长款大衣外套"),
    "套装": ("context_sensitive", "不总是西式 suit", "运动套装"),
    "冠冕": ("context_sensitive", "tiara 结构不覆盖所有冠冕", "封闭王冠"),
    "星星": ("context_sensitive", "star_(symbol) 是符号，不必是天空恒星", "夜空中的星星"),
    "小孩": ("context_sensitive", "词义成立但名字、称呼或比喻不能变成年龄事实", "成年角色被昵称为小孩"),
    "开口": ("context_sensitive", "衣物开口或发言不保证张嘴状态", "胸前开口连衣裙"),
    "背心": ("context_sensitive", "可指 vest 或 sweater_vest，不能都当 tank_top", "西装针织背心"),
    "白丝袜": ("context_sensitive", "白色丝袜不一定连裤袜", "白色独立长袜"),
    "白裙子": ("context_sensitive", "口语裙子可指连衣裙，white_skirt 固定半身裙", "白色连衣裙"),
    "裙子": ("context_sensitive", "裙子口语形制不确定", "连衣裙也常叫裙子"),
    "眨眼": ("context_sensitive", "一般双眼眨动不是单眼 wink", "两只眼同时眨一下"),
    "卷发": ("context_sensitive", "卷曲强度多样，wavy_hair 固定波浪发", "紧密小卷"),
    "手腕袖口": ("unnatural", "中文表达生硬，袖口与独立 wrist_cuffs 要区别", "普通衬衫袖口"),
    "敞开外套": ("context_sensitive", "外套泛词固定 open_coat", "敞开的夹克"),
    "敞开的外套": ("context_sensitive", "外套泛词固定 open_jacket，与上一条近同义却不同", "敞开的大衣"),
    "浴袍": ("context_sensitive", "robe 泛袍不保证浴袍条件", "浴后浴袍"),
    "玩偶": ("context_sensitive", "不总是毛绒动物，可为人形 doll", "硬质人形玩偶"),
    "耳钉": ("lost_object", "ear_piercing 编码穿孔，不编码耳钉形制", "耳孔与耳钉区别"),
    "胸部": ("context_sensitive", "不必是乳房 breasts，可为男角色胸口", "胸部护甲"),
    "裙底": ("context_sensitive", "物件部位不必是向裙底拍摄", "裙底缝着蕾丝"),
    "说出心声": ("added_condition", "spoken_heart 需要心形符号，不是泛指真心话", "说出自己的担忧"),
    "走光": ("added_condition", "暴露部位未指定，pantyshot 固定内裤", "肩带滑落走光"),
    "趴": ("short_alias", "单字子串可命中字面提及和其他词", "趴这个字写在纸上"),
    "过时": ("context_sensitive", "画面年代感 dated 不等于物件过时", "过时的技术"),
    "长手套": ("added_condition", "长不保证 elbow_gloves 到肘长度", "过腕但不到肘的手套"),
    "闪烁": ("context_sensitive", "灯光闪动不必有 sparkle 星芒", "屏幕忽明忽暗"),
    "露出": ("short_alias", "任何露出都会触发 exposure；本地目录也缺失", "露出微笑"),
    "项圈": ("context_sensitive", "choker 常为贴颈饰品，动物项圈另有 collar", "狗项圈"),
    "马赛克": ("context_sensitive", "可为镶嵌图案，不一定 censored", "马赛克地砖"),
    "鸭舌帽": ("non_equivalent", "mob_cap 是褶边软帽，不是鸭舌帽", "普通鸭舌帽"),
    "运动服": ("context_sensitive", "泛运动服不都属于 gym_uniform", "职业队训练服"),
    "高帽": ("context_sensitive", "形容帽子高不保证 top_hat 礼帽", "高尖女巫帽"),
    "认真": ("context_sensitive", "行为态度不等于严肃表情", "认真看比赛但笑着"),
    "紫色上衣": ("context_sensitive", "上衣泛指不一定 shirt", "紫色毛衣"),
    "三人行": ("context_sensitive", "既可性语境，也可三人同行，不能无条件判定", "三人一起散步"),
    "骷髅": ("context_sensitive", "可能指完整骨架，不只是 skull", "完整人形骷髅"),
    "黑丝袜": ("catalog_missing", "black_stocking 无本地精确命中；袜型待明确", "黑色连裤袜与大腿袜"),
    "黑色紧身连衣裤": ("context_sensitive", "leotard 不覆盖带长裤腿的连体服", "全长连体裤"),
    "白色紧身连衣裤": ("context_sensitive", "leotard 不覆盖带长裤腿的连体服", "全长连体裤"),
    "白皮肤": ("context_sensitive", "white_skin 可能比浅色肤色更特异，需要核对", "普通浅肤色"),
    "双手相握": ("context_sensitive", "own_hands_together 是自己的双手，不是两个人握手", "两人双手相握"),
    "双辫": ("context_sensitive", "twin_braids 不一定覆盖两束非编织发束", "两束直发"),
}


def main():
    original=read(ROOT/"original_mappings.json")
    known=resolver(original)._configured_mappings("danbooru_term_mappings")
    local=read(ROOT/"tag_validation.json")["results"]
    inventory=[]
    for i,line in enumerate(original,1):
        from mapping_syntax import parse_mapping_entry
        pair=parse_mapping_entry(line)
        alias,tag=pair or ("","")
        normalized=known.get(alias.lower())
        records=[r for group in local.get(normalized,{}).get("confirmed_tags",{}).values() for r in group]
        exact=[r for r in records if r.get("tag")==normalized and r.get("match_layer")=="exact_tag"]
        flag=FLAGS.get(alias)
        judgement="user_confirmed_context" if alias=="事后" else "issue_flagged" if flag else "screened_no_obvious_issue_not_certified"
        inventory.append({"row":i,"original":line,"alias":alias,"canonical_tag":normalized,
            "parser_valid":bool(pair and normalized),"local_exact_present":bool(exact),
            "local_categories":[r.get("category") for r in exact],"semantic_screen":judgement,
            "issue_type":flag[0] if flag else None,"reason":flag[1] if flag else "用户确认该词库使用语境为性行为之后" if alias=="事后" else "全表词面对照初筛未发现显著问题；未做每项 Wiki 或真实使用语境验收",
            "nearby_counterexample":flag[2] if flag else None,"action":"report_only_preserve_original"})
    counts=Counter(r["semantic_screen"] for r in inventory)
    normalized_counts=Counter(r["alias"].lower() for r in inventory)
    overlaps=[{"short":a,"long":b,"short_tag":known[a],"long_tag":known[b]} for a in known for b in known if a!=b and a in b and known[a]!=known[b]]
    write("original_audit.json",{"scope":"全部 1427 行格式、目录存在性、别名字面语义初筛；不是全库语义正确认证",
        "counts":dict(counts),"unique_alias_count":len(known),"unique_tag_count":len(set(known.values())),
        "invalid_rows":[r["row"] for r in inventory if not r["parser_valid"]],
        "duplicate_aliases":{k:v for k,v in normalized_counts.items() if v>1},
        "kana_marked_only":[r["row"] for r in inventory if any("\u3040"<=ch<="\u30ff" for ch in r["alias"])],
        "local_missing_rows":[r for r in inventory if not r["local_exact_present"]],
        "substring_overlaps":overlaps,"rows":inventory})
    lines=["# 原有词库初筛问题清单", "", "完整逐行审查记录见 original_audit.json。原值全部保留。问题标记分为确定不等价、修饰条件丢失、语境风险和目录缺失；语境风险不等于该词条在所有使用中错误。", "", "用户本轮确认 `事后=aftersex` 在本词库语境中正确，已认可保留，不列为问题。", "", "| 行 | 原映射 | 类型 | 依据 | 邻近反例 |", "| --- | --- | --- | --- | --- |"]
    for r in inventory:
        if r["semantic_screen"]=="issue_flagged":
            lines.append(f"| {r['row']} | `{r['original']}` | {r['issue_type']} | {r['reason']} | {r['nearby_counterexample']} |")
    (ROOT/"original_issues.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Audit",dict(counts),"local missing rows",sum(not r["local_exact_present"] for r in inventory),"overlaps",len(overlaps))


if __name__=="__main__":
    main()
