"""Curated offline proposals from development gaps; no bulk Toolbox import."""
from pathlib import Path
import json
import subprocess
from prepare_evaluation import CONCEPTS

ROOT=Path(__file__).resolve().parent
SKILL=Path("C:/Users/69493/.codex/skills/danbooru-tags")

# Natural equivalent phrases only. Some are intentionally deferred after review;
# they remain visible as proposals rather than silently disappearing.
ALIASES=[
    ["抱抱", "相拥"], ["摸摸头", "摸头杀"], ["把手牵起来", "牵起手"],
    ["十指紧扣", "十指交扣"], ["亲亲", "亲一口"], ["露出笑容", "笑眯眯"],
    ["面颊泛红", "双颊泛红"], ["哭得掉眼泪", "哭出来"], ["双眼闭上", "闭上双眼"],
    ["眨一只眼", "单眼眨眼"], ["把嘴张开", "张开嘴巴"], ["站着"], ["坐着"],
    ["双膝跪地", "跪地"], ["蹲着"], ["躺平"], ["肚子朝下趴着"],
    ["双臂交叉抱在胸前"], ["两只胳膊举起来", "双臂高举"], ["两条腿并在一起", "两腿并拢"],
    ["白色连身裙", "白连身裙"], ["白色半身裙", "白半身裙"],
    ["黑色裤袜", "黑裤袜"], ["白色裤袜", "白裤袜"],
    ["连体式泳装", "连体泳装"], ["帽衫"], ["针织毛衣"], ["球鞋"],
    ["半指手套", "露指手套"], ["牛仔热裤"], ["雨衣"],
    ["举着伞", "撑着伞"], ["捧着书"], ["端着杯子", "捧着杯子"],
    ["握着手机"], ["握着剑"], ["双肩包"], ["头发上别着蝴蝶结"],
    ["毛绒公仔", "绒毛玩偶"], ["太阳眼镜"], ["贴颈项链"], ["围脖"],
    ["下着雨", "雨天"], ["下着雪", "飘雪"], ["海边沙滩"],
    ["教室"], ["公园"], ["满天繁星", "繁星满天"],
]

# Semantically overlapping phrases that are unsafe in literal matching are
# retained in the maintenance file but excluded from the proposed additions.
DEFER={
    "hug":"抱抱既可指亲密接触，也可为口头安慰或抽象对象；本阶段不强绑。相拥可继续单独试验。",
    "headpat":"摸头杀含评价或招式用法；且轻摸他人头的语序变化不能靠固定短语覆盖。",
    "kiss":"亲亲既是称呼又是动作；原文给 LLM。",
    "on_back":"躺平常见比喻且未必面朝上，不能等价于仰卧。",
    "on_stomach":"当前正例无新增覆盖需求，且单字趴存在原库误命中。",
    "closed_eyes":"动作变化与最终闭眼状态需要时态上下文，先保持候选记录。",
    "wink":"单眼眨动还可为眼部反射，未必 wink；需动作语境。",
    "snow":"降雪不同于积雪物件，需确认 canonical 定义后再接受；本地存在不是语义证明。",
    "beach":"原库沙滩已覆盖该开发正例，不因规模预算添加冗余词。",
    "scarf":"围脖可以是圈状颈套，未必等价 scarf 的具体结构。",
    "choker":"贴颈项链仍可有不同结构，canonical 适用需进一步语义核对。",
    "sweater":"针织毛衣本身已被原库毛衣覆盖，首批无需扩展。",
    "hair_bow":"短语附带安装位置与动作；更自然的发饰别名原库已有，暂不以长句补词。",
    "crossed_arms":"较长姿态描述不适合作为字面别名补洞，原库抱臂已有。",
}

DEFINITIONS = [
    "拥抱他人", "轻柔拍摸他人的头", "两人牵手", "手指交错扣住", "亲吻", "微笑表情",
    "面部泛红", "哭泣", "双眼闭合", "单眼眨眼示意", "嘴巴张开", "站立姿态", "坐姿",
    "跪姿", "蹲姿", "面朝上仰卧", "腹部朝下俯卧", "双臂交叉", "双臂举起", "双腿并拢",
    "白色连衣裙", "白色半身裙", "黑色连裤袜", "白色连裤袜", "连成一件的泳装",
    "连帽卫衣", "毛衣", "运动鞋", "露出手指的手套", "牛仔短裤", "防雨外衣",
    "手持雨伞", "手持书", "手持杯", "手持手机", "手持剑", "双肩背包", "头发上的蝴蝶结",
    "毛绒玩具", "太阳镜", "贴颈饰品", "围巾", "降雨", "雪", "海滩", "教室", "公园", "星空",
]


def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    existing = json.loads((ROOT/"candidates.json").read_text(encoding="utf-8")) if (ROOT/"candidates.json").exists() else None
    if existing and [(c["canonical_tag"],c["aliases"]) for c in existing["concepts"]] != [(c[0],a) for c,a in zip(CONCEPTS,ALIASES,strict=True)]:
        raise SystemExit("Candidate content changed; refusing overwrite")
    original=json.loads((ROOT/"original_mappings.json").read_text(encoding="utf-8"))
    from run_evaluation import resolver
    known=resolver(original)._configured_mappings("danbooru_term_mappings")
    # Exact local inventory query; raw results are evidence of presence only.
    tags=sorted(set(known.values()) | {c[0] for c in CONCEPTS})
    dump("tag_queries.json",{"queries":[{"id":tag,"keyword":tag,"match_mode":"exact","limit":5} for tag in tags]})
    if (ROOT/"tag_validation.json").exists():
        data=json.loads((ROOT/"tag_validation.json").read_text(encoding="utf-8"))
    else:
        result=subprocess.run([str(SKILL/"bin/danbooru-tags.exe"),"--batch-file",str(ROOT/"tag_queries.json"),
            "--match-mode","exact","--batch-workers","8","--json","--compact"],capture_output=True)
        if result.returncode:
            raise SystemExit("Local tag CLI failed; no fallback")
        data=json.loads(result.stdout.decode("utf-8"))
    if "results" not in data:
        raise SystemExit("Local tag CLI returned unexpected schema")
    dump("tag_validation.json",data)
    concepts=[]
    for concept,aliases,definition in zip(CONCEPTS,ALIASES,DEFINITIONS,strict=True):
        tag,pos,neg,reason,incidental=concept
        raw=data["results"].get(tag,{})
        records=[r for bucket in ("confirmed_tags","candidate_tags")
                 for values in raw.get(bucket,{}).values() if isinstance(values,list)
                 for r in values if isinstance(r,dict)]
        exact=[r for r in records if r.get("tag")==tag and r.get("match_layer")=="exact_tag"]
        coverage={"anima_local_catalog":"exact_present" if exact else "not_exactly_present",
                  "records":exact,"nai":"not_verified; absence in Anima is not evidence of NAI incompatibility"}
        conflicts=[{"alias":a,"existing_tag":known[a.lower()],"type":"same" if known[a.lower()]==tag else "conflict"} for a in aliases if a.lower() in known]
        verdict="deferred_semantics" if tag in DEFER else "deferred_local_missing" if not exact else "proposed"
        concepts.append({"canonical_tag":tag,"aliases":aliases,"chinese_definition":definition,
            "equivalence_basis":reason,
            "source":"本任务人工撰写；依据开发集缺口；未复用 Toolbox 别名、未调用翻译模型",
            "existing_mapping_conflicts":conflicts,"backend_coverage":coverage,
            "positive_examples":[pos],"nearby_counterexamples":[neg],
            "review_decision":verdict,"review_basis":DEFER.get(tag,reason),
            "authority":"词汇参考；不是 hard tag、衣柜权限或穿着者证据"})
    dump("candidates.json",{"status":"offline_proposals_not_applied", "concepts":concepts,
        "ambiguous_skips":[{"expression":x,"reason":y} for x,y in [
            ("白裙子","连衣裙与半身裙多义，不新增强绑"),("黑丝","袜型不明确"),
            ("张开","没有部位对象"),("拍头","力度与对象不明"),("抱抱","口头安慰或抽象对象"),
            ("巨乳","保留原值，不统一 large/huge/gigantic 胸部尺寸等级")]]})
    print("Concept proposals",len(concepts),"local exact",sum(c["backend_coverage"]["anima_local_catalog"]=="exact_present" for c in concepts))


if __name__=="__main__":
    main()
