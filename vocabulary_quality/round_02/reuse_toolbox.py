"""Select existing/focus concepts from Toolbox as maintenance-only evidence."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PLUGIN=ROOT.parents[1]
SOURCE=Path("D:/Users/69493/Desktop/llm-nai-toolbox-0.9.8-win-x64/resources/tagdb")
FIELDS=("zhFull", "zhShort", "zhNick", "zh")


def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output=PLUGIN/"third_party/llm_nai_toolbox/vocabulary_review_catalog.json"
    if output.exists(): raise SystemExit("Maintenance catalog exists; refusing overwrite")
    original=read(ROOT/"baseline.json")["original_rows"]
    current={row.split("=",1)[0].casefold():row.split("=",1)[1] for row in original}
    old_candidates=read(PLUGIN/"vocabulary_quality/candidates.json")["concepts"]
    focus={r["canonical_tag"]:r for r in old_candidates}
    audit=read(PLUGIN/"vocabulary_quality/original_audit.json")
    issues=defaultdict(list)
    for row in audit["rows"]:
        if row["semantic_screen"] in {"issue_flagged","user_confirmed_context"}:
            issues[row["canonical_tag"]].append({"alias":row["alias"],"reason":row["reason"],
                "type":row["semantic_screen"],"counterexample":row.get("nearby_counterexample")})
    index=read(SOURCE/"tags_index_v2.json")["general"]
    gloss=read(SOURCE/"tag_gloss.json")
    browse=read(SOURCE/"tag_browse.json")
    by_tag={row["tag"]:row for row in index}
    owners=defaultdict(set)
    for row in index:
        for field in FIELDS:
            for alias in row.get(field,[]):
                if isinstance(alias,str): owners[" ".join(alias.casefold().split())].add(row["tag"])
    selected=set(current.values())|set(focus)
    # Include alternative meanings of selected aliases for review, not binding.
    alternatives=set()
    for tag in sorted(selected):
        row=by_tag.get(tag,{})
        for field in FIELDS:
            for alias in row.get(field,[]):
                alternatives.update(owners[" ".join(alias.casefold().split())])
        for pair in gloss.get(tag,{}).get("vs",[]):
            if isinstance(pair,list) and pair: alternatives.add(pair[0])
    selected |= alternatives
    categories=defaultdict(list)
    for name,rows in browse.items():
        for row in rows:
            if row.get("t") in selected: categories[row["t"]].append(name)
    validation=read(PLUGIN/"vocabulary_quality/tag_validation.json")["results"]
    records=[]
    for tag in sorted(selected,key=lambda t:(t not in focus,t not in issues,t)):
        row=by_tag.get(tag,{})
        values=defaultdict(list)
        for field in FIELDS:
            for alias in row.get(field,[]):
                if isinstance(alias,str) and 2<=len(alias.strip())<=80 and not any(c in alias for c in "=|\r\n\t"):
                    values[alias.strip()].append(field)
        aliases=[]
        for alias,fields in values.items():
            targets=sorted(owners[" ".join(alias.casefold().split())])
            aliases.append({"alias":alias,"source_fields":list(dict.fromkeys(fields)),
                "owners":targets,"ambiguous":len(targets)>1,
                "language_note":"含假名，仅作标记，需要语义审查" if any("\u3040"<=c<="\u30ff" for c in alias) else ""})
        local=validation.get(tag,{})
        confirmed=[r for rs in local.get("confirmed_tags",{}).values() for r in rs if r.get("tag")==tag]
        g=gloss.get(tag,{})
        f=focus.get(tag,{})
        records.append({"tag":tag,"definition":g.get("g", ""),"trap":g.get("trap", ""),
            "contrasts":g.get("vs",[]),"categories":categories[tag],"aliases":aliases,
            "source_present":bool(row),"local_coverage":"exact_present" if confirmed else "not_checked" if tag not in validation else "missing_in_local_catalog",
            "local_categories":[r.get("category") for r in confirmed],"nai_coverage":"未验证；本地缺失不等于 NAI 不支持",
            "focus":tag in focus,"issues":issues[tag],"positive_examples":f.get("positive_examples",[]),
            "counterexamples":f.get("nearby_counterexamples",[]),"previous_candidate_aliases":f.get("aliases",[]),
            "prior_review":f.get("review_decision", ""),"prior_basis":f.get("review_basis", ""),
            "source":"Toolbox 0.9.8 general / gloss / browse；释义为来源材料，待审核"})
    source={"version":"0.9.8","upstream":"https://github.com/sadlywish/llm-nai-toolbox",
        "notice":"third_party/llm_nai_toolbox/NOTICE.md",
        "sha256":{name:digest(SOURCE/name) for name in ("tags_index_v2.json","tag_gloss.json","tag_browse.json")},
        "scope":"当前词库、首轮缺口与相邻歧义概念；维护材料，不参与运行匹配"}
    value={"source":source,"records":records,"applied":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    (ROOT/"reuse_summary.json").write_text(json.dumps({"selected_concepts":len(records),
        "focus_concepts":len(focus),"with_gloss":sum(bool(r["definition"]) for r in records),
        "alias_proposals":sum(len(r["aliases"]) for r in records),
        "ambiguous_alias_occurrences":sum(a["ambiguous"] for r in records for a in r["aliases"]),
        "source":source,"catalog_sha256":digest(output),"applied":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Maintenance concepts",len(records),"gloss",sum(bool(r["definition"]) for r in records),"not applied")


if __name__=="__main__": main()
