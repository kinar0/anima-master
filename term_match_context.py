"""Conservative occurrence filtering for vocabulary references, not semantics.

Works in the same normalized text coordinate system as alias matching. It only
withholds references in bounded explicit negative/literal contexts. Original
text still reaches the writer. Unknown context is not converted into a tag.
"""
from __future__ import annotations
import re

_CLAUSE = re.compile(r"[^，,。；;！!？?\n]+")
_TURN = re.compile(r"但是|但(?!是)|而且|而|却|然后|同时|改成|改为|换成|\bbut\b", re.I)
_NEGATIVE = re.compile(
    r"(?:禁止|不要|(?<!告)(?<!区)别(?!人|着)|不想|没想|不打算|没打算|不准备|不希望|不需要|不愿|不再|"
    r"没有|没|并非|不是|去掉|移除|删除|取消|不(?!但|仅|只|得不|同|对称))"
    r"\s*(?:(?:想|打算|准备|计划|希望|需要|愿意|要|再|曾|穿着|穿|戴着|戴|"
    r"带着|带|使用|画出|画|出现|呈现|表现|描绘|添加|加入|展示|提供|有|留|保留|"
    r"任何|一切|一件|一个|一双|一些)\s*)*"
    r"|\b(?:without|remove|delete|no|not(?!\s+only\b)|don't|doesn't)\b"
    r"\s*(?:(?:want\s+to|intend\s+to|wear|use|show|draw|the|any|an?|one)\s*)*",
    re.I,
)
_POSITIVE_ACTION = re.compile(r"穿着|穿|戴着|戴|拿着|拿|持有|保留|有|在|来到|\b(?:wearing|holding|keep|in|at)\b",re.I)
_QUOTES = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"')
_SPEECH = re.compile(r"(?:说|说道|说出|念|读|喊|台词|对白|文字|字样|写着|写上|写有|标语|题词)\s*[:：]?\s*$")
_LITERAL_SUFFIX = re.compile(r"(?:这|那)?(?:两|三|几|二|一个|两个)?(?:个)?(?:字|词|这个词|这个字)")
_LITERAL_PREFIX = re.compile(r"(?:说|念|读|写|提到|提及|喊|台词是|名字是)\s*$")
_MENTAL_PREFIX = re.compile(r"(?:心里|心中|内心)(?:的)?\s*$")
_METAPHOR = re.compile(r"(?:只是|仅是|只是一种|只是个|仅仅是)(?:比喻|隐喻|修辞|形容|象征说法)")


class TermMatchContext:
    def __init__(self, text: str):
        self.text = text
        self.blocked: list[tuple[int,int]] = []
        self.negative: list[tuple[int,int]] = []
        for quoted in _QUOTES.finditer(text):
            if _SPEECH.search(text[max(0,quoted.start()-24):quoted.start()]):
                self.blocked.append(quoted.span())
        for clause in _CLAUSE.finditer(text):
            # Explicit "just a metaphor" is evidence; generic "like" is not.
            if metaphor := _METAPHOR.search(clause.group()):
                self.blocked.append((clause.start(),clause.start()+metaphor.end()))
            left = clause.start()
            boundaries = [clause.start()+m.start() for m in _TURN.finditer(clause.group())]+[clause.end()]
            for right in boundaries:
                part = text[left:right]
                for negative in _NEGATIVE.finditer(part):
                    start = left+negative.end()
                    # A new positive verb after an object ends the negative
                    # range ("不戴帽子穿衬衫"). Lists without a new verb remain
                    # negative ("不要帽子和手套"). Unclear quantifiers remain a
                    # documented limitation rather than a guessed ownership.
                    action = _POSITIVE_ACTION.search(text,start,right)
                    end = action.start() if action and action.start()>start else right
                    self.negative.append((start,end))
                left=right

    def suppressed(self, start: int, end: int) -> bool:
        if any(start<b and a<end for a,b in self.blocked):
            return True
        # A configured absence/state phrase can itself begin with "没穿" or
        # "no". Negation inside that complete alias is not an outside command
        # negating it. Longest overlap has already removed internal short words.
        if any(a <= start < b for a,b in self.negative):
            return True
        prefix=self.text[max(0,start-24):start]
        suffix=self.text[end:end+8]
        if _MENTAL_PREFIX.search(prefix):
            return True
        if re.match(r"(?:这个字|这个词|这两个字|这几个字|两个字|三个字|二字)", suffix):
            return True
        return bool(_LITERAL_PREFIX.search(prefix) and _LITERAL_SUFFIX.match(suffix))
