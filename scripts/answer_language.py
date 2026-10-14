"""Conservative zh/en publication checks, not general language identification.

Only obvious script mismatches are rejected. Ambiguity, requested literals and
unsupported languages remain explicitly unverified; script compatibility does
not prove fluent or complete answers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


_HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
_WORDS = re.compile(r'[A-Za-z]+')
_OTHER_SCRIPT = re.compile(r'[\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]')
_QUOTES = re.compile(r'`[^`]*`|“[^”]*”|「[^」]*」|"[^"]*"')
_DIRECTIVE = re.compile(
    r'(?:请)?(?:用|使用|以)\s*(?P<zh>[^\s，。！？；,.;!?]{1,12}?(?:文|语))\s*'
    r'(?:来)?(?:回答|作答|解释|说明|回复)|'
    r'(?:answer|respond|reply|explain)\b[^.!?;\n]{0,100}?\bin\s+(?P<en>[a-z]+)', re.I)
_EN_FUNCTION_WORDS = re.compile(
    r'\b(?:what|which|who|how|when|why|is|are|does|the|a|an|of|in|to|for)\b', re.I)
_TERM_REQUEST = re.compile(
    r'(?:英文|英语|english)\s*(?:全称|名称|名字|术语|name|term)|'
    r'(?:英文|英语).*?(?:叫什么|怎么说)|english\s+(?:name|term)', re.I)
_QUOTE_REQUEST = re.compile(
    r'(?:原文|原句|原话)\s*(?:是什么|是啥|怎么写)|'
    r'(?:给出|提供|引用|摘录|返回).*?(?:原文|原句|原话)|'
    r'verbatim|quote\s+(?:the\s+)?(?:source|original)', re.I)


def _prose(text: str) -> str:
    # Exclude marked literal material when deciding the surrounding prose script.
    return _QUOTES.sub('', text)


@dataclass(frozen=True)
class AnswerLanguage:
    expected: str | None
    source: str

    @classmethod
    def from_question(cls, question: str) -> 'AnswerLanguage':
        prose = _prose(question)
        explicit = []
        for match in _DIRECTIVE.finditer(prose):
            prefix = prose[max(0, match.start() - 8):match.start()]
            if re.search(r'(?:不要|不用|不需|无需|别|不|not|never|don.t)\s*$', prefix, re.I):
                continue
            label = (match.group('zh') or match.group('en')).lower()
            explicit.append({'中文': 'zh', '汉语': 'zh', 'chinese': 'zh',
                             '英文': 'en', '英语': 'en', 'english': 'en'}.get(label))
        if explicit:
            if len(set(explicit)) == 1 and explicit[0] is not None:
                return cls(explicit[0], 'explicit')
            return cls(None, 'ambiguous_or_unsupported')
        if _OTHER_SCRIPT.search(prose):
            return cls(None, 'unsupported')
        han = len(_HAN.findall(prose))
        words = _WORDS.findall(prose)
        if han >= 4 and len(words) <= han:
            return cls('zh', 'question_script')
        if not han and prose.isascii() and _EN_FUNCTION_WORDS.search(prose):
            return cls('en', 'question_script')
        return cls(None, 'undetermined')

    def check(self, claim: str, subquestion: str, excerpt: str) -> str:
        if self.expected is None or _OTHER_SCRIPT.search(claim):
            return 'unverified'
        text = _prose(claim)
        if not text.strip():
            text = claim  # Wrapping an entire answer in quotes cannot exempt it.
        han = len(_HAN.findall(text))
        words = _WORDS.findall(text)
        prose_words = [word for word in words if not word.isupper()]
        if not han and (not words or (re.search(r'[=+*/]', text) and len(prose_words) <= 2)):
            return 'unverified'  # Formula/numerical output is not language prose.
        if (_TERM_REQUEST.search(_prose(subquestion)) and len(words) <= 5
                and not _EN_FUNCTION_WORDS.search(text)):
            return 'unverified'
        if (_QUOTE_REQUEST.search(_prose(subquestion))
                and re.sub(r'\s+', ' ', claim.strip('“”" '))
                in re.sub(r'\s+', ' ', excerpt)):
            return 'unverified'  # Exemption requires actual requested literal content.
        if self.expected == 'zh':
            if not han and len(words) >= 4:
                return 'mismatch'
            if han < len(words) and len(words) >= 6:
                return 'mismatch'
            if han >= 4 and len(words) <= han:
                return 'script_compatible'
        elif self.expected == 'en':
            if han >= 4 and len(words) < han:
                return 'mismatch'
            if not han and len(words) >= 3 and text.isascii():
                return 'script_compatible'
        return 'unverified'
