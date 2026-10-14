"""Lightweight, shared season and explicit-date contracts for query planning."""
from __future__ import annotations

import re
from datetime import date

SEASON_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}(?![-\d])")
_YEAR_DIGITS = '0-9零〇一二三四五六七八九'
_DIGIT_TRANSLATION = str.maketrans('零〇一二三四五六七八九', '00123456789')
_SEASON_SPELLING = re.compile(
    rf"(?<![{_YEAR_DIGITS}-])(?P<first>[{_YEAR_DIGITS}]{{4}})"
    rf"(?:\s*年)?\s*[-–—/至到]\s*(?P<last>[{_YEAR_DIGITS}]{{2,4}})"
    rf"(?![{_YEAR_DIGITS}])(?!\s*[-/]\s*[{_YEAR_DIGITS}])(?:\s*年)?"
)
DATE_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}-\d{2}(?!\d)")
_CHINESE_DATE = re.compile(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_MONTH_DAY = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日")


class QueryScopeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_question_seasons(question: str) -> str:
    def canonical(match: re.Match) -> str:
        first, last = (match.group(key).translate(_DIGIT_TRANSLATION) for key in ('first', 'last'))
        start, expected = int(first), int(first) + 1
        if not 1900 <= start <= 2099:
            return match.group()
        if len(last) not in {2, 4} or int(last) != (expected % 100 if len(last) == 2 else expected):
            raise ValueError('season must contain consecutive years')
        return f"{start}-{expected % 100:02d}"

    return _SEASON_SPELLING.sub(canonical, question)


def require_canonical_season(value: str) -> str:
    if SEASON_PATTERN.fullmatch(value) is None or normalize_question_seasons(value) != value:
        raise ValueError('season_id must use consecutive YYYY-YY years')
    return value


def normalize_question_dates(question: str) -> str:
    def stamp(year: int, month: int, day: int) -> str:
        try:
            return date(year, month, day).isoformat()
        except ValueError as exc:
            raise QueryScopeError('invalid_date', '请提供有效的完整日期。') from exc

    # Replace full dates first so their explicit year is never season-inferred.
    normalized = _CHINESE_DATE.sub(lambda m: stamp(*(int(part) for part in m.groups())), question)
    seasons = list(dict.fromkeys(SEASON_PATTERN.findall(normalized)))

    def partial(match: re.Match) -> str:
        if len(seasons) != 1:
            raise QueryScopeError('missing_date_year', '请提供完整日期年份或唯一赛季。')
        month, day = (int(part) for part in match.groups())
        year = int(seasons[0][:4]) + int(month < 10)
        return stamp(year, month, day)

    return _MONTH_DAY.sub(partial, normalized)


def explicit_dates(question: str) -> list[str]:
    values = list(dict.fromkeys(DATE_PATTERN.findall(question)))
    for value in values:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise QueryScopeError('invalid_date', '请提供有效的完整日期。') from exc
    return values


def requested_date_bounds(question: str) -> tuple[str | None, str | None]:
    dates = explicit_dates(question)
    if len(dates) > 2:
        raise QueryScopeError('ambiguous_dates', '最多支持两个日期边界，请明确查询范围。')
    if len(dates) == 2:
        if dates[0] > dates[1]:
            raise QueryScopeError('invalid_date_range', '起始日期不能晚于截止日期。')
        return dates[0], dates[1]
    if not dates:
        return None, None
    until = bool(re.search(r'截至|之前|以前|\b(?:until|before|through)\b', question, re.I))
    since = bool(re.search(r'自|从|以来|\b(?:since|from|after)\b', question, re.I))
    if until and since:
        raise QueryScopeError('ambiguous_dates', '请明确日期是起始还是截止边界。')
    return (None, dates[0]) if until else (dates[0], None) if since else (dates[0], dates[0])
