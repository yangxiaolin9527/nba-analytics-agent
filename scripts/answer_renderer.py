"""Pure deterministic Chinese rendering; arithmetic remains in registered tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


_METRIC_LABELS = {
    'points_total': '总得分', 'points_per_appearance': '场均得分',
    'games_appeared': '出场次数', 'games_played': '比赛场数',
    'assists_total': '总助攻', 'rebounds_total': '总篮板',
}
_PHASES = {'regular_season': '常规赛', 'playoffs': '季后赛'}
_LIMITATION_LABELS = {
    'season_completeness_not_checked': '尚未确认是否覆盖该赛季全部比赛',
    'capability_evidence_conditional': '该统计范围的独立核验依据不完整',
    'historical_shooting_unverified': '投篮数据尚未完成独立核验',
    'capability_not_profiled': '该赛季和阶段尚未建立完整的数据质量评估',
    'fewer_than_requested_games': '实际场次少于请求数量',
    'partial_result': '结果只覆盖已取得的记录',
    'zero_baseline': '基线为 0，无法计算百分比变化',
    'inferred_appearance_included': '结果包含根据数据推断的出场记录',
    'game_count_mismatch': '数据库比赛数与参照赛程不一致',
    'team_context_incomplete': '部分球队关联信息尚未完成核验',
}


def _plain_limitations(tool: str, limitations: list[str]) -> list[str]:
    """Translate internal quality codes and avoid redundant caveats."""
    limitations = list(dict.fromkeys(limitations))
    if tool == 'single_game_score':
        limitations = [item for item in limitations if item != 'season_completeness_not_checked']
    if 'season_completeness_not_checked' in limitations:
        limitations = [item for item in limitations if item != 'capability_evidence_conditional']
    labels = dict(_LIMITATION_LABELS)
    labels['season_completeness_not_checked'] = {
        'season_summary': '尚未确认是否覆盖该赛季全部比赛',
        'window_query': '尚未确认当前查询范围是否覆盖全部比赛',
        'season_compare': '尚未确认两个赛季是否都覆盖全部比赛',
    }.get(tool, '尚未确认该查询范围是否完整')
    return [labels.get(item, '存在未展开的数据限制') for item in limitations]


@dataclass(frozen=True)
class RenderedAnswer:
    text: str
    values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError('answer requires a finite numeric tool value')
    if isinstance(value, int):
        return str(value)
    text = f'{value:.4f}'.rstrip('0').rstrip('.')
    return '0' if text == '-0' else text


def render_tool_answer(tool: str, arguments: dict, tool_result: dict, *, evidence: dict | None = None) -> RenderedAnswer:
    result, args = tool_result, arguments
    ev = dict(evidence or result.get('evidence') or {})
    limitations = list(ev.get('limitations', result.get('limitations') or []))
    warnings = []
    if result.get('status') == 'partial':
        warnings.append('partial_result')
        limitations.append('partial_result')
    quality = result.get('quality_status')
    if quality in {'conditional', 'unverified', 'unsupported'}:
        warnings.append('quality_' + quality)
    if result.get('calculation_limitation'):
        limitations.append(result['calculation_limitation'])
    limitations = list(dict.fromkeys(limitations))
    if result.get('status') == 'not_computable':
        return RenderedAnswer(render_failure('not_computable', result.get('reason_code'), result.get('reason')), warnings=warnings, evidence=ev)
    if result.get('status') not in {'complete', 'partial'}:
        raise ValueError('cannot render an unsuccessful tool response')
    name = (ev.get('entity') or {}).get('name') or f"{args.get('subject', 'player')} {args.get('entity_id', '')}"
    phase = _PHASES.get(args.get('competition_phase'), args.get('competition_phase', ''))
    scope = f"{name}在 {args.get('season_id', '')} {phase}"
    values: dict[str, Any]
    if tool == 'season_summary':
        values = {key: result[key] for key in args['metrics']}
        text = scope + '的' + '，'.join(f'{_METRIC_LABELS[key]}为 {_fmt(value)}' for key, value in values.items()) + '。'
        if ev.get('sample_count') is not None:
            text += f"实际样本数为 {_fmt(ev['sample_count'])}。"
    elif tool == 'single_game_score':
        values = {key: result[key] for key in ('home_score','away_score','winner_team_id')}
        home, away = ev.get('home_team') or {}, ev.get('away_team') or {}
        winner = home if result['winner_team_id'] == home.get('id') else away
        text = (f"{ev.get('game_date', '')} 比赛 {args['game_id']}："
                f"主队 {home.get('name', '')} 得分为 {_fmt(result['home_score'])}，"
                f"客队 {away.get('name', '')} 得分为 {_fmt(result['away_score'])}；"
                f"胜者为 {winner.get('name') or result['winner_team_id']}。")
    elif tool == 'window_query':
        values = dict(result['values'])
        if not values or any(key not in values for key in args.get('metrics', [])):
            raise ValueError('window metrics missing')
        text = scope + '，'
        if args.get('opponent_team_id'):
            text += f"对手为 {(ev.get('opponent') or {}).get('name', args['opponent_team_id'])}，"
        if args.get('date_from'):
            text += f"查询起始日期为 {args['date_from']}，"
        if args.get('date_to'):
            text += f"查询截止日期为 {args['date_to']}，"
        if args.get('last_n'):
            text += f"请求最近 {args['last_n']} {'次出场' if args.get('subject') == 'player' else '场比赛'}，"
        text += f"实际样本数为 {_fmt(result['actual_sample_count'])}，"
        text += '，'.join(f'{_METRIC_LABELS[k]}为 {_fmt(v)}' for k, v in values.items()) + '。'
        text += f"实际日期范围为 {result['actual_date_from']} 至 {result['actual_date_to']}。"
        if result['status'] == 'partial':
            text += '样本不足，仅回答上述实际样本范围。'
    elif tool == 'season_compare':
        baseline, comparison = result['baseline'], result['comparison']
        values = {'baseline_value':baseline['value'], 'comparison_value':comparison['value'],
                  'absolute_delta':result['absolute_delta'], 'relative_change_pct':result['relative_change_pct']}
        text = (f"{name}在{phase}的{_METRIC_LABELS[args['metric']]}："
                f"基线赛季 {baseline['season_id']} 为 {_fmt(baseline['value'])}，"
                f"比较赛季 {comparison['season_id']} 为 {_fmt(comparison['value'])}，"
                f"绝对变化为 {_fmt(result['absolute_delta'])}，")
        if result['relative_change_pct'] is None:
            if result.get('relative_change_status') != 'not_computable':
                raise ValueError('missing relative change without explanation')
            text += '相对变化无法计算（基线为零）。'
        else:
            text += f"相对变化为 {_fmt(result['relative_change_pct'])}%。"
        text += f"基线样本数为 {_fmt(baseline['sample_count'])}，比较样本数为 {_fmt(comparison['sample_count'])}。"
        text += f"公式：{ev['absolute_delta_formula']}；{ev['relative_change_formula']}。"
    elif tool == 'efg_compare':
        values = {key: result[key] for key in ('left_efg_percentage', 'right_efg_percentage', 'absolute_delta')}
        left, right = args['left_entity_id'], args['right_entity_id']
        delta = values['absolute_delta']
        conclusion = '两者相同' if delta == 0 else f"{left if delta > 0 else right} 更高"
        text = (f"{args['season_id']} {phase}，{left} 与 {right} 的 eFG% "
                f"相差 {_fmt(abs(delta))} 个百分点；{conclusion}。"
                f"有向差值（前者减后者）为 {_fmt(delta)} 个百分点。")
    elif tool == 'player_efg':
        values = {'efg_percentage':result['efg_percentage'], **result['inputs']}
        text = (scope + f"的 eFG% 为 {_fmt(result['efg_percentage'])}%，"
                f"计算输入为 FGM={_fmt(values['field_goals_made'])}、"
                f"3PM={_fmt(values['three_pointers_made'])}、FGA={_fmt(values['field_goals_attempted'])}。"
                f"实际样本数为 {_fmt(values['sample_count'])}。公式：{ev['formula']}。")
    else:
        raise ValueError('unregistered renderer')
    plain_limitations = _plain_limitations(tool, limitations)
    if plain_limitations:
        if 'season_completeness_not_checked' in limitations and tool != 'single_game_score':
            intro = {
                'season_summary': '以上结果按当前数据库中的比赛记录计算。',
                'window_query': '以上结果按当前数据库中当前查询范围内的比赛记录计算。',
                'season_compare': '以上结果按当前数据库中两个赛季找到的比赛记录计算。',
            }.get(tool, '以上结果按当前数据库中找到的记录计算。')
            text += intro
        text += f"说明：{'；'.join(plain_limitations)}。"
    return RenderedAnswer(text, values, warnings, ev)


def render_failure(status: str, reason_code: str | None, reason: str | None) -> str:
    clarifications = {
        'missing_phase': '请明确赛事阶段：常规赛还是季后赛。',
        'ambiguous_phase': '请只指定一个赛事阶段：常规赛或季后赛。',
        'missing_season': '请指定历史赛季，例如 2023-24。',
        'invalid_season': '赛季年份必须连续，请使用如 2024-25 或 2024-2025。',
        'missing_entity': '请指定一名球员或一支球队。',
        'ambiguous_entity': '实体名称有歧义，请提供完整球员或球队名称。',
        'missing_game': '请提供能唯一确定比赛的日期和双方球队。',
        'ambiguous_game': '多场比赛符合条件，请补充日期和双方球队。',
        'missing_window_size': '请指定最近多少场比赛或多少次出场。',
        'ambiguous_comparison_direction': '请明确基线赛季和比较赛季。',
        'missing_comparison_season': '请补充两个比较赛季或明确上一季。',
    }
    if status == 'clarification_required':
        return clarifications.get(reason_code, '请补充问题所缺少的赛季、赛事阶段、实体或比较范围。')
    return {
        'unsupported': '该问题超出当前已注册的 NBA 统计能力范围，暂不支持。',
        'not_computable': '当前数据或质量条件不足，无法可靠计算该结果。',
        'incomplete': '请求未能完成，未生成可验证的数值答案。',
    }[status]
