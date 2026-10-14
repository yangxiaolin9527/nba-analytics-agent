"""Answer-only lineage enrichment; never changes tool results or analytics."""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from app.schemas import AskResponse


UNITS = {
    'points_total': 'points', 'points_per_appearance': 'points/appearance',
    'games_appeared': 'appearances', 'games_played': 'games',
    'assists_total': 'assists', 'rebounds_total': 'rebounds',
}


def build_answer_evidence(raw: AskResponse, database: Path, *, deadline: float) -> dict:
    """Copy provenance and bind SQL parameters from the validated tool arguments.

    Only game identity and omitted season sample counts require supplemental
    read-only lookups. These are lineage lookups, not additional tool plans.
    """
    args, result = raw.arguments or {}, raw.tool_result or {}
    evidence = raw.evidence.model_dump() if raw.evidence else {}
    evidence['source_snapshot'] = f"{evidence.get('source_name')}@{evidence.get('source_version')}"
    evidence.update(deepcopy(result.get('evidence') or {}))
    evidence.update(requested_scope=deepcopy(args), tool_status=result.get('status'),
                    quality_status=result.get('quality_status'),
                    limitations=list(result.get('limitations') or []),
                    precision={'calculation': 'full_precision', 'display_decimal_places': 4})
    if result.get('calculation_limitation'):
        evidence['limitations'].append(result['calculation_limitation'])
    if result.get('status') == 'partial':
        evidence['limitations'].append('partial_result')
    evidence['limitations'] = list(dict.fromkeys(evidence['limitations']))
    links = raw.resolved_values
    entities = (links.players + links.teams) if links else []
    names = {item.id: item.alias or item.display_name for item in entities}
    entity_id = args.get('entity_id')
    if entity_id:
        evidence['entity'] = {'id': entity_id, 'subject': args.get('subject', 'player'),
                              'name': names.get(entity_id, entity_id)}
    evidence['season_id'] = args.get('season_id')
    evidence['competition_phase'] = args.get('competition_phase')
    evidence['filters'] = {k: v for k, v in args.items() if k in (
        'date_from', 'date_to', 'last_n', 'opponent_team_id')}
    if args.get('opponent_team_id'):
        evidence['opponent'] = {'id': args['opponent_team_id'],
                               'name': names.get(args['opponent_team_id'], args['opponent_team_id'])}
    evidence['units'] = {key: UNITS[key] for key in args.get('metrics', []) if key in UNITS}
    for key in ('source_table', 'source_sql', 'source_parameters', 'inputs',
                'actual_sample_count', 'available_sample_count', 'actual_date_from',
                'actual_date_to', 'selected_game_ids'):
        if key in result:
            evidence[key] = deepcopy(result[key])
    if result.get('sql'):
        evidence.update(source_sql=result['sql'], source_parameters=[])
    if raw.tool == 'window_query':
        evidence['sample_count'] = result.get('actual_sample_count')
    elif raw.tool == 'player_efg':
        evidence['source_parameters'] = [args[k] for k in ('entity_id', 'season_id', 'competition_phase')]
        evidence['sample_count'] = (result.get('inputs') or {}).get('sample_count')
    elif raw.tool == 'season_compare':
        evidence['sources'] = []
        for role in ('baseline', 'comparison'):
            source = deepcopy(result.get(role) or {})
            if source:
                source.update(role=role, source_parameters=[args['entity_id'], source['season_id'], args['competition_phase']])
                evidence['sources'].append(source)
        evidence['sample_count'] = {s['role']: s['sample_count'] for s in evidence['sources']}
    if result.get('status') not in {'complete', 'partial'}:
        return evidence
    if raw.tool not in {'season_summary', 'single_game_score'}:
        return evidence
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('evidence deadline reached')
    with closing(sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True, timeout=remaining)) as conn:
        conn.execute('PRAGMA query_only = ON')
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 100)
        if raw.tool == 'season_summary':
            table, key, count = (('player_season_stat','person_id','games_appeared') if args['subject']=='player'
                                 else ('team_season_stat','team_id','games_played'))
            sql = f'SELECT {count} FROM {table} WHERE {key} = ? AND season_id = ? AND competition_phase = ?'
            params = [args[k] for k in ('entity_id','season_id','competition_phase')]
            row = conn.execute(sql, params).fetchone()
            if row is None or row[0] is None:
                raise ValueError('season sample evidence unavailable')
            evidence.update(sample_count=row[0], sample_source_sql=sql, sample_source_parameters=params)
        else:
            sql = 'SELECT game_date, season_id, competition_phase, home_team_id, away_team_id FROM game WHERE game_id = ?'
            row = conn.execute(sql, [args['game_id']]).fetchone()
            if row is None:
                raise ValueError('game evidence unavailable')
            evidence.update(game_id=args['game_id'], game_date=row[0], season_id=row[1],
                            competition_phase=row[2], sample_count=1)
            for role, team_id in (('home_team', row[3]), ('away_team', row[4])):
                name = conn.execute('SELECT display_name FROM team_identity WHERE team_id = ?', [team_id]).fetchone()
                evidence[role] = {'id': team_id, 'name': names.get(team_id) or (name[0] if name and name[0] else team_id)}
            if result.get('winner_team_id') not in {row[3], row[4]}:
                raise ValueError('winner not part of grounded game')
    return evidence
