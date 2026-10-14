"""Resolve language references to identities in the pinned relational database."""
from __future__ import annotations

import sqlite3
from app.semantic_contracts import EntityReference
from scripts.ask_grounding import AskGroundingError
from scripts.query_context import EntityLink, _link_entities


def resolve_entity(connection: sqlite3.Connection, reference: EntityReference, kind: str,
                   question: str, previous=None) -> tuple[EntityLink, str]:
    table, key = ('player_identity', 'person_id') if kind == 'player' else ('team_identity', 'team_id')
    def exact(name):
        return connection.execute(f'SELECT {key}, display_name FROM {table} WHERE lower(trim(display_name)) = lower(?)', (name,)).fetchall()

    if reference.inherit:
        explicit = _link_entities(connection, question)
        items = explicit.players if kind == 'player' else explicit.teams
        if items:
            raise AskGroundingError('entity_conflict', '本轮已指定实体，不能继承上一轮实体。', 'clarification_required')
        ids = getattr(previous, 'player_ids' if kind == 'player' else 'team_ids', ())
        if len(ids) != 1:
            raise AskGroundingError('missing_entity', '请提供完整球员或球队名称。', 'clarification_required')
        rows = connection.execute(f'SELECT {key}, display_name FROM {table} WHERE {key} = ?', (ids[0],)).fetchall()
        source = 'conversation'
    else:
        mention = (reference.mention or '').strip()
        if not mention or mention.casefold() not in question.casefold():
            raise AskGroundingError('missing_entity', '实体名称必须来自问题，请提供完整名称。', 'clarification_required')
        links = _link_entities(connection, mention)
        known = links.players if kind == 'player' else links.teams
        rows = [(item.id, item.display_name) for item in known]
        source = 'alias_or_database'
        proposed = exact(reference.canonical_name.strip()) if reference.canonical_name else []
        if rows and proposed and {str(row[0]) for row in rows} != {str(row[0]) for row in proposed}:
            raise AskGroundingError('entity_conflict', '名称和模型提出的实体冲突，请提供完整名称。', 'clarification_required')
        # A surname/partial English name is only a candidate query, never a
        # license to narrow several identities using the model's guess.
        if not rows and mention.isascii():
            rows = connection.execute(f'SELECT {key}, display_name FROM {table} WHERE instr(lower(display_name), lower(?)) > 0 ORDER BY display_name LIMIT 10', (mention,)).fetchall()
            source = 'database_candidate'
        if not rows and proposed:
            rows, source = proposed, 'model_proposed_name'
    if len(rows) != 1:
        names = ', '.join(str(row[1]) for row in rows[:5])
        code = 'ambiguous_entity' if rows else 'unknown_entity'
        raise AskGroundingError(code, f'请提供完整名称。候选：{names}' if names else '数据库未能确认该实体，请提供完整英文名称。', 'clarification_required')
    return EntityLink(alias=reference.mention or str(rows[0][1]), id=str(rows[0][0]), display_name=str(rows[0][1])), source
