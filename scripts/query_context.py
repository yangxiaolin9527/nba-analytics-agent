"""Bounded schema and value context for the three Text-to-SQL experiments."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llama_index.core import SQLDatabase
from pydantic import BaseModel
from sqlalchemy import create_engine

from scripts.query_lexicon import load_lexicon
from scripts.schema_context import compile_context
from scripts.sql_gateway import DEFAULT_CATALOG
from scripts.query_scope import DATE_PATTERN, SEASON_PATTERN, normalize_question_seasons


ContextMode = Literal["full_schema", "schema_retrieval", "schema_value_retrieval"]
MODES = ("full_schema", "schema_retrieval", "schema_value_retrieval")


class EntityLink(BaseModel):
    alias: str
    id: str
    display_name: str


class GameLink(BaseModel):
    game_id: str
    game_date: str
    season_id: str
    competition_phase: str
    home_team_id: str
    away_team_id: str


class QueryLinks(BaseModel):
    players: list[EntityLink] = []
    teams: list[EntityLink] = []
    seasons: list[str] = []
    phases: list[str] = []
    games: list[GameLink] = []
    ambiguous_entities: list[str] = []


@dataclass(frozen=True)
class QueryContext:
    context: dict
    links: QueryLinks
    source_snapshot: str
    mode: ContextMode


def find_game_candidates(connection: sqlite3.Connection, question: str) -> list[dict[str, str]]:
    """Return deterministic game candidates from date, ID, and team grounding."""
    date_match = DATE_PATTERN.search(question)
    game_match = re.search(r"(?<!\d)[24]\d{7}(?!\d)", question)
    clauses: list[str] = []
    params: list[str] = []
    if date_match:
        clauses.append("game_date = ?")
        params.append(date_match.group(0))
    if game_match:
        clauses.append("game_id = ?")
        params.append(game_match.group(0).lstrip("0") or "0")
    # Team names alone describe an aggregation scope, not one unique game.
    # Avoid returning every historical game for player-vs-opponent questions.
    if not clauses:
        return []
    for team_id in [identity_id for _, identity_id in _aliases(question, "team")]:
        clauses.append("(home_team_id = ? OR away_team_id = ?)")
        params.extend([team_id, team_id])
    sql = "SELECT game_id, game_date, season_id, competition_phase, home_team_id, away_team_id FROM game WHERE "
    sql += " AND ".join(clauses) + " LIMIT 9"
    rows = connection.execute(sql, tuple(params)).fetchall()
    return [
        GameLink(game_id=row[0], game_date=row[1] or "", season_id=row[2] or "",
                 competition_phase=row[3], home_team_id=row[4] or "", away_team_id=row[5] or "")
        for row in rows
    ]


def _aliases(question: str, kind: str) -> list[tuple[str, str]]:
    aliases = load_lexicon()["entity_aliases"][kind]
    matches: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias, identity_id in sorted(aliases.items(), key=lambda pair: (-len(pair[0]), pair[0])):
        haystack, needle = question, alias
        if alias.isascii():
            haystack, needle = question.casefold(), alias.casefold()
        if needle in haystack and identity_id not in seen:
            matches.append((alias, identity_id))
            seen.add(identity_id)
    return matches


def select_tables(question: str, links: QueryLinks | None = None) -> tuple[str, ...]:
    players = _aliases(question, "player")
    teams = _aliases(question, "team")
    if links is not None:
        players = players or [(item.alias, item.id) for item in links.players]
        teams = teams or [(item.alias, item.id) for item in links.teams]
    game_question = bool(DATE_PATTERN.search(question) or re.search(r"(?<!\d)[24]\d{7}(?!\d)", question))
    game_question |= any(token in question for token in ("比分", "那场", "单场", "哪场"))
    opponent_question = bool(players and teams and any(token in question for token in ("对", "面对")))
    if opponent_question:
        return ("game", "player_game_stat", "player_identity", "team_identity")
    if game_question:
        return ("game", "team_identity")
    if players:
        return ("player_season_stat", "player_identity")
    if teams:
        return ("team_season_stat", "team_identity")
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    return tuple(sorted(name for name, entry in catalog["tables"].items()
                        if entry.get("layer") in {"conformed", "mart"}))


def _link_entities(connection, question: str) -> QueryLinks:
    question = normalize_question_seasons(question)
    lexicon = load_lexicon()
    entities: dict[str, list[EntityLink]] = {"player": [], "team": []}
    ambiguous_entities: list[str] = []
    for kind, table, key in (
        ("player", "player_identity", "person_id"),
        ("team", "team_identity", "team_id"),
    ):
        aliases = _aliases(question, kind)
        for alias, identity_id in aliases:
            row = connection.execute(
                f"SELECT display_name FROM {table} WHERE {key} = ?",
                (identity_id,),
            ).fetchone()
            if row is not None:
                entities[kind].append(EntityLink(alias=alias, id=identity_id, display_name=row[0]))
        rows = connection.execute(
            f"SELECT {key}, display_name FROM {table} "
            "WHERE display_name IS NOT NULL AND trim(display_name) <> '' "
            "AND instr(lower(?), lower(display_name)) > 0 "
            "ORDER BY length(display_name) DESC LIMIT 9",
            (question,),
        ).fetchall()
        by_name: dict[str, list[tuple[str, str]]] = {}
        for identity_id, display_name in rows:
            by_name.setdefault(display_name, []).append((str(identity_id), display_name))
        for display_name, matches in by_name.items():
            if len(matches) > 1:
                ambiguous_entities.append(f"{kind}:{display_name}")
                continue
            identity_id, canonical_name = matches[0]
            if identity_id not in {item.id for item in entities[kind]}:
                entities[kind].append(EntityLink(alias=display_name, id=identity_id, display_name=canonical_name))
    phases = [canonical for canonical, aliases in lexicon["value_domains"]["competition_phase"].items()
              if any(alias.lower() in question.lower() for alias in aliases)]
    return QueryLinks(
        players=entities["player"],
        teams=entities["team"],
        seasons=list(dict.fromkeys(SEASON_PATTERN.findall(question))),
        phases=phases,
        ambiguous_entities=sorted(ambiguous_entities),
    )


def build_query_context(db_path: Path, question: str, mode: ContextMode) -> QueryContext:
    if mode not in MODES:
        raise ValueError(f"unknown context mode: {mode}")
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    allowed = sorted(name for name, entry in catalog["tables"].items()
                     if entry.get("layer") in {"conformed", "mart"})
    links = QueryLinks()
    if mode == "schema_value_retrieval":
        with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            links = _link_entities(connection, question)
            links = links.model_copy(update={"games": find_game_candidates(connection, question)})
    tables = allowed if mode == "full_schema" else list(select_tables(question, links))
    context = compile_context(DEFAULT_CATALOG, tables=tables, metrics=list(catalog.get("metrics", {})))

    def connect() -> sqlite3.Connection:
        return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)

    engine = create_engine("sqlite://", creator=connect)
    try:
        sql_database = SQLDatabase(
            engine, include_tables=tables, sample_rows_in_table_info=0, view_support=True
        )
        context["physical_schema"] = {
            table: sql_database.get_single_table_info(table) for table in tables
        }
    finally:
        engine.dispose()
    return QueryContext(context, links, catalog["source_snapshot"], mode)
