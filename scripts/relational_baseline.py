"""Build and query the M1 conformed SQLite model.

The module deliberately uses the Python standard library.  It is a small,
deterministic baseline for validating the logical model before choosing a
production database or Text-to-SQL runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable

from scripts.game_semantics import classify_game
from scripts.player_quality import classify_player, parse_count, parse_minutes


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/eoinamoore/v515"
SNAPSHOT_ID = "eoin-v515"
SOURCE_SNAPSHOT = "eoinamoore/historical-nba-data-and-player-box-scores@515"
CAPABILITY_MATRIX = ROOT / "data/profile/eoin-v515-capability-matrix.json"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dataset_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_version TEXT NOT NULL,
    acquired_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_file (
    snapshot_id TEXT NOT NULL REFERENCES dataset_snapshot(snapshot_id),
    file_name TEXT NOT NULL,
    file_sha256 TEXT,
    row_count INTEGER,
    PRIMARY KEY (snapshot_id, file_name)
);
CREATE TABLE IF NOT EXISTS source_record (
    snapshot_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    source_row_id INTEGER NOT NULL,
    raw_record_json TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, file_name, source_row_id),
    FOREIGN KEY (snapshot_id, file_name) REFERENCES source_file(snapshot_id, file_name)
);
CREATE TABLE IF NOT EXISTS player_identity (
    person_id TEXT PRIMARY KEY,
    display_name TEXT,
    identity_quality TEXT NOT NULL DEFAULT 'source'
);
CREATE TABLE IF NOT EXISTS team_identity (
    team_id TEXT PRIMARY KEY,
    display_name TEXT,
    identity_quality TEXT NOT NULL DEFAULT 'source'
);
CREATE TABLE IF NOT EXISTS game (
    game_id TEXT PRIMARY KEY,
    game_date TEXT,
    season_id TEXT,
    competition_phase TEXT NOT NULL,
    raw_game_type TEXT,
    raw_game_label TEXT,
    raw_game_sub_label TEXT,
    season_status TEXT NOT NULL,
    phase_status TEXT NOT NULL,
    home_team_id TEXT,
    away_team_id TEXT,
    home_score INTEGER,
    away_score INTEGER,
    source_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshot(snapshot_id)
);
CREATE TABLE IF NOT EXISTS game_team_participation (
    game_id TEXT NOT NULL REFERENCES game(game_id),
    team_id TEXT NOT NULL,
    opponent_team_id TEXT NOT NULL,
    home_away TEXT NOT NULL,
    team_score INTEGER,
    opponent_score INTEGER,
    win INTEGER,
    participation_quality TEXT NOT NULL,
    source_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshot(snapshot_id),
    PRIMARY KEY (game_id, team_id)
);
CREATE TABLE IF NOT EXISTS player_game_stat (
    source_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshot(snapshot_id),
    source_row_id INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES game(game_id),
    person_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    opponent_team_id TEXT NOT NULL,
    appearance_status TEXT NOT NULL,
    minutes_quality TEXT NOT NULL,
    raw_comment TEXT NOT NULL,
    points INTEGER,
    assists INTEGER,
    rebounds_total INTEGER,
    field_goals_made INTEGER,
    three_pointers_made INTEGER,
    field_goals_attempted INTEGER,
    raw_record_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_snapshot_id, source_row_id)
);
CREATE TABLE IF NOT EXISTS team_game_stat (
    source_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshot(snapshot_id),
    source_row_id INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES game(game_id),
    team_id TEXT NOT NULL,
    opponent_team_id TEXT NOT NULL,
    team_score INTEGER,
    opponent_score INTEGER,
    home_away TEXT NOT NULL,
    win INTEGER,
    assists INTEGER,
    rebounds_total INTEGER,
    field_goals_made INTEGER,
    three_pointers_made INTEGER,
    field_goals_attempted INTEGER,
    raw_record_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_snapshot_id, source_row_id)
);

CREATE INDEX IF NOT EXISTS idx_player_game_scope
    ON player_game_stat(person_id, game_id, appearance_status);
CREATE INDEX IF NOT EXISTS idx_team_game_scope
    ON team_game_stat(game_id, team_id);

CREATE VIEW IF NOT EXISTS player_season_stat AS
SELECT p.person_id, g.season_id, g.competition_phase,
       COUNT(*) AS games_appeared,
       SUM(p.points) AS points_total,
       SUM(p.assists) AS assists_total,
       SUM(p.rebounds_total) AS rebounds_total,
       SUM(p.field_goals_made) AS field_goals_made,
       SUM(p.three_pointers_made) AS three_pointers_made,
       SUM(p.field_goals_attempted) AS field_goals_attempted
FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
WHERE p.appearance_status IN ('played', 'played_inferred')
GROUP BY p.person_id, g.season_id, g.competition_phase;

CREATE VIEW IF NOT EXISTS team_season_stat AS
SELECT t.team_id, g.season_id, g.competition_phase,
       COUNT(*) AS games_played,
       SUM(t.team_score) AS points_total,
       SUM(t.assists) AS assists_total,
       SUM(t.rebounds_total) AS rebounds_total
FROM team_game_stat t JOIN game g ON g.game_id = t.game_id
WHERE t.team_id <> ''
GROUP BY t.team_id, g.season_id, g.competition_phase;
"""


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT OR IGNORE INTO dataset_snapshot VALUES (?, ?, ?, ?)",
        (SNAPSHOT_ID, SOURCE_SNAPSHOT.split("@")[0], "515", datetime.now(UTC).isoformat()),
    )
    connection.commit()


def _int(value: object) -> int | None:
    status, parsed = parse_count(str(value or ""))
    return int(parsed) if status == "valid" and parsed is not None else None


def _canonical_game_id(value: str) -> str:
    value = str(value).strip()
    return value.lstrip("0") or "0"


def _insert_game(connection: sqlite3.Connection, row: dict) -> None:
    connection.execute(
        """INSERT OR REPLACE INTO game
        (game_id, game_date, season_id, competition_phase, raw_game_type,
         raw_game_label, raw_game_sub_label, season_status, phase_status,
         home_team_id, away_team_id, home_score, away_score, source_snapshot_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row["game_id"], row.get("game_date"), row.get("season_id"),
            row.get("competition_phase", "unknown"), row.get("raw_game_type"),
            row.get("raw_game_label", ""), row.get("raw_game_sub_label", ""),
            row.get("season_status", "unknown"), row.get("phase_status", "unknown"),
            row.get("home_team_id"), row.get("away_team_id"),
            row.get("home_score"), row.get("away_score"), SNAPSHOT_ID,
        ),
    )


def insert_fixture_rows(
    connection: sqlite3.Connection,
    *,
    games: Iterable[dict],
    players: Iterable[dict],
    teams: Iterable[dict],
) -> None:
    """Insert already-conformed rows, primarily for deterministic tests."""
    create_schema(connection)
    for row in games:
        _insert_game(connection, row)
    for row in players:
        person_id = str(row["person_id"])
        connection.execute(
            "INSERT OR IGNORE INTO player_identity VALUES (?, ?, ?)",
            (person_id, row.get("display_name", person_id), "fixture"),
        )
        connection.execute(
            """INSERT INTO player_game_stat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (SNAPSHOT_ID, row["source_row_id"], row["game_id"], person_id,
             str(row.get("team_id", "")), str(row.get("opponent_team_id", "")),
             row["appearance_status"], row["minutes_quality"], row.get("raw_comment", ""),
             row.get("points"), row.get("assists"), row.get("rebounds_total"),
             row.get("field_goals_made"), row.get("three_pointers_made"),
             row.get("field_goals_attempted"), row.get("raw_record_json", "{}")),
        )
    for row in teams:
        connection.execute(
            "INSERT OR IGNORE INTO team_identity VALUES (?, ?, ?)",
            (str(row["team_id"]), row.get("team_name", row["team_id"]), "fixture"),
        )
        connection.execute(
            """INSERT INTO team_game_stat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (SNAPSHOT_ID, row["source_row_id"], row["game_id"], row["team_id"],
             row["opponent_team_id"], row.get("team_score"), row.get("opponent_score"),
             row.get("home_away", "unknown"), row.get("win"), row.get("assists"),
             row.get("rebounds_total"), row.get("field_goals_made"),
             row.get("three_pointers_made"), row.get("field_goals_attempted"),
             row.get("raw_record_json", "{}")),
        )
        connection.execute(
            """INSERT OR REPLACE INTO game_team_participation
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row["game_id"], row["team_id"], row["opponent_team_id"],
             row.get("home_away", "unknown"), row.get("team_score"),
             row.get("opponent_score"), row.get("win"), "valid", SNAPSHOT_ID),
        )
    connection.commit()


def _read_csv(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source_row_id, row in enumerate(csv.DictReader(handle), start=2):
            yield source_row_id, row


def _register_file(connection: sqlite3.Connection, path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    connection.execute(
        "INSERT INTO source_file VALUES (?, ?, ?, NULL)",
        (SNAPSHOT_ID, path.name, digest.hexdigest()),
    )


def _source_rows(connection: sqlite3.Connection, path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    _register_file(connection, path)
    count = 0
    for source_row_id, row in _read_csv(path):
        connection.execute(
            "INSERT INTO source_record VALUES (?, ?, ?, ?)",
            (SNAPSHOT_ID, path.name, source_row_id, json.dumps(row, ensure_ascii=False, sort_keys=True)),
        )
        count += 1
        yield source_row_id, row
    connection.execute(
        "UPDATE source_file SET row_count = ? WHERE snapshot_id = ? AND file_name = ?",
        (count, SNAPSHOT_ID, path.name),
    )


def _aliases(value: str) -> set[str]:
    raw = str(value).strip()
    if not raw:
        return set()
    return {raw, raw.zfill(10), _canonical_game_id(raw)}


def _resolve_game(value: str, aliases: dict[str, str]) -> str | None:
    for alias in _aliases(value):
        if alias in aliases:
            return aliases[alias]
    return None


def build_database(raw_dir: Path = RAW, db_path: Path | str = ":memory:") -> sqlite3.Connection:
    """Load the five M1 files into a conformed SQLite database."""
    connection = sqlite3.connect(str(db_path))
    create_schema(connection)
    aliases: dict[str, str] = {}
    game_rows: dict[str, dict] = {}
    for _, raw in _source_rows(connection, raw_dir / "Games.csv"):
        classified = classify_game({"gameId": raw["gameId"], "gameDate": raw.get("gameDate", ""), "gameType": raw.get("gameType", ""), "gameLabel": raw.get("gameLabel", ""), "gameSubLabel": raw.get("gameSubLabel", "")})
        canonical = _canonical_game_id(raw["gameId"])
        game = {
            "game_id": canonical,
            "game_date": raw.get("gameDate", "")[:10],
            **classified,
            "home_team_id": raw.get("hometeamId", ""),
            "away_team_id": raw.get("awayteamId", ""),
            "home_score": _int(raw.get("homeScore")),
            "away_score": _int(raw.get("awayScore")),
        }
        game_rows[canonical] = game
        for alias in _aliases(raw["gameId"]):
            aliases[alias] = canonical
        _insert_game(connection, game)
        for team_id, name in ((raw.get("hometeamId", ""), raw.get("hometeamName", "")), (raw.get("awayteamId", ""), raw.get("awayteamName", ""))):
            if team_id:
                connection.execute("INSERT OR IGNORE INTO team_identity VALUES (?, ?, ?)", (team_id, name, "game_source"))

    for source_row_id, raw in _source_rows(connection, raw_dir / "Players.csv"):
        del source_row_id
        person_id = raw.get("personId", "")
        if person_id:
            connection.execute("INSERT OR IGNORE INTO player_identity VALUES (?, ?, ?)", (person_id, f"{raw.get('firstName', '')} {raw.get('lastName', '')}".strip(), "source"))

    for source_row_id, raw in _source_rows(connection, raw_dir / "TeamHistories.csv"):
        del source_row_id
        team_id = raw.get("teamId", "")
        if team_id:
            connection.execute("INSERT OR IGNORE INTO team_identity VALUES (?, ?, ?)", (team_id, f"{raw.get('teamCity', '')} {raw.get('teamName', '')}".strip(), "history"))

    for source_row_id, raw in _source_rows(connection, raw_dir / "PlayerStatistics.csv"):
        game_id = _resolve_game(raw.get("gameId", ""), aliases)
        if not game_id or game_id not in game_rows:
            continue
        quality = classify_player(raw)
        connection.execute(
            """INSERT INTO player_game_stat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (SNAPSHOT_ID, source_row_id, game_id, raw.get("personId", ""),
             raw.get("playerteamId", ""), raw.get("opponentteamId", ""),
             quality["appearance_status"], quality["minutes_status"], raw.get("comment", ""),
             _int(raw.get("points")), _int(raw.get("assists")), _int(raw.get("reboundsTotal")),
             _int(raw.get("fieldGoalsMade")), _int(raw.get("threePointersMade")), _int(raw.get("fieldGoalsAttempted")),
             json.dumps(raw, ensure_ascii=False, sort_keys=True)),
        )

    for source_row_id, raw in _source_rows(connection, raw_dir / "TeamStatistics.csv"):
        game_id = _resolve_game(raw.get("gameId", ""), aliases)
        if not game_id or game_id not in game_rows:
            continue
        team_id = raw.get("teamId", "")
        opponent = raw.get("opponentTeamId", "")
        home_away = "home" if raw.get("home") == "1" else "away"
        connection.execute(
            """INSERT INTO team_game_stat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (SNAPSHOT_ID, source_row_id, game_id, team_id, opponent, _int(raw.get("teamScore")),
             _int(raw.get("opponentScore")), home_away, _int(raw.get("win")), _int(raw.get("assists")),
             _int(raw.get("reboundsTotal")), _int(raw.get("fieldGoalsMade")),
             _int(raw.get("threePointersMade")), _int(raw.get("fieldGoalsAttempted")),
             json.dumps(raw, ensure_ascii=False, sort_keys=True)),
        )
        connection.execute("INSERT OR IGNORE INTO team_identity VALUES (?, ?, ?)", (team_id, f"{raw.get('teamCity', '')} {raw.get('teamName', '')}".strip(), "fact_source"))
        if team_id and opponent:
            game = game_rows[game_id]
            expected = {game.get("home_team_id"), game.get("away_team_id")}
            quality = "valid" if {team_id, opponent} == expected else "team_id_mismatch"
            connection.execute(
                """INSERT OR REPLACE INTO game_team_participation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (game_id, team_id, opponent, home_away, _int(raw.get("teamScore")),
                 _int(raw.get("opponentScore")), _int(raw.get("win")), quality, SNAPSHOT_ID),
            )
    connection.commit()
    return connection


def _lineage() -> dict[str, str]:
    return {"source_snapshot_id": SNAPSHOT_ID, "source_snapshot": SOURCE_SNAPSHOT}


def _capability_status(season_id: str | None, phase: str | None, subject: str, metric: str) -> str | None:
    if not season_id or not phase or not CAPABILITY_MATRIX.exists():
        return None
    matrix = json.loads(CAPABILITY_MATRIX.read_text(encoding="utf-8"))
    period = matrix.get("periods", {}).get(f"{season_id}|{phase}", {})
    target = period.get("subjects", {}).get(subject, {})
    if metric == "team_participation":
        return target.get("participation")
    if metric == "team_opponent_filter":
        return target.get("team_opponent_filter")
    return target.get("metrics", {}).get(metric)


def _require_supported(status: str | None, label: str) -> None:
    if status == "unsupported":
        raise ValueError(f"{label} is unsupported for this season and phase")


def query_game_score(connection: sqlite3.Connection, game_id: str) -> dict:
    game = connection.execute(
        "SELECT season_id, competition_phase FROM game WHERE game_id = ?",
        (_canonical_game_id(game_id),),
    ).fetchone()
    if game is None:
        raise ValueError("game not found")
    _require_supported(_capability_status(game[0], game[1], "team", "team_participation"), "team participation")
    rows = connection.execute(
        "SELECT * FROM game_team_participation WHERE game_id = ? ORDER BY home_away",
        (_canonical_game_id(game_id),),
    ).fetchall()
    if len(rows) != 2 or any(row[7] != "valid" for row in rows):
        raise ValueError("game requires two complete team rows")
    by_side = {row[3]: row for row in rows}
    if set(by_side) != {"home", "away"}:
        raise ValueError("game requires home and away rows")
    expected = connection.execute(
        "SELECT home_score, away_score FROM game WHERE game_id = ?",
        (_canonical_game_id(game_id),),
    ).fetchone()
    if (by_side["home"][4], by_side["away"][4]) != expected:
        raise ValueError("score mismatch between game and team facts")
    return {
        "status": "ok", "game_id": _canonical_game_id(game_id),
        "home_score": by_side["home"][4], "away_score": by_side["away"][4],
        "team_row_count": len(rows), "lineage": _lineage(),
    }


def query_player_season(
    connection: sqlite3.Connection,
    person_id: str,
    season_id: str,
    competition_phase: str,
    *,
    opponent_team_id: str | None = None,
    last_n: int | None = None,
) -> dict:
    _require_supported(_capability_status(season_id, competition_phase, "player", "points"), "player points")
    if opponent_team_id is not None:
        _require_supported(
            _capability_status(season_id, competition_phase, "player", "team_opponent_filter"),
            "team/opponent filter",
        )
    sql = """SELECT p.*, g.game_date, g.season_status, g.phase_status
             FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
             WHERE p.person_id = ? AND g.season_id = ? AND g.competition_phase = ?
             AND p.appearance_status IN ('played', 'played_inferred')"""
    params: list[object] = [person_id, season_id, competition_phase]
    sql += " ORDER BY g.game_date, p.game_id"
    rows = connection.execute(sql, params).fetchall()
    columns = [column[0] for column in connection.execute("SELECT * FROM player_game_stat LIMIT 0").description]
    indexes = {name: columns.index(name) for name in columns}
    if opponent_team_id is not None:
        if any(not row[indexes["team_id"]] or not row[indexes["opponent_team_id"]] for row in rows):
            raise ValueError("team context is incomplete for opponent filter")
        for row in rows:
            participation = connection.execute(
                "SELECT participation_quality FROM game_team_participation WHERE game_id = ? AND team_id = ?",
                (row[indexes["game_id"]], row[indexes["team_id"]]),
            ).fetchone()
            if participation is None or participation[0] != "valid":
                raise ValueError("team context is incomplete for opponent filter")
        rows = [row for row in rows if row[indexes["opponent_team_id"]] == opponent_team_id]
    duplicate = connection.execute(
        """SELECT p.game_id, COUNT(*) FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
           WHERE p.person_id = ? AND g.season_id = ? AND g.competition_phase = ?
             AND p.appearance_status IN ('played', 'played_inferred')
           GROUP BY p.game_id HAVING COUNT(*) > 1""",
        (person_id, season_id, competition_phase),
    ).fetchone()
    if duplicate:
        raise ValueError("duplicate player game rows")
    if last_n is not None:
        rows = rows[-last_n:]
    if not rows:
        return {"status": "no_data", "person_id": person_id, "lineage": _lineage()}

    def total(name: str) -> int | None:
        values = [row[indexes[name]] for row in rows]
        return sum(values) if all(value is not None for value in values) else None

    points = total("points")
    fgm, threes, fga = total("field_goals_made"), total("three_pointers_made"), total("field_goals_attempted")
    limitations: list[str] = []
    if fgm is None or threes is None or fga is None:
        efg = None
        limitations.append("efg_inputs_missing")
    elif fga == 0:
        efg = None
        limitations.append("efg_zero_denominator")
    else:
        efg = (fgm + 0.5 * threes) / fga * 100
    if any(row[indexes["appearance_status"]] == "played_inferred" for row in rows):
        limitations.append("inferred_appearance_included")
    if _capability_status(season_id, competition_phase, "player", "efg_percentage") == "conditional":
        limitations.append("historical_shooting_evidence_conditional")
    result = {
        "status": "ok", "person_id": person_id, "season_id": season_id,
        "competition_phase": competition_phase, "games_appeared": len(rows),
        "points_total": points, "points_per_appearance": points / len(rows) if points is not None else None,
        "assists_total": total("assists"), "rebounds_total": total("rebounds_total"),
        "field_goals_made": fgm, "three_pointers_made": threes, "field_goals_attempted": fga,
        "efg_percentage": efg, "quality_status": "conditional" if limitations else "validated",
        "limitations": limitations, "lineage": _lineage(),
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the M1 SQLite baseline")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    build_database(args.raw_dir, args.db).close()
    print(f"Built {args.db}")
