import asyncio
import json
import sqlite3
import time
from pathlib import Path

from httpx import ASGITransport, AsyncClient
import pytest

from scripts.model_adapter import ModelRequest, ModelResponse
from scripts.relational_baseline import create_schema, insert_fixture_rows

TEST_MATRIX = Path(__file__).parent / "fixtures/capability_matrix_v1.json"


class StaticSQLModel:
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.sql, "test-model")


def post_query(app: object, question: str):
    async def send_request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/query", json={"question": question})

    return asyncio.run(send_request())


def post_compare(app: object, payload: dict):
    async def send_request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/calculate/compare", json=payload)

    return asyncio.run(send_request())


def make_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    create_schema(connection)
    insert_fixture_rows(
        connection,
        games=[{
            "game_id": "22300061",
            "game_date": "2023-10-24",
            "season_id": "2023-24",
            "competition_phase": "regular_season",
            "home_team_id": "1610612747",
            "away_team_id": "1610612743",
            "home_score": 119,
            "away_score": 107,
            "season_status": "valid",
            "phase_status": "valid",
        }],
        players=[],
        teams=[],
    )
    connection.close()


def make_comparison_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    from scripts.relational_baseline import insert_fixture_rows

    insert_fixture_rows(
        connection,
        games=[
            {
                "game_id": "22300061", "game_date": "2023-10-24",
                "season_id": "2023-24", "competition_phase": "regular_season",
                "home_team_id": "1610612747", "away_team_id": "1610612743",
                "home_score": 119, "away_score": 107,
                "season_status": "valid", "phase_status": "valid",
            },
            {
                "game_id": "22400061", "game_date": "2024-10-22",
                "season_id": "2024-25", "competition_phase": "regular_season",
                "home_team_id": "1610612747", "away_team_id": "1610612743",
                "home_score": 120, "away_score": 110,
                "season_status": "valid", "phase_status": "valid",
            },
        ],
        players=[],
        teams=[
            {"source_row_id": 1, "game_id": "22300061", "team_id": "1610612747",
             "opponent_team_id": "1610612743", "team_score": 100, "opponent_score": 90,
             "home_away": "home", "win": 1, "assists": 20, "rebounds_total": 40},
            {"source_row_id": 2, "game_id": "22400061", "team_id": "1610612747",
             "opponent_team_id": "1610612743", "team_score": 120, "opponent_score": 110,
             "home_away": "home", "win": 1, "assists": 22, "rebounds_total": 42},
        ],
    )
    connection.close()


def test_query_api_generates_executes_and_returns_structured_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    sql = "SELECT game_id, home_score, away_score FROM game WHERE game_id = '22300061'"
    model = StaticSQLModel(sql)

    from app.main import create_app

    response = post_query(
        create_app(model=model, db_path=db_path),
        "湖人对掘金那场比赛双方得分是多少？",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sql"] == sql
    assert body["columns"] == ["game_id", "home_score", "away_score"]
    assert body["rows"] == [["22300061", 119, 107]]
    assert body["evidence"] == {
        "source_name": "eoinamoore/historical-nba-data-and-player-box-scores",
        "source_version": "515",
        "snapshot_id": "eoin-v515",
        "catalog_version": "0.5.2",
    }
    assert body["elapsed_ms"] >= 0
    assert body["model_elapsed_ms"] >= 0
    assert body["gateway_elapsed_ms"] >= 0
    assert body["elapsed_ms"] >= body["model_elapsed_ms"] + body["gateway_elapsed_ms"] - 0.002
    assert "answer" not in body
    assert len(model.requests) == 1
    assert "湖人对掘金" in model.requests[0].prompt
    assert '"game"' in model.requests[0].prompt
    assert body["execution_mode"] == "direct"
    assert body["context_mode"] == "schema_value_retrieval"


def test_calculate_compare_returns_source_values_delta_and_quality(tmp_path: Path) -> None:
    db_path = tmp_path / "comparison.sqlite"
    make_comparison_database(db_path)

    from app.main import create_app

    response = post_compare(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {
            "subject": "team",
            "entity_id": "1610612747",
            "metric": "points_total",
            "baseline_season": "2023-24",
            "comparison_season": "2024-25",
            "competition_phase": "regular_season",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["baseline"]["value"] == 100
    assert body["comparison"]["value"] == 120
    assert body["absolute_delta"] == 20
    assert body["relative_change_pct"] == 20.0
    assert body["quality_status"] == "conditional"
    assert body["limitations"] == ["game_count_mismatch", "capability_evidence_conditional"]
    assert body["evidence"]["source_snapshot"] == "eoinamoore/historical-nba-data-and-player-box-scores@515"


def test_calculate_compare_returns_partial_for_zero_baseline(tmp_path: Path) -> None:
    db_path = tmp_path / "comparison.sqlite"
    make_comparison_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE team_game_stat SET team_score = 0 WHERE game_id = '22300061'"
        )
        connection.commit()

    from app.main import create_app

    body = post_compare(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {
            "subject": "team", "entity_id": "1610612747", "metric": "points_total",
            "baseline_season": "2023-24", "comparison_season": "2024-25",
            "competition_phase": "regular_season",
        },
    ).json()
    assert body["status"] == "partial"
    assert body["absolute_delta"] == 120
    assert body["relative_change_pct"] is None
    assert body["relative_change_status"] == "not_computable"
    assert body["calculation_limitation"] == "zero_baseline"


def test_calculate_compare_rejects_missing_requested_team_score(tmp_path: Path) -> None:
    db_path = tmp_path / "comparison.sqlite"
    make_comparison_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE team_game_stat SET team_score = NULL WHERE game_id = '22300061'")
        connection.commit()

    from app.main import create_app

    body = post_compare(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {
            "subject": "team", "entity_id": "1610612747", "metric": "points_total",
            "baseline_season": "2023-24", "comparison_season": "2024-25",
            "competition_phase": "regular_season",
        },
    ).json()

    assert body["status"] == "not_computable"
    assert body["reason_code"] == "comparison_inputs_missing"
    assert body["absolute_delta"] is None


def test_calculate_compare_rejects_mismatched_phases(tmp_path: Path) -> None:
    db_path = tmp_path / "comparison.sqlite"
    make_comparison_database(db_path)
    from app.main import create_app

    response = post_compare(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {
            "subject": "team", "entity_id": "1610612747", "metric": "points_total",
            "baseline_season": "2023-24", "comparison_season": "2024-25",
            "competition_phase": "playoffs",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "not_computable"
    assert "no team observation" in response.json()["reason"]


def test_calculate_compare_rejects_unsupported_metric_capability(tmp_path: Path) -> None:
    db_path = tmp_path / "comparison.sqlite"
    make_comparison_database(db_path)
    matrix = json.loads(TEST_MATRIX.read_text(encoding="utf-8"))
    matrix["periods"]["2023-24|regular_season"]["subjects"]["team"]["metrics"]["points"] = "unsupported"
    matrix_path = tmp_path / "unsupported-matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    from app.main import create_app

    body = post_compare(
        create_app(db_path=db_path, quality_matrix_path=matrix_path),
        {
            "subject": "team", "entity_id": "1610612747", "metric": "points_total",
            "baseline_season": "2023-24", "comparison_season": "2024-25",
            "competition_phase": "regular_season",
        },
    ).json()
    assert body["status"] == "not_computable"
    assert "unsupported:points" in body["reason"]


def test_calculate_compare_rejects_metric_subject_mismatch_at_boundary(tmp_path: Path) -> None:
    db_path = tmp_path / "comparison.sqlite"
    make_comparison_database(db_path)
    from app.main import create_app

    response = post_compare(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {
            "subject": "team", "entity_id": "1610612747", "metric": "points_per_appearance",
            "baseline_season": "2023-24", "comparison_season": "2024-25",
            "competition_phase": "regular_season",
        },
    )
    assert response.status_code == 422
    assert "not supported for subject team" in response.json()["detail"]


def test_query_api_rejects_cross_season_scope_before_model_call(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    model = StaticSQLModel("SELECT home_score FROM game")

    from app.main import create_app

    response = post_query(
        create_app(model=model, db_path=db_path),
        "比较 2022-23 和 2023-24 常规赛湖人的总得分？",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unsupported_scope"
    assert response.json()["result_status"] == "not_executed"
    assert response.json()["reason"] == "cross-season queries are deferred to M1.9"
    assert model.requests == []


def test_query_api_rejects_ambiguous_entity_before_model_call(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            [("1", "Alex Smith", "valid"), ("2", "Alex Smith", "valid")],
        )
        connection.commit()
    model = StaticSQLModel("SELECT points_total FROM player_season_stat")

    from app.main import create_app

    response = post_query(
        create_app(model=model, db_path=db_path),
        "Alex Smith 在 2023-24 常规赛得分？",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ambiguous_entity"
    assert response.json()["result_status"] == "not_executed"
    assert response.json()["resolved_values"]["ambiguous_entities"] == ["player:Alex Smith"]
    assert model.requests == []


def test_query_direct_rejects_sql_for_another_grounded_entity(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("201939", "Stephen Curry", "valid"),
        )
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("203081", "Damian Lillard", "valid"),
        )
        connection.execute(
            """INSERT INTO player_game_stat
               (source_snapshot_id, source_row_id, game_id, person_id, team_id, opponent_team_id,
                appearance_status, minutes_quality, raw_comment, points, field_goals_made,
                three_pointers_made, field_goals_attempted, raw_record_json)
               VALUES ('eoin-v515', 90, '22300061', '203081', '1610612747', '1610612743',
                       'played', 'positive', '', 12, 4, 1, 10, '{}')"""
        )
        connection.commit()

    model = StaticSQLModel(
        "SELECT points_total FROM player_season_stat "
        "WHERE person_id = '203081' AND season_id = '2023-24' "
        "AND competition_phase = 'regular_season'"
    )
    from app.main import create_app

    response = post_query(
        create_app(model=model, db_path=db_path, quality_matrix_path=TEST_MATRIX),
        "库里 2023-24 常规赛总得分？",
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "plan_invalid"
    assert body["result_status"] == "not_executed"
    assert body["plan_error_code"] == "unverified_sql_entity"
    assert body["rows"] == []


def test_query_direct_rejects_union_before_gateway_execution(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("201939", "Stephen Curry", "valid"),
        )
        connection.commit()
    model = StaticSQLModel(
        "SELECT points_total FROM player_season_stat "
        "WHERE person_id = '201939' AND season_id = '2023-24' AND competition_phase = 'regular_season' "
        "UNION SELECT points_total FROM player_season_stat "
        "WHERE person_id = '2544' AND season_id = '2023-24' AND competition_phase = 'regular_season'"
    )
    from app.main import create_app

    body = post_query(
        create_app(model=model, db_path=db_path, quality_matrix_path=TEST_MATRIX),
        "库里 2023-24 常规赛总得分？",
    ).json()
    assert body["status"] == "plan_invalid"
    assert body["result_status"] == "not_executed"
    assert body["plan_error_code"] == "unverified_sql_shape"
    assert body["rows"] == []


@pytest.mark.parametrize("sql", [
    "WITH scoped AS (SELECT points_total FROM player_season_stat WHERE person_id = '201939') SELECT points_total FROM scoped",
    "SELECT points_total FROM player_season_stat WHERE person_id IN (SELECT person_id FROM player_identity)",
    "SELECT points_total FROM player_season_stat WHERE person_id = '201939'; SELECT points_total FROM player_season_stat WHERE person_id = '201939'",
])
def test_query_direct_rejects_unverifiable_nested_shapes(tmp_path: Path, sql: str) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("201939", "Stephen Curry", "valid"),
        )
        connection.commit()
    from app.main import create_app

    body = post_query(
        create_app(model=StaticSQLModel(sql), db_path=db_path, quality_matrix_path=TEST_MATRIX),
        "库里 2023-24 常规赛总得分？",
    ).json()
    assert body["status"] == "plan_invalid"
    assert body["plan_error_code"] == "unverified_sql_shape"
    assert body["rows"] == []


def test_query_api_exposes_context_mode_and_prompt_size(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    from app.main import create_app

    model = StaticSQLModel("SELECT home_score FROM game WHERE game_id = '22300061'")
    body = post_query(create_app(model=model, db_path=db_path, context_mode="schema_retrieval"),
                      "2023-10-24 湖人对掘金比分？").json()

    assert body["status"] == "ok"
    assert body["context_mode"] == "schema_retrieval"
    assert body["prompt_chars"] == len(model.requests[0].prompt)
    assert "player_game_stat" not in model.requests[0].prompt
    assert body["context_elapsed_ms"] >= 0


def test_query_api_plan_mode_compiles_and_executes_validated_plan(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)

    class PlanModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                '{"query_family":"single_game_score","game_id":"22300061",'
                '"requested_metrics":["home_score","away_score"]}', "plan-model"
            )

    from app.main import create_app
    body = post_query(
        create_app(model=PlanModel(), db_path=db_path, execution_mode="plan", quality_matrix_path=TEST_MATRIX),
        "2023-10-24 湖人对掘金比分？",
    ).json()
    assert body["status"] == "ok"
    assert body["execution_mode"] == "plan"
    assert body["plan"]["query_family"] == "single_game_score"
    assert body["sql"] == "SELECT home_score, away_score FROM game WHERE game_id = '22300061'"
    assert body["rows"] == [[119, 107]]
    assert body["quality_status"] == "validated"
    assert body["limitations"] == []


def test_direct_and_plan_routes_share_playoff_quality_disclosure(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE game SET competition_phase = 'playoffs' WHERE game_id = '22300061'"
        )
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("203999", "Nikola Jokic", "valid"),
        )
        connection.execute(
            """INSERT INTO player_game_stat
               (source_snapshot_id, source_row_id, game_id, person_id, team_id, opponent_team_id,
                appearance_status, minutes_quality, raw_comment, points, raw_record_json)
               VALUES ('eoin-v515', 1, '22300061', '203999', '1610612743', '1610612747',
                       'played', 'positive', '', 344, '{}')"""
        )
        connection.commit()

    question = "约基奇在 2023-24 季后赛总得分和出场次数是多少？"
    sql = """SELECT points_total, games_appeared FROM player_season_stat
             WHERE person_id = '203999' AND season_id = '2023-24'
               AND competition_phase = 'playoffs'"""

    from app.main import create_app

    direct = post_query(
        create_app(model=StaticSQLModel(sql), db_path=db_path, quality_matrix_path=TEST_MATRIX), question
    ).json()

    class PlanModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                '{"query_family":"player_season_summary","player_id":"203999",'
                '"season_id":"2023-24","competition_phase":"playoffs",'
                '"requested_metrics":["points_total","games_appeared"]}', "plan-model"
            )

    plan = post_query(
        create_app(model=PlanModel(), db_path=db_path, execution_mode="plan", quality_matrix_path=TEST_MATRIX),
        question,
    ).json()

    assert direct["rows"] == plan["rows"] == [[344, 1]]
    assert direct["quality_status"] == plan["quality_status"] == "conditional"
    assert direct["limitations"] == plan["limitations"] == [
        "season_completeness_not_checked", "capability_evidence_conditional"
    ]


def test_direct_route_rejects_unsupported_quality_before_gateway(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    matrix = json.loads(TEST_MATRIX.read_text(encoding="utf-8"))
    matrix["periods"]["2023-24|regular_season"]["subjects"]["player"]["metrics"]["points"] = "unsupported"
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    sql = """SELECT points_total, games_appeared FROM player_season_stat
             WHERE person_id = '201939' AND season_id = '2023-24'
               AND competition_phase = 'regular_season'"""

    from app.main import create_app

    body = post_query(
        create_app(model=StaticSQLModel(sql), db_path=db_path, quality_matrix_path=matrix_path),
        "库里在 2023-24 常规赛总得分和出场次数？",
    ).json()

    assert body["status"] == "unsupported_data"
    assert body["result_status"] == "not_executed"
    assert body["sql"] == ""
    assert body["limitations"] == ["unsupported:points"]


def test_plan_route_discloses_unsupported_quality_before_gateway(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    matrix = json.loads(TEST_MATRIX.read_text(encoding="utf-8"))
    matrix["periods"]["2023-24|regular_season"]["subjects"]["player"]["metrics"]["points"] = "unsupported"
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    class UnsupportedPlanModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(json.dumps({
                "query_family": "player_season_summary", "player_id": "201939",
                "season_id": "2023-24", "competition_phase": "regular_season",
                "requested_metrics": ["points_total"],
            }), "plan-model")

    from app.main import create_app

    body = post_query(create_app(
        model=UnsupportedPlanModel(), db_path=db_path,
        execution_mode="plan", quality_matrix_path=matrix_path,
    ), "库里在 2023-24 常规赛总得分？").json()

    assert body["status"] == "unsupported_data"
    assert body["result_status"] == "not_executed"
    assert body["sql"] == ""
    assert body["quality_status"] == "unsupported"
    assert body["limitations"] == ["unsupported:points"]


def test_query_api_plan_failure_does_not_execute_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)

    class InvalidPlanModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse('{"query_family":"single_game_score","game_id":123,', "plan-model")

    from app.main import create_app
    body = post_query(
        create_app(model=InvalidPlanModel(), db_path=db_path, execution_mode="plan"),
        "2023-10-24 湖人对掘金比分？",
    ).json()
    assert body["status"] == "plan_invalid"
    assert body["result_status"] == "not_executed"
    assert body["sql"] == ""
    assert body["rows"] == []
    assert body["plan_error_code"] == "invalid_plan"


def test_query_api_rejects_ambiguous_game_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """INSERT INTO game (game_id, game_date, season_id, competition_phase,
               season_status, phase_status, home_team_id, away_team_id, home_score,
               away_score, source_snapshot_id)
               VALUES ('22300062', '2023-10-24', '2023-24', 'regular_season',
               'valid', 'valid', '1610612747', '1610612743', 100, 99, 'eoin-v515')"""
        )
        connection.commit()

    class AmbiguousPlanModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                '{"query_family":"single_game_score","game_id":"22300061",'
                '"requested_metrics":["home_score","away_score"]}', "plan-model"
            )

    from app.main import create_app
    body = post_query(
        create_app(model=AmbiguousPlanModel(), db_path=db_path, execution_mode="plan"),
        "2023-10-24 双方各得多少分？",
    ).json()

    assert body["status"] == "plan_invalid"
    assert body["plan_error_code"] == "ambiguous_game"
    assert body["sql"] == ""
    assert body["rows"] == []


def test_query_api_traces_exact_prompt_sql_and_bounded_result(tmp_path: Path) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = StaticSQLModel("SELECT home_score FROM game WHERE game_id = '22300061'")
    from app.main import create_app

    body = post_query(create_app(model=model, db_path=db_path, tracer=provider.get_tracer("test")),
                      "2023-10-24 湖人对掘金比分？").json()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert body["status"] == "ok"
    assert spans["nba.model"].attributes["input.value"] == model.requests[0].prompt
    assert spans["nba.model"].attributes["output.value"] == model.sql
    assert spans["nba.gateway"].attributes["input.value"] == model.sql
    assert '119' in spans["nba.gateway"].attributes["output.value"]
    assert spans["nba.query"].attributes["nba.route"] == "direct_schema_value"
    assert spans["nba.model"].attributes["nba.route"] == "direct_schema_value"
    assert spans["nba.model"].attributes["nba.execution_mode"] == "direct"
    assert spans["nba.model"].attributes["nba.context_mode"] == "schema_value_retrieval"
    assert spans["nba.model"].attributes["nba.prompt_chars"] == len(model.requests[0].prompt)
    assert "Authorization" not in str(spans)


def test_query_api_separates_slow_model_from_gateway_time(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)

    class SlowModel(StaticSQLModel):
        def generate(self, request: ModelRequest) -> ModelResponse:
            time.sleep(0.025)
            return super().generate(request)

    model = SlowModel("SELECT game_id FROM game WHERE game_id = '22300061'")
    from app.main import create_app

    body = post_query(create_app(model=model, db_path=db_path), "查询这场比赛").json()

    assert body["status"] == "ok"
    assert body["model_elapsed_ms"] >= 20
    assert body["gateway_elapsed_ms"] < body["model_elapsed_ms"]
    assert body["elapsed_ms"] >= body["model_elapsed_ms"]


def test_query_api_returns_gateway_rejection_without_running_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    model = StaticSQLModel("UPDATE game SET home_score = 0")

    from app.main import create_app

    response = post_query(create_app(model=model, db_path=db_path), "将比赛比分更新为零")

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["rows"] == []
    assert response.json()["reason"]
    assert len(model.requests) == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT home_score FROM game").fetchone() == (119,)


def test_query_api_marks_empty_and_undefined_results(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    from app.main import create_app

    empty = post_query(create_app(model=StaticSQLModel(
        "SELECT game_id FROM game WHERE game_id = 'absent'"
    ), db_path=db_path), "查询不存在的比赛").json()
    undefined = post_query(create_app(model=StaticSQLModel(
        "SELECT AVG(home_score) AS average_score FROM game WHERE game_id = 'absent'"
    ), db_path=db_path), "查询不存在比赛的平均分").json()

    assert empty["status"] == "ok"
    assert empty["result_status"] == "empty"
    assert empty["result_reason"] == "query returned no rows"
    assert undefined["status"] == "ok"
    assert undefined["result_status"] == "contains_null"
    assert undefined["result_reason"] == "query returned NULL values"


def test_query_api_rejects_blank_question(tmp_path: Path) -> None:
    model = StaticSQLModel("SELECT game_id FROM game")

    from app.main import create_app

    response = post_query(create_app(model=model, db_path=tmp_path / "missing.sqlite"), "  ")

    assert response.status_code == 422
    assert model.requests == []


def test_query_api_reports_missing_database_before_calling_model(tmp_path: Path) -> None:
    model = StaticSQLModel("SELECT game_id FROM game")

    from app.main import create_app

    response = post_query(create_app(model=model, db_path=tmp_path / "missing.sqlite"), "查询一场比赛")

    assert response.status_code == 503
    assert response.json()["detail"] == "query database is unavailable"
    assert model.requests == []


def test_query_api_maps_model_failure_to_upstream_error(tmp_path: Path, caplog) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)

    class FailingModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError("private endpoint detail")

    from app.main import create_app

    response = post_query(create_app(model=FailingModel(), db_path=db_path), "查询一场比赛")

    assert response.status_code == 502
    assert response.json()["detail"] == "SQL generation failed"
    assert "private endpoint detail" not in response.text
    assert "private endpoint detail" not in caplog.text


def test_query_api_reports_missing_model_configuration(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    from app.main import create_app

    response = post_query(create_app(db_path=db_path), "查询一场比赛")

    assert response.status_code == 503
    assert response.json()["detail"] == "SQL model is not configured"


def test_query_api_rejects_database_snapshot_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE dataset_snapshot SET source_version = '999'")
    model = StaticSQLModel("SELECT game_id FROM game")

    from app.main import create_app

    response = post_query(create_app(model=model, db_path=db_path), "查询一场比赛")

    assert response.status_code == 503
    assert model.requests == []


def test_query_api_uses_database_path_from_explicit_config(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "fixture.sqlite"
    make_database(db_path)
    from scripts.provider_config import ProviderConfig
    config = ProviderConfig(database_path=str(db_path))
    model = StaticSQLModel("SELECT game_id FROM game")

    from app.main import create_app

    response = post_query(create_app(model=model, provider_config=config), "查询一场比赛")

    assert response.status_code == 200
    assert response.json()["rows"] == [["22300061"]]
