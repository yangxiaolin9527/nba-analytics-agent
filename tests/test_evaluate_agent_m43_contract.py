from scripts.evaluate_agent_m43_contract import run_evaluation, score_http_case


def _case(expected=None):
    return {"category": "answerable", "expected": expected or {
        "route": "structured", "status": "complete", "capability": "player_efg",
        "result": {"efg_percentage": 1.0},
    }}


def test_m43_scorer_requires_arguments_and_provenance():
    body = {
        "status": "complete", "route": "structured", "trace_id": "t1",
        "answer": "1",
        "steps": [{"capability": "player_efg", "status": "complete", "arguments": {}, "result": {"efg_percentage": 1.0},
                   "evidence": {"structured": {"source_sql": "SELECT", "snapshot_id": "s", "catalog_version": "c", "formula": "f", "units": {"x": "percent"}, "precision": {"rounding": "full_precision"}}}}],
        "evidence": [{"structured": {"source_sql": "SELECT", "snapshot_id": "s", "catalog_version": "c", "formula": "f", "units": {"x": "percent"}, "precision": {"rounding": "full_precision"}}}],
    }
    case = _case({
        "route": "structured", "status": "complete", "capability": "player_efg",
        "arguments": {"entity_id": "201939"}, "result": {"efg_percentage": 1.0},
    })
    scored = score_http_case(body, case, http_status=200, trace_parent_ok=True)
    assert scored["outcome"] == "failed"
    assert scored["checks"]["arguments"] is False


def test_m43_scorer_separates_runtime_from_capability():
    body = {"status": "incomplete", "route": "structured", "reason_code": "tool_timeout", "steps": []}
    scored = score_http_case(body, _case(), http_status=200, trace_parent_ok=False)
    assert scored["outcome"] == "provider_runtime"
    assert scored["provider_runtime"] is True
    assert scored["capability_denominator"] is False


def test_m43_scorer_rejects_malformed_http_response():
    scored = score_http_case({}, _case(), http_status=200, trace_parent_ok=False)
    assert scored["outcome"] == "evaluation_contract"
    assert scored["evaluation_contract"] is True


def test_m43_scorer_rejects_extra_route_arguments():
    body = {
        "status": "complete", "route": "structured", "trace_id": "t1", "answer": "总得分为 50。",
        "task_state": {"route": {"steps": [{"capability": "season_summary", "arguments": {
            "subject": "player", "entity_id": "201939", "season_id": "2023-24",
            "competition_phase": "regular_season", "metrics": ["points_total"],
            "opponent_team_id": "1610612747",
        }}]}},
        "steps": [{"capability": "season_summary", "status": "complete", "result": {"points_total": 50},
                   "evidence": {"structured": {"source_sql": "SELECT", "snapshot_id": "s", "catalog_version": "c", "precision": {"rounding": "full_precision"}}}}],
    }
    case = {"category": "answerable", "expected": {
        "route": "structured", "status": "complete", "capability": "season_summary",
        "arguments": {"subject": "player", "entity_id": "201939", "season_id": "2023-24",
                       "competition_phase": "regular_season", "metrics": ["points_total"]},
        "result": {"points_total": 50},
    }}
    scored = score_http_case(body, case, http_status=200, trace_parent_ok=True)
    assert scored["outcome"] == "failed"
    assert scored["checks"]["arguments"] is False


def test_m43_scorer_rejects_wrong_rendered_numeric_value():
    body = {
        "status": "complete", "route": "structured", "trace_id": "t1", "answer": "总得分为 999。",
        "task_state": {"route": {"steps": [{"capability": "season_summary", "arguments": {}}]}},
        "steps": [{"capability": "season_summary", "status": "complete", "result": {"points_total": 50},
                   "evidence": {"structured": {"source_sql": "SELECT", "snapshot_id": "s", "catalog_version": "c", "precision": {"rounding": "full_precision"}}}}],
    }
    case = {"category": "answerable", "expected": {
        "route": "structured", "status": "complete", "capability": "season_summary", "result": {"points_total": 50},
    }}
    scored = score_http_case(body, case, http_status=200, trace_parent_ok=True)
    assert scored["outcome"] == "failed"
    assert scored["checks"]["answer_fidelity"] is False


def test_m43_scorer_requires_nonempty_knowledge_citations():
    body = {
        "status": "complete", "route": "knowledge", "trace_id": "t1", "answer": "eFG% 是投篮效率指标。",
        "steps": [{"capability": "knowledge_glossary", "status": "complete", "result": {"answer": "eFG% 是投篮效率指标。"},
                   "evidence": {"knowledge": {"citations": []}}}],
    }
    case = {"category": "answerable", "expected": {
        "route": "knowledge", "status": "complete", "capability": "knowledge_glossary", "knowledge": True,
    }}
    scored = score_http_case(body, case, http_status=200, trace_parent_ok=True)
    assert scored["outcome"] == "failed"
    assert scored["checks"]["evidence"] is False


def test_m43_scorer_checks_nested_answer_values_and_mixed_citations():
    case = {"category": "answerable", "expected": {
        "route": "mixed", "status": "complete", "capability": "player_efg",
        "knowledge": True, "result": {"efg_percentage": 57.5, "inputs": {"sample_count": 10}},
    }}
    body = {
        "status": "complete", "route": "mixed", "trace_id": "t1",
        "answer": "eFG% 为 57.5%，样本数为 9。",
        "task_state": {"route": {"steps": [{"capability": "player_efg", "arguments": {}}]}},
        "steps": [
            {"capability": "player_efg", "status": "complete", "result": {"efg_percentage": 57.5, "inputs": {"sample_count": 10}},
             "evidence": {"structured": {"source_sql": "SELECT", "snapshot_id": "s", "catalog_version": "c", "precision": {"rounding": "full_precision"}}}},
            {"capability": "knowledge_glossary", "status": "complete",
             "evidence": {"knowledge": {"citations": [{"citation_id": "c1", "excerpt": "eFG", "source_id": "src", "source_version": "v1"}]}}},
        ],
    }
    scored = score_http_case(body, case, http_status=200, trace_parent_ok=True)
    assert scored["outcome"] == "failed"
    assert scored["reason"] == "answer_fidelity"


def test_m43_report_records_git_provenance(tmp_path):
    report = run_evaluation(output_path=tmp_path / "report.json")
    assert report["provenance"]["code_revision"]
    assert isinstance(report["provenance"]["git_dirty"], bool)


def test_m43_cli_honors_output_and_preserves_archive(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    archive = tmp_path / "reports/m4/agent-m43-contract-v1.json"
    archive.parent.mkdir(parents=True)
    archive.write_text("archived evidence\n")
    output = tmp_path / "fresh-contract.json"
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_agent_m43_contract.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert output.is_file(), result.stdout
    assert archive.read_text() == "archived evidence\n"
    assert json.loads(output.read_text())["summary"]["passed"] == 22


def test_m43_cli_requires_explicit_output(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_agent_m43_contract.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--output" in result.stderr
    assert not (tmp_path / "reports").exists()
