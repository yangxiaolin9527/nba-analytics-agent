"""Small, explicit demo surface for the interview workflow."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "metadata/m3_4_demo_cases_v2.json"
AGENT_REPORT_PATH = ROOT / "reports/demo/contract-replay.json"
LIVE_AGENT_REPORT_PATH = ROOT / "reports/demo/historical-mixed-replay.json"
STATIC_PATH = ROOT / "static"

AGENT_REPLAY_CASES = {
    "M43-001": ("库里 2023-24 常规赛总得分？", "fixture_contract"),
    "M43-003": ("2023-10-01 湖人对勇士谁赢了？", "fixture_contract"),
    "M43-004": ("库里 2023-24 常规赛最近两次出场的场均得分？", "fixture_contract"),
    "M43-007": ("库里 2023-24 常规赛总得分比上一季增加多少？", "fixture_contract"),
    "M43-009": ("库里 2023-24 常规赛 eFG% 是多少？", "fixture_contract"),
    "M43-010": ("eFG% 是什么？", "synthetic_knowledge_contract"),
    "M43-011": ("库里 2023-24 常规赛 eFG% 是多少？这个指标代表什么？", "synthetic_mixed_contract"),
    "M43-019": ("库里 2023-24 常规赛 eFG% 是多少？这个指标代表什么？", "synthetic_partial_contract"),
    "M43-012": ("库里 2023-24 最近两次出场的总得分？", "fixture_contract"),
    "M43-016": ("计算库里 2023-24 常规赛的 TS%？", "fixture_contract"),
}


class FixtureRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)


def _agent_report(path: Path = AGENT_REPORT_PATH) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("agent replay report is unavailable") from exc
    if not isinstance(document.get("cases"), list):
        raise RuntimeError("agent replay report is invalid")
    return document


def _agent_replay_cases() -> list[dict[str, Any]]:
    report = _agent_report()
    records = {item.get("case_id"): item for item in report["cases"]}
    cases = []
    for case_id, (question, source_type) in AGENT_REPLAY_CASES.items():
        record = records.get(case_id)
        if record is None:
            continue
        cases.append({
            "case_id": case_id, "question": question, "source_type": source_type,
            "status": record.get("status"), "route": record.get("route"),
        })
    live_path = LIVE_AGENT_REPORT_PATH
    if live_path.exists():
        try:
            live = _agent_report(live_path)
            for record in live.get("cases", []):
                if record.get("outcome") == "passed":
                    cases.append({
                        "case_id": record.get("case_id", "M44-LIVE-MIXED"),
                        "question": record.get("question", ""),
                        "source_type": "live_provider",
                        "status": record.get("status"), "route": record.get("route"),
                    })
        except RuntimeError:
            pass
    return cases


def _catalog() -> dict[str, Any]:
    try:
        document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("demo case catalog is unavailable") from exc
    if not isinstance(document.get("cases"), list):
        raise RuntimeError("demo case catalog is invalid")
    return document


def register_demo_routes(application: FastAPI, settings: Any = None) -> None:
    """Register routes that expose presentation fixtures without changing API contracts."""

    @application.get("/", include_in_schema=False)
    async def home_page():
        return RedirectResponse("/demo")

    @application.get("/demo/static/{asset_name}", include_in_schema=False)
    async def demo_asset(asset_name: str):
        allowed = {"demo.css", "demo.js", "demo-state.js", "demo-data.js", "lucide.min.js", "curry.png", "james.png"}
        asset = STATIC_PATH / asset_name
        if asset_name not in allowed or not asset.is_file():
            raise HTTPException(404, detail="asset not found")
        media_type = "text/javascript" if asset_name.endswith(".js") else None
        if asset_name.endswith(".css"):
            media_type = "text/css"
        elif asset_name.endswith(".png"):
            media_type = "image/png"
        return Response(asset.read_bytes(), media_type=media_type)

    @application.get("/demo", include_in_schema=False)
    async def demo_page():
        try:
            html = (STATIC_PATH / "demo.html").read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=503, detail="demo page is unavailable") from exc
        return HTMLResponse(html.replace("<!-- served-base -->", '<base href="/demo/static/">', 1))

    @application.get("/demo/cases")
    async def demo_cases():
        document = _catalog()
        return JSONResponse({
            "version": document["version"],
            "fixture_mode": document.get("fixture_mode", "static_replay"),
            "source_snapshot": document["source_snapshot"],
            "provenance": document.get("provenance", {}),
            "cases": [
                {"case_id": case["case_id"], "label": case["label"], "tool": case["tool"], "question": case["question"]}
                for case in document["cases"]
            ],
        })

    @application.get("/demo/status")
    async def demo_status():
        database = getattr(settings, "database", None)
        quality_matrix = getattr(settings, "quality_matrix_path", None)
        index_value = settings.provider_config.knowledge_index_path
        index_path = Path(index_value) if index_value else None
        provider_config = settings.provider_config
        model_configured = bool(getattr(settings, "model", None) or provider_config.model_configured)
        embedding_configured = provider_config.embedding_configured
        nodes_available = bool(index_path and (index_path / "nodes.json").is_file())
        dense_available = bool(index_path and (index_path / "dense.json").is_file())
        database_configured = database is not None
        database_exists = bool(database and database.is_file())
        database_readable = False
        if database_exists:
            try:
                with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
                    connection.execute("SELECT 1").fetchone()
                database_readable = True
            except (OSError, sqlite3.Error):
                database_readable = False
        database_available = database_exists and database_readable
        quality_matrix_available = bool(quality_matrix and quality_matrix.is_file())
        mixed_ready = all((database_available, quality_matrix_available, nodes_available,
                           dense_available, model_configured, embedding_configured))
        return {
            'routing': {'default': settings.routing_mode or 'rules', 'available': ['model', 'rules']},
            "modes": {
                "replay": True,
                "structured": bool(database_available and quality_matrix_available),
                "mixed": mixed_ready,
            },
            "resources": {
                "database": {"configured": database_configured, "exists": database_exists,
                              "readable": database_readable, "available": database_available},
                "quality_matrix": {"configured": bool(quality_matrix), "available": quality_matrix_available},
                "knowledge_index": {"configured": bool(index_path), "available": nodes_available,
                                     "dense_available": dense_available},
                "model": {"configured": model_configured},
                "embedding": {"configured": embedding_configured},
            },
        }

    @application.get("/healthz")
    async def healthz():
        """Process liveness; it intentionally does not require optional resources."""
        return {"status": "ok"}

    @application.get("/readyz")
    async def readyz():
        """Deployment readiness for the replay-capable application surface."""
        catalog_ready = CASES_PATH.is_file()
        page_ready = (STATIC_PATH / "demo.html").is_file()
        ready = catalog_ready and page_ready
        payload = {"status": "ok" if ready else "unavailable",
                   "modes": {"replay": ready}}
        return JSONResponse(payload, status_code=200 if ready else 503)

    @application.get("/demo/agent/cases")
    async def demo_agent_cases():
        return {
            "version": "m4.4.1",
            "fixture_mode": "m43_contract_replay",
            "source_snapshot": "fixture:eoin-v515-shape-v1",
            "cases": _agent_replay_cases(),
        }

    @application.post("/demo/agent/replay")
    async def demo_agent_replay(request: FixtureRequest):
        report = _agent_report()
        record = next((item for item in report["cases"] if item.get("case_id") == request.case_id), None)
        source_type = AGENT_REPLAY_CASES.get(request.case_id, (None, "fixture_contract"))[1]
        if record is None and LIVE_AGENT_REPORT_PATH.exists():
            live = _agent_report(LIVE_AGENT_REPORT_PATH)
            record = next((item for item in live.get("cases", []) if item.get("case_id") == request.case_id), None)
            source_type = "live_provider"
        if record is None:
            raise HTTPException(status_code=404, detail="unknown agent replay case")
        return {
            "mode": "replay", "case_id": request.case_id,
            "provenance": {
                "source_type": source_type,
                "report": str((LIVE_AGENT_REPORT_PATH if source_type == "live_provider" else AGENT_REPORT_PATH).relative_to(ROOT)),
                "case_set_version": report.get("case_set_version"),
            },
            "response": record.get("response", {}),
        }

    @application.post("/demo/fixture")
    async def demo_fixture(request: FixtureRequest):
        document = _catalog()
        case = next((item for item in document["cases"] if item.get("case_id") == request.case_id), None)
        if case is None:
            raise HTTPException(status_code=404, detail="unknown demo case")
        return {"mode": "fixture", "case_id": case["case_id"], "response": case["response"]}
