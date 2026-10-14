"""Run one local deterministic structured answer for an interview demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.model_adapter import FakeModel
from scripts.provider_config import load_provider_config


FIXTURE_QUESTION = "库里 2023-24 常规赛的 eFG%？"
FIXTURE_PLAN = {"action": "call", "tool": "player_efg", "arguments": {
    "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
}}


async def run(question: str, *, fixture: bool = False, provider_config=None) -> dict:
    if fixture and question != FIXTURE_QUESTION:
        raise ValueError("fixture mode supports only the pinned eFG% example")
    model = FakeModel({None: json.dumps(FIXTURE_PLAN)}) if fixture else None
    app = create_app(model=model, provider_config=provider_config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://demo") as client:
        response = await client.post("/answer", json={"question": question})
    return {"http_status": response.status_code, **response.json()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=FIXTURE_QUESTION)
    parser.add_argument("--fixture", action="store_true", help="use the pinned offline plan")
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args.question, fixture=args.fixture,
                                 provider_config=load_provider_config()))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["http_status"] != 200 or result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
