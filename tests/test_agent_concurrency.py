import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.evaluate_agent_loop_conversation import FixtureDecider
from tests.test_agent_routes import FakeKnowledge
from tests.test_ask import make_ask_database, make_matrix


@pytest.mark.parametrize("first_route,second_route", [
    ("answer", "answer"), ("loop", "loop"), ("answer", "loop"), ("loop", "answer"),
])
def test_same_conversation_rejects_overlap_without_advancing_state(tmp_path, first_route, second_route):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class WaitingKnowledge(FakeKnowledge):
            attempts = 0
            async def execute(self, question, arguments):
                self.attempts += 1
                if not entered.is_set():
                    entered.set()
                    await release.wait()
                return await super().execute(question, arguments)

        database, matrix = tmp_path / "db.sqlite", tmp_path / "matrix.json"
        make_ask_database(database)
        make_matrix(matrix)
        knowledge = WaitingKnowledge()
        app = create_app(db_path=database, quality_matrix_path=matrix,
                         agent_knowledge_adapter=knowledge, agent_loop_enabled=True,
                         agent_loop_decider=FixtureDecider())
        store = app.state.agent_conversations
        original = store.put(conversation_id="shared", player_ids=["201939"],
                             season_id="2023-24", competition_phase="regular_season")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {"question": "eFG% 是什么？", "conversation_id": "shared"}
            first = asyncio.create_task(client.post(f"/agent/{first_route}", json=payload))
            await asyncio.wait_for(entered.wait(), 2)
            try:
                second = await client.post(f"/agent/{second_route}", json=payload)
                assert second.status_code == 409
                assert second.json() == {"detail": {"code": "conversation_busy", "conversation_id": "shared"}}
                assert store.get("shared") == original
                assert knowledge.attempts == 1
                # A different conversation is never blocked by the occupied ID.
                other = await client.post(f"/agent/{second_route}", json={**payload, "conversation_id": "other"})
                assert other.status_code == 200
                assert other.json()["turn_index"] == 1
            finally:
                release.set()
                completed = await first
            assert completed.status_code == 200
            assert completed.json()["turn_index"] == 2
            following = await client.post(f"/agent/{second_route}", json=payload)
            assert following.status_code == 200
            assert following.json()["turn_index"] == 3
    asyncio.run(run())


@pytest.mark.parametrize("route", ["answer", "loop"])
@pytest.mark.parametrize("failure", ["exception", "cancel"])
def test_conversation_occupancy_released_after_failure(tmp_path, monkeypatch, route, failure):
    async def run():
        entered = asyncio.Event()

        class BrokenKnowledge(FakeKnowledge):
            async def execute(self, question, arguments):
                entered.set()
                await asyncio.Event().wait()

        database, matrix = tmp_path / "db.sqlite", tmp_path / "matrix.json"
        make_ask_database(database)
        make_matrix(matrix)
        app = create_app(db_path=database, quality_matrix_path=matrix,
                         agent_knowledge_adapter=BrokenKnowledge(), agent_loop_enabled=True,
                         agent_loop_decider=FixtureDecider())
        if failure == "exception":
            module = "app.agent_routes.plan_question" if route == "answer" else "app.agent_loop_routes.build_task_profile"
            monkeypatch.setattr(module, lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fixture")))
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            pending = asyncio.create_task(client.post(f"/agent/{route}", json={
                "question": "eFG% 是什么？", "conversation_id": "broken"}))
            if failure == "cancel":
                await asyncio.wait_for(entered.wait(), 2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                assert (await pending).status_code == 500
            store = app.state.agent_conversations
            assert store.try_acquire("broken")
            store.release("broken")
    asyncio.run(run())
