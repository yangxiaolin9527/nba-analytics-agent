from app.conversation import ConversationStore


def test_conversation_store_increments_turn_and_evicts_expired_context():
    now = [100.0]
    store = ConversationStore(ttl_seconds=10, clock=lambda: now[0])
    first = store.put(conversation_id="c1", player_ids=["201939"], season_id="2023-24")
    second = store.put(conversation_id="c1", player_ids=["2544"], season_id="2023-24")
    assert first.turn_index == 1
    assert second.turn_index == 2
    assert store.get("c1").player_ids == ("2544",)
    now[0] = 111.0
    assert store.get("c1") is None


def test_store_evicts_least_recently_used_context():
    store = ConversationStore(max_items=2)
    store.put(conversation_id="a")
    store.put(conversation_id="b")
    assert store.get("a")
    store.put(conversation_id="c")
    assert store.get("b") is None
    assert store.get("a")
