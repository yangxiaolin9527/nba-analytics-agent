"""Small in-process short-term conversation context for the local demo."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class ConversationSnapshot:
    conversation_id: str
    turn_index: int
    player_ids: tuple[str, ...] = ()
    team_ids: tuple[str, ...] = ()
    season_id: str | None = None
    competition_phase: str | None = None
    metric_terms: tuple[str, ...] = ()
    expires_at: float = 0.0
    pending: dict = field(default_factory=dict)


class ConversationStore:
    def __init__(self, *, max_items: int = 256, ttl_seconds: float = 1800.0, clock=time.monotonic):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, ConversationSnapshot] = OrderedDict()
        self._active: set[str] = set()

    def try_acquire(self, conversation_id: str) -> bool:
        # Called on the application event loop before any await or state read.
        if conversation_id in self._active:
            return False
        self._active.add(conversation_id)
        return True

    def release(self, conversation_id: str) -> None:
        self._active.discard(conversation_id)

    def get(self, conversation_id: str) -> ConversationSnapshot | None:
        item = self._items.get(conversation_id)
        if item is None:
            return None
        if item.expires_at <= self._clock():
            self._items.pop(conversation_id, None)
            return None
        self._items.move_to_end(conversation_id)
        return item

    def put(self, *, conversation_id: str, player_ids=(), team_ids=(), season_id=None, competition_phase=None, metric_terms=(), pending=None) -> ConversationSnapshot:
        previous = self.get(conversation_id)
        item = ConversationSnapshot(
            conversation_id=conversation_id,
            turn_index=(previous.turn_index + 1 if previous else 1),
            player_ids=tuple(player_ids), team_ids=tuple(team_ids),
            season_id=season_id, competition_phase=competition_phase,
            metric_terms=tuple(metric_terms),
            expires_at=self._clock() + self.ttl_seconds,
            pending=dict(pending or {}),
        )
        self._items[conversation_id] = item
        self._items.move_to_end(conversation_id)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return item
