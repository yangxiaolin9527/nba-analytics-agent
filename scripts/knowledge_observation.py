"""Default-off, per-request snapshots for offline diagnosis of the real pipeline.

Recording is synchronous and memory-only. The evaluator closes each recorder
before flushing it to disk. No exporter, model call, or global request state.
"""
from __future__ import annotations

from copy import deepcopy


class RequestObservation:
    def __init__(self):
        self.events: dict = {}
        self.errors: list[str] = []
        self.closed = False

    def record(self, stage: str, payload) -> None:
        if not self.closed:
            self.events[stage] = deepcopy(payload)

    def snapshot(self) -> dict:
        return {"events": deepcopy(self.events), "valid": not self.errors,
                "errors": list(self.errors)}

    def close(self) -> None:
        self.closed = True


def observe(observer, stage: str, payload) -> None:
    if observer is None:
        return
    try:
        observer.record(stage, payload)
    except Exception as exc:
        # Never let diagnostic failure change the public response. The recorder
        # contract includes errors so evaluation cannot silently trust this run.
        observer.errors.append(f"{stage}: {type(exc).__name__}")


def candidate_rows(items, score_type: str) -> list[dict]:
    return [{"citation_id": item.node.metadata["citation_id"], "rank": rank,
             "score": float(item.score or 0), "score_type": score_type,
             "source_id": item.node.metadata["id"],
             "page": item.node.metadata.get("page"), "anchor": item.node.metadata.get("anchor")}
            for rank, item in enumerate(items, 1)]


def record_ranking(observer, nodes, scores, dense_scores, ranked_ids, intent, hits):
    """Expand diagnostic boundaries from the actual ranking inputs, without queries.

    Production stops after K deduplicated locations; the full diagnostic dedup
    list is a pure derivation from the same already-observed candidates.
    """
    if observer is None:
        return
    ordered = sorted(scores, key=lambda cid: (-scores[cid], -dense_scores.get(cid, 0), cid))
    domain = [cid for cid in ranked_ids if not (
        (intent == "rules" and nodes[cid].metadata["page"] is None)
        or (intent == "glossary" and nodes[cid].metadata["anchor"] is None))]
    dedup, seen = [], set()
    for cid in domain:
        meta = nodes[cid].metadata
        location = (meta["id"], meta["page"] if meta["page"] is not None else meta.get("anchor"))
        if location not in seen:
            dedup.append(cid)
            seen.add(location)
    for stage, ids in (("rrf_ranking", ordered), ("dense_threshold", ranked_ids),
                       ("domain_filter", domain), ("dedup", dedup)):
        observe(observer, stage, [{"citation_id": cid, "rank": rank,
            "rrf_score": scores[cid], "dense_score": dense_scores.get(cid),
            "score_type": "rrf", "source_id": nodes[cid].metadata["id"],
            "page": nodes[cid].metadata.get("page"), "anchor": nodes[cid].metadata.get("anchor")}
            for rank, cid in enumerate(ids, 1)])
    observe(observer, "final_top_k", hits)
