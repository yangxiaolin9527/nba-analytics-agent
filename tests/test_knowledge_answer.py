import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from llama_index.core.schema import TextNode


def node(citation, excerpt, page=42):
    return TextNode(id_=citation, text=excerpt, metadata={
        "citation_id": citation, "id": "nba-rulebook-2025-26", "page": page,
        "anchor": None, "heading": None, "source_url": "https://example.com/rules",
        "version": "2025-26", "sha256": "fixture",
    })


class Search:
    def __init__(self, nodes):
        self.nodes = {item.id_: item for item in nodes}
        self.dense = object()


class Guided:
    async def asearch(self, question, *, top_k):
        return {"retrieval_status": "candidates_found", "reason_code": None,
                "hits": [{"citation_id": "first", "excerpt": "Unrelated first chunk.",
                          "source_id": "nba-rulebook-2025-26", "page": 42,
                          "source_url": "https://example.com/rules", "source_version": "2025-26",
                "sha256": "fixture", "anchor": None, "heading": None}]}


class GlossaryGuided(Guided):
    async def asearch(self, question, *, top_k):
        result = await super().asearch(question, top_k=top_k)
        result.update({"intent": "glossary", "question_scope": "single"})
        result["hits"][0].update({"excerpt": "A player in the game shoots.", "source_id": "nba-stats-glossary", "page": None,
                                  "anchor": "term", "source_version": "test"})
        return result


class Model:
    model = "fixture"

    def __init__(self, generation, verification):
        self.responses = [generation, verification]
        self.prompts = []
        self.timeouts = []
        self.thinking = []

    def generate(self, request):
        from scripts.model_adapter import ModelResponse
        self.prompts.append(request.prompt)
        self.timeouts.append(request.timeout)
        self.thinking.append(getattr(request, "thinking", None))
        return ModelResponse(json.dumps(self.responses.pop(0)), self.model)


def fixture(generation, verification):
    from scripts.knowledge_answer import KnowledgeAnswer

    search = Search([node("first", "Unrelated first chunk."),
                     node("target", "Technical free throws must be attempted by a player in the game.")])
    model = Model(generation, verification)
    return KnowledgeAnswer(search, model, guided=Guided()), model


def part(question="Who shoots?", claim="A player in the game shoots.",
         citation="target", excerpt="a player in the game"):
    return {"question": question, "claim": claim, "citation_id": citation, "excerpt": excerpt}


def test_expands_same_page_to_answer_bearing_chunk():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "answered"
    assert result["claims"][0]["citation_id"] == "target"
    assert result["claims"][0]["excerpt"] == "a player in the game"
    assert "Technical free throws" in model.prompts[0]
    assert "Technical free throws" in model.prompts[1]
    assert len(model.prompts) == 2


def test_answer_reports_evidence_budget_usage():
    answer, _ = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["evidence_chunk_count"] == 2
    assert result["evidence_char_count"] > 0
    assert result["evidence_page_count"] == 1
    assert result["evidence_citation_ids"] == ["first", "target"]


def test_answer_uses_separate_candidate_pool_budget():
    from scripts.knowledge_answer import KnowledgeAnswer

    class RecordingGuided(Guided):
        def __init__(self):
            self.top_k = None

        async def asearch(self, question, *, top_k):
            self.top_k = top_k
            return await super().asearch(question, top_k=top_k)

    guided = RecordingGuided()
    search = Search([node("first", "Unrelated first chunk.")])
    model = Model({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer = KnowledgeAnswer(search, model, guided=guided, candidate_top_k=10)

    asyncio.run(answer.aanswer("Who shoots?"))

    assert guided.top_k == 10


def test_runtime_reports_actual_stages_including_skipped_verification():
    answer, _ = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["stage_statuses"] == {"retrieval": "complete", "generation": "complete",
                                        "citation_validation": "complete", "verification": "complete"}
    answer, _ = fixture({"parts": [part(citation="unknown")]}, {})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["stage_statuses"]["citation_validation"] == "complete"
    assert result["stage_statuses"]["verification"] == "skipped"


def test_runtime_events_include_evidence_and_terminal_skipped_stages():
    from app.execution_events import ExecutionEventSink, execution_event_context

    answer, _ = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    sink = ExecutionEventSink(request_id="req-1")
    with execution_event_context(sink):
        result = asyncio.run(answer.aanswer("Who shoots?"))

    pairs = [(event["stage"], event["status"]) for event in sink.snapshot()]
    assert ("evidence", "running") in pairs
    assert ("evidence", "complete") in pairs
    assert ("verification", "complete") in pairs
    assert ("result", "complete") in pairs


def test_failed_generation_reports_attempt_metadata_and_skipped_stages():
    from scripts.knowledge_answer import KnowledgeAnswerError
    answer, model = fixture({}, {})
    model.base_url = 'https://example.test/v1'
    def fail(request):
        raise TimeoutError('fixture')
    model.generate = fail
    with pytest.raises(KnowledgeAnswerError) as caught:
        asyncio.run(answer.aanswer('Who shoots?'))
    error = caught.value
    assert error.stage_statuses == {'retrieval': 'complete', 'generation': 'failed',
                                    'citation_validation': 'skipped', 'verification': 'skipped'}
    assert error.model_calls['generation']['provider'] == 'example.test'
    assert error.model_calls['generation']['usage'] is None
    assert error.model_calls['generation']['status'] == 'failed'


def test_bounded_evidence_stops_after_first_budget_overflow():
    from scripts.knowledge_answer import bounded_evidence

    hits = [{"citation_id": str(index), "excerpt": text} for index, text in
            enumerate(["a" * 400, "b" * 400, "c" * 10])]
    selected = bounded_evidence("question", hits, max_prompt_chars=1800)
    assert [hit["citation_id"] for hit in selected] == ["0"]


def test_generation_prompt_stays_within_character_budget():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.search.nodes["first"].text = '"' * 12000
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "abstained"
    assert result["reason_code"] == "no_evidence"
    assert model.prompts == []


def test_two_blocking_model_calls_complete_without_executor_callback_wakeup():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})

    async def run():
        return await asyncio.wait_for(answer.aanswer("Who shoots?"), timeout=1.0)

    result = asyncio.run(run())
    assert result["status"] == "answered"
    assert len(model.prompts) == 2


def test_answer_stages_pass_provider_timeout_to_each_model_request():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.generation_timeout = 0.25
    answer.verification_timeout = 0.35

    result = asyncio.run(answer.aanswer("Who shoots?"))

    assert result["status"] == "answered"
    assert model.timeouts == [0.25, 0.35]




def test_answer_can_disable_reasoning_for_generation_but_not_verification():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.generation_thinking = "disabled"

    result = asyncio.run(answer.aanswer("Who shoots?"))

    assert result["status"] == "answered"
    assert model.thinking == [{"type": "disabled"}, None]


def test_answer_can_disable_reasoning_for_verification_independently():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.verification_thinking = "disabled"
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "answered"
    assert model.thinking == [None, {"type": "disabled"}]


def test_adaptive_policy_uses_fast_path_only_for_single_glossary_evidence():
    from scripts.knowledge_answer import KnowledgeAnswer
    answer, model = fixture({"parts": [part(citation="first", excerpt="A player in the game shoots.")]},
                            {"verdicts": [{"index": 0, "supported": True}]})
    answer.guided = GlossaryGuided()
    answer.search.nodes["first"].metadata.update({"id": "nba-stats-glossary", "page": None, "anchor": "term"})
    answer.search.nodes["first"].text = "A player in the game shoots."
    answer.search.search = lambda *args, **kwargs: {"hits": []}
    answer.generation_policy = "adaptive"

    result = asyncio.run(answer.aanswer("What is the metric?"))

    assert result["status"] == "answered"
    assert result["generation_policy"] == "adaptive_glossary"
    assert result["generation_thinking"] == "disabled"
    assert result["generation_prompt_strategy"] == "concise"
    assert model.thinking == [{"type": "disabled"}, None]


def test_adaptive_policy_keeps_baseline_for_rule_evidence():
    from scripts.knowledge_answer import KnowledgeAnswer
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.generation_policy = "adaptive"

    result = asyncio.run(answer.aanswer("What is the rule?"))

    assert result["status"] == "answered"
    assert result["generation_policy"] == "fixed"
    assert result["generation_thinking"] is None
    assert result["generation_prompt_strategy"] == "baseline"
    assert model.thinking == [None, None]


def test_generation_prompt_strategy_changes_only_generation_prompt():
    from scripts.knowledge_answer import generation_prompt
    hits = [{"citation_id": "target", "excerpt": "Exact source text with all conditions and penalties."}]
    baseline = generation_prompt("Who shoots?", hits)
    concise = generation_prompt("Who shoots?", hits, strategy="concise")
    assert baseline != concise
    assert "Exact source text with all conditions and penalties." in concise
    assert "citation_id" in concise and "VERBATIM" in concise
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.generation_prompt_strategy = "concise"
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "answered"
    assert "Extract" in model.prompts[0]
    assert "Independently check" in model.prompts[1]


def test_generation_prompt_strategy_rejects_invalid_value():
    from scripts.knowledge_answer import KnowledgeAnswer
    answer, model = fixture({}, {})
    with pytest.raises(ValueError, match="generation_prompt_strategy"):
        KnowledgeAnswer(answer.search, model, guided=Guided(), generation_prompt_strategy="unknown")


def test_answer_caps_stage_call_timeout_by_remaining_total_budget():
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    answer.total_timeout = 0.5
    answer.generation_timeout = 10.0
    answer.verification_timeout = 10.0

    result = asyncio.run(answer.aanswer("Who shoots?"))

    assert result["status"] == "answered"
    assert model.timeouts[0] < 0.5


def test_early_provider_timeout_keeps_generation_classification():
    from scripts.knowledge_answer import KnowledgeAnswerError
    answer, _ = fixture({"parts": [part()]}, {"verdicts": []})
    answer.total_timeout = 0.5
    answer.generation_timeout = 10
    async def timed_out(prompt, timeout):
        raise TimeoutError("provider closed early")
    answer._generate = timed_out
    with pytest.raises(KnowledgeAnswerError) as caught:
        asyncio.run(answer.aanswer("Who shoots?"))
    assert caught.value.reason_code == "generation_timeout"


def test_total_timeout_preserves_retrieval_and_active_generation_timings():
    from scripts.knowledge_answer import KnowledgeAnswerError
    answer, _ = fixture({"parts": [part()]}, {"verdicts": []})
    answer.total_timeout = .05
    async def slow(prompt, timeout):
        await asyncio.sleep(1)
    answer._generate = slow
    with pytest.raises(KnowledgeAnswerError) as caught:
        asyncio.run(answer.aanswer("Who shoots?"))
    assert caught.value.reason_code == "total_timeout"
    assert caught.value.timings["retrieval_ms"] > 0
    assert caught.value.timings["generation_ms"] > 0
    assert caught.value.timings["total_ms"] >= caught.value.timings["generation_ms"]


def test_generation_usage_is_request_local_and_keeps_verifier_usage():
    from scripts.knowledge_observation import RequestObservation
    from scripts.model_adapter import ModelResponse
    answer, model = fixture({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    generate = model.generate
    def with_usage(request):
        response = generate(request)
        return ModelResponse(response.text, response.model, usage={"total_tokens": 5})
    model.generate = with_usage
    answer.observer = RequestObservation()
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["model_calls"]["generation"]["usage"] == {"total_tokens": 5}
    assert result["model_calls"]["verification"]["usage"] == {"total_tokens": 5}


def test_concurrent_answers_keep_call_usage_and_deadlines_isolated():
    from scripts.knowledge_runtime import current_runtime
    from scripts.model_adapter import ModelResponse
    answer, model = fixture({}, {})
    def generate(request):
        marker = 'BBB' if 'BBB' in request.prompt else 'A'
        payload = ({'verdicts': [{'index': 0, 'supported': True}]}
                   if 'Original question:' in request.prompt else {'parts': [part(question=marker)]})
        return ModelResponse(json.dumps(payload), model.model, usage={'total_tokens': len(marker)})
    model.generate = generate
    async def run():
        results = await asyncio.gather(answer.aanswer('Who shoots A?'), answer.aanswer('Who shoots BBB?'))
        assert current_runtime.get() is None
        return results
    results = asyncio.run(run())
    for result, tokens in zip(results, (1, 3)):
        assert result['status'] == 'answered'
        assert result['model_calls']['generation']['usage']['total_tokens'] == tokens
        assert result['model_calls']['verification']['usage']['total_tokens'] == tokens


@pytest.mark.parametrize('stage,exception,code', [
    ('generation', TimeoutError(), 'generation_timeout'),
    ('verification', TimeoutError(), 'verification_timeout'),
    ('generation', RuntimeError('provider unavailable'), 'generation_error'),
    ('verification', RuntimeError('provider unavailable'), 'verification_error'),
])
def test_stage_failures_keep_classification_and_completed_timings(stage, exception, code):
    from scripts.knowledge_answer import KnowledgeAnswerError
    from scripts.knowledge_runtime import current_runtime
    answer, model = fixture({'parts': [part()]}, {'verdicts': []})
    generate = answer._generate
    async def fail(prompt, timeout):
        if current_runtime.get().stage == stage:
            raise exception
        return await generate(prompt, timeout)
    answer._generate = fail
    with pytest.raises(KnowledgeAnswerError) as caught:
        asyncio.run(answer.aanswer('Who shoots?'))
    assert caught.value.reason_code == code
    assert caught.value.stage == stage
    assert caught.value.retryable == code.endswith('timeout')
    assert caught.value.timings['total_ms'] >= caught.value.timings['retrieval_ms']
    assert len(model.prompts) == (stage == 'verification')


@pytest.mark.parametrize("mode", ["bogus", "", "DISABLED"])
def test_generation_thinking_rejects_invalid_configuration(mode):
    from scripts.knowledge_answer import KnowledgeAnswer
    answer, model = fixture({"parts": []}, {"verdicts": []})
    with pytest.raises(ValueError, match="generation_thinking"):
        KnowledgeAnswer(answer.search, model, guided=Guided(), generation_thinking=mode)


def test_answer_enforces_total_timeout_at_answerer_boundary():
    from scripts.knowledge_answer import KnowledgeAnswer, KnowledgeAnswerError

    class SlowGuided:
        async def asearch(self, question, *, top_k):
            await asyncio.sleep(0.05)
            return {"retrieval_status": "no_candidates", "reason_code": "no_evidence"}

    answer = KnowledgeAnswer(
        Search([node("first", "Unrelated first chunk.")]),
        Model({}, {}),
        guided=SlowGuided(),
        retrieval_timeout=1,
        total_timeout=0.01,
    )

    with pytest.raises(KnowledgeAnswerError) as error:
        asyncio.run(answer.aanswer("Who shoots?"))

    assert error.value.stage == "total"
    assert error.value.reason_code == "total_timeout"


def test_answer_pipeline_emits_nested_diagnostic_spans():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from scripts.knowledge_answer import KnowledgeAnswer
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    class SequenceModel:
        model = "fixture"

        def __init__(self):
            self.responses = [
                {"action": "search", "intent": "rules", "english_query": "Who shoots?"},
                {"parts": [part()]},
                {"verdicts": [{"index": 0, "supported": True}]},
            ]

        def generate(self, request):
            from scripts.model_adapter import ModelResponse
            return ModelResponse(json.dumps(self.responses.pop(0)), self.model)

    target = node("target", "Technical free throws must be attempted by a player in the game.")
    other = node("first", "Unrelated first chunk.")
    class Dense:
        class Embedding:
            model = "text-embedding-3-small"

            async def aembed(self, texts):
                return [[float(index)] for index, _ in enumerate(texts)]

        def __init__(self):
            self.embedding = self.Embedding()
            self.last_vector_query_ms = 0.0

        def retrieve_vector(self, vector, top_k):
            from llama_index.core.schema import NodeWithScore
            return [NodeWithScore(node=target, score=0.9), NodeWithScore(node=other, score=0.2)]

    dense = Dense()
    search = KnowledgeSearch([other, target], dense=dense)
    model = SequenceModel()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    guided = GuidedKnowledgeSearch(search, model, min_dense_score=0.0, tracer=tracer)
    answer = KnowledgeAnswer(search, model, guided=guided, tracer=tracer)

    result = asyncio.run(answer.aanswer("Who shoots?"))

    assert result["status"] == "answered"
    assert result["trace_id"]
    answer_span = next(item for item in exporter.get_finished_spans()
                       if item.name == "nba.knowledge.answer")
    assert '"status":"answered"' in answer_span.attributes["output.value.preview"]
    assert answer_span.attributes["output.mime_type"] == "application/json"
    assert {item.name for item in exporter.get_finished_spans()} == {
        "nba.knowledge.answer", "nba.knowledge.retrieval", "nba.knowledge.planner",
        "nba.knowledge.embedding", "nba.knowledge.ranking", "nba.knowledge.evidence",
        "nba.knowledge.generation", "nba.knowledge.citation_validation",
        "nba.knowledge.answer_language_validation",
        "nba.knowledge.verification",
    }


def test_expansion_caps_five_rule_pages_and_twelve_chunks():
    from scripts.knowledge_answer import expand_evidence

    nodes = [node(f"p{page}-{number}", f"Page {page} chunk {number}.", page)
             for page in range(1, 5) for number in range(4)]
    search = Search(nodes)
    hits = [{"citation_id": f"p{page}-0", "excerpt": f"Page {page} chunk 0.",
             "source_id": "nba-rulebook-2025-26", "page": page} for page in range(1, 5)]
    expanded = expand_evidence(search, hits)
    assert len(expanded) == 12
    assert {hit["page"] for hit in expanded} == {1, 2, 3, 4}
    assert {"p1-1", "p2-1", "p3-1", "p4-1"} <= {hit["citation_id"] for hit in expanded}


def test_lexical_glossary_candidate_can_supplement_guided_evidence():
    from scripts.knowledge_answer import supplement_hits

    class Lexical:
        def search(self, question, *, mode, top_k):
            assert mode == "bm25"
            return {"hits": [{"citation_id": "glossary:ast", "source_id": "glossary",
                              "page": None, "anchor": "ast", "excerpt": "Assists are passes."}]}

    hits = supplement_hits(Lexical(), [{"citation_id": "glossary:other", "source_id": "glossary",
                                     "page": None, "anchor": "other"}], "AST definition")
    assert [hit["citation_id"] for hit in hits] == ["glossary:other", "glossary:ast"]


def test_exact_glossary_headings_precede_nearby_specialized_terms():
    from scripts.knowledge_answer import supplement_hits

    class Glossary:
        nodes = {item.id_: item for item in [
            TextNode(id_="oreb", text="Name Offensive Rebounds Definition basic", metadata={
                "citation_id": "oreb", "id": "glossary", "page": None, "anchor": "oreb",
                "heading": "Offensive Rebounds", "source_url": "https://example.com",
                "version": "test", "sha256": "fixture"}),
            TextNode(id_="dreb", text="Name Defensive Rebounds Definition basic", metadata={
                "citation_id": "dreb", "id": "glossary", "page": None, "anchor": "dreb",
                "heading": "Defensive Rebounds", "source_url": "https://example.com",
                "version": "test", "sha256": "fixture"}),
        ]}

        def search(self, question, *, mode, top_k):
            return {"hits": []}

    guided = [{"citation_id": "contested", "source_id": "glossary", "page": None,
               "anchor": "contested_oreb", "excerpt": "Contested offensive rebounds."}]
    hits = supplement_hits(Glossary(), guided, "offensive rebounds and defensive rebounds")
    assert [hit["citation_id"] for hit in hits] == ["contested", "oreb", "dreb"]


@pytest.mark.parametrize("citation,excerpt", [
    ("invented", "a player in the game"),
    ("target", "any player on the bench"),
])
def test_invalid_citation_or_excerpt_fails_closed(citation, excerpt):
    answer, model = fixture({"parts": [part(citation=citation, excerpt=excerpt)]},
                            {"verdicts": [{"index": 0, "supported": True}]})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "abstained"
    assert result["claims"] == []
    assert len(model.prompts) == 1


def test_forged_hit_excerpt_is_rejected_against_index():
    answer, model = fixture({"parts": [part(citation="first", excerpt="Forged text")]},
                            {"verdicts": [{"index": 0, "supported": True}]})

    class Forged:
        async def asearch(self, question, *, top_k):
            return {"retrieval_status": "candidates_found", "hits": [{
                "citation_id": "first", "excerpt": "Forged text", "source_id": "nba-rulebook-2025-26",
                "page": 42, "source_url": "https://example.com/rules", "source_version": "2025-26",
                "sha256": "fixture", "anchor": None, "heading": None,
            }]}

    answer.guided = Forged()
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "abstained"
    assert len(model.prompts) == 1


def test_pdf_whitespace_quote_is_canonicalized_to_verbatim_source():
    from scripts.knowledge_answer import canonical_excerpt

    source = "If a player is eligible to \nshoot, they may shoot."
    assert canonical_excerpt(source, "eligible to  \nshoot") == "eligible to \nshoot"
    assert canonical_excerpt(source, "eligible to score") is None


def test_semantic_rejection_prevents_unsupported_claim():
    answer, _ = fixture({"parts": [part(claim="Anyone on the bench shoots.")]},
                        {"verdicts": [{"index": 0, "supported": False}]})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "abstained"
    assert result["answer"] is None


def test_verification_receives_full_source_context_for_short_quote():
    from scripts.knowledge_answer import verification_prompt

    prompt = verification_prompt("Who shoots?", [{"index": 0, "question": "Who shoots?",
        "claim": "A player in the game shoots.", "excerpt": "a player in the game",
        "source_excerpt": "Technical free throws must be attempted by a player in the game."}])
    assert "Technical free throws must be attempted" in prompt
    assert "specific named player" in prompt


def test_verification_requires_short_citation_excerpt_to_cover_every_claim_clause():
    from scripts.knowledge_answer import verification_prompt

    prompt = verification_prompt("How is the participant punished?", [{
        "index": 0,
        "question": "How is the participant punished?",
        "claim": "The participant is ejected and may receive a fine.",
        "excerpt": "The participant will be ejected immediately.",
        "source_excerpt": "The participant will be ejected immediately. A fine may be imposed.",
    }])
    assert "cited excerpt itself" in prompt
    assert "every independent factual clause" in prompt


def test_generation_requests_the_same_clause_coverage_as_verification():
    from scripts.knowledge_answer import generation_prompt

    prompt = generation_prompt("How is Contested DREB% calculated?", [])
    assert "every independent factual clause" in prompt
    assert "definition and formula" in prompt


def test_generation_prompt_allows_independent_claims_when_one_excerpt_is_incomplete():
    from scripts.knowledge_answer import generation_prompt

    prompt = generation_prompt("When is this foul called and what is the penalty?", [{
        "citation_id": "rule", "excerpt": "A dribbler shall not charge into a legal guarding position."
    }])
    assert "split independently supported facts into separate parts" in prompt.lower()
    assert "leave the unsupported part unresolved" in prompt


@pytest.mark.parametrize("full_quote", [True, False])
def test_formula_claim_keeps_quote_coverage_and_verifier_rejection(full_quote):
    definition = "Contested rebounds have an opponent within 3.5 feet. Formula: (Contested DREB)/(DREB)."
    quote = definition if full_quote else "(Contested DREB)/(DREB)"
    answer, model = fixture({"parts": [part(
        question="How is the percentage calculated?",
        claim="Contested DREB divided by DREB; contested means an opponent within 3.5 feet.",
        excerpt=quote,
    )]}, {"verdicts": [{"index": 0, "supported": full_quote}]})
    answer.search.nodes["target"].text = definition
    result = asyncio.run(answer.aanswer("How is the percentage calculated?"))
    assert "every independent factual clause" in model.prompts[0]
    assert len(model.prompts) == 2
    if full_quote:
        assert result["status"] == "answered"
        assert result["claims"][0]["excerpt"] == definition
    else:
        assert result["status"] == "abstained"
        assert result["reason_code"] == "no_verified_claims"
        assert not result["claims"]


def test_supported_part_is_returned_with_unresolved_part():
    answer, _ = fixture({"parts": [
        part(question="Who shoots?"),
        {"question": "How many free throws?", "claim": None, "citation_id": None, "excerpt": None},
    ]}, {"verdicts": [{"index": 0, "supported": True}]})
    result = asyncio.run(answer.aanswer("Who shoots and how many free throws?"))
    assert result["status"] == "partial"
    assert result["unresolved"] == ["How many free throws?"]
    assert result["answer"] == "A player in the game shoots."


def test_verification_response_must_cover_every_claim():
    answer, _ = fixture({"parts": [part()]}, {"verdicts": []})
    result = asyncio.run(answer.aanswer("Who shoots?"))
    assert result["status"] == "abstained"
    assert result["reason_code"] == "verification_failed"


def test_retrieval_abstention_skips_model_calls():
    answer, model = fixture({"parts": []}, {"verdicts": []})

    class Abstained:
        async def asearch(self, question, *, top_k):
            return {"retrieval_status": "abstained", "reason_code": "outside_corpus_scope", "hits": []}

    answer.guided = Abstained()
    result = asyncio.run(answer.aanswer("Who won?"))
    assert result["status"] == "abstained"
    assert result["reason_code"] == "outside_corpus_scope"
    assert model.prompts == []


def test_retrieval_stage_error_is_stable_and_includes_stage_timings():
    answer, model = fixture({"parts": []}, {"verdicts": []})

    class TimedOut:
        async def asearch(self, question, *, top_k):
            return {"retrieval_status": "error", "reason_code": "embedding_timeout",
                    "error_stage": "embedding", "hits": [], "planning_ms": 12.0,
                    "embedding_ms": 8000.0, "vector_query_ms": 0.0}

    answer.guided = TimedOut()
    from scripts.knowledge_answer import KnowledgeAnswerError

    with pytest.raises(KnowledgeAnswerError) as caught:
        asyncio.run(answer.aanswer("Who shoots?"))
    assert caught.value.reason_code == "embedding_timeout"
    assert caught.value.stage == "embedding"
    assert caught.value.retryable is True
    assert caught.value.timings["embedding_ms"] == 8000.0
    assert model.prompts == []


def test_answer_route_requires_dense_index():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    app = create_app(knowledge_search=KnowledgeSearch([node("first", "A rule.")]))

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/knowledge/answer", json={"question": "Who shoots?"})

    response = asyncio.run(request())
    assert response.status_code == 503


def test_answer_http_contract_returns_verified_claim(monkeypatch):
    from app.main import create_app
    import scripts.knowledge_guidance as guidance

    monkeypatch.setattr(guidance, "GuidedKnowledgeSearch", lambda search, model: Guided())
    model = Model({"parts": [part()]}, {"verdicts": [{"index": 0, "supported": True}]})
    search = Search([node("first", "Unrelated first chunk."),
                     node("target", "Technical free throws must be attempted by a player in the game.")])
    app = create_app(model=model, knowledge_search=search)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/knowledge/answer", json={"question": "Who shoots?"})

    response = asyncio.run(request())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["claims"][0]["citation_id"] == "target"
    assert body["claims"][0]["source_url"] == "https://example.com/rules"
    assert body["verification_ms"] >= 0
    assert body["total_ms"] >= body["elapsed_ms"] - 1


def test_answer_http_contract_exposes_stage_timeout(monkeypatch):
    from app.main import create_app
    import scripts.knowledge_guidance as guidance

    class TimedOut:
        async def asearch(self, question, *, top_k):
            return {"retrieval_status": "error", "reason_code": "planner_timeout",
                    "error_stage": "planner", "hits": [], "planning_ms": 5000.0,
                    "embedding_ms": 0.0, "vector_query_ms": 0.0}

    monkeypatch.setattr(guidance, "GuidedKnowledgeSearch", lambda search, model: TimedOut())
    search = Search([node("first", "A rule.")])
    app = create_app(model=Model({"parts": []}, {"verdicts": []}), knowledge_search=search)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/knowledge/answer", json={"question": "Who shoots?"})

    response = asyncio.run(request())
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "planner_timeout"
    assert response.json()["detail"]["stage"] == "planner"


def test_answer_http_total_timeout_keeps_active_stage_timings(monkeypatch):
    from app.main import create_app
    import scripts.knowledge_guidance as guidance
    class SlowGuided:
        async def asearch(self, question, *, top_k):
            await asyncio.sleep(1)
    monkeypatch.setattr(guidance, 'GuidedKnowledgeSearch', lambda search, model: SlowGuided())
    from scripts.provider_config import ProviderConfig
    monkeypatch.setenv('KNOWLEDGE_TOTAL_TIMEOUT_SECONDS', '999')
    app = create_app(model=Model({}, {}), knowledge_search=Search([node('first', 'A rule.')]),
                     provider_config=ProviderConfig(knowledge_total_timeout_seconds=.03))
    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            return await client.post('/knowledge/answer', json={'question': 'Who shoots?'})
    response = asyncio.run(request())
    detail = response.json()['detail']
    assert response.status_code == 504
    assert detail['code'] == 'total_timeout'
    assert detail['stage'] == 'total'
    assert detail['timings']['retrieval_ms'] > 0


def test_agent_knowledge_failure_preserves_stage_diagnostics(tmp_path):
    from app.agent_routes import KnowledgeGlossaryAdapter
    from app.main import create_app
    from scripts.knowledge_answer import KnowledgeAnswerError

    class FailedAnswerer:
        async def aanswer(self, question):
            raise KnowledgeAnswerError(
                "generation", "generation_timeout", retryable=True,
                timings={"generation_ms": 100.0, "retrieval_ms": 20.0},
            )

    app = create_app(agent_knowledge_adapter=KnowledgeGlossaryAdapter(None, answerer=FailedAnswerer()))

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/agent/answer", json={"question": "什么是进攻三秒违例"})

    response = asyncio.run(request())
    body = response.json()
    failure = body["steps"][0]["result"]
    assert failure["reason_code"] == "generation_timeout"
    assert failure["error_stage"] == "generation"
    assert failure["timings"] == {"generation_ms": 100.0, "retrieval_ms": 20.0}
    assert body["error_stage"] == "generation"
    assert body["reason_code"] == "generation_timeout"
    assert body["timings"]["s1_generation_ms"] == 100.0
    assert body["evidence"][0]["error_stage"] == "generation"




def test_frozen_answer_cases_have_valid_independent_labels():
    from pathlib import Path
    from scripts.evaluate_knowledge_answer import validate_cases

    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "metadata/knowledge_answer_cases_m21_v1.json").read_text())["cases"]
    validate_cases(cases)
    assert len(cases) == 30
    assert sum(case["answerable"] for case in cases) == 20


def test_m21a_holdout_cases_have_valid_independent_labels():
    from pathlib import Path
    from scripts.evaluate_knowledge_answer import validate_cases

    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "metadata/knowledge_answer_cases_m21a_v1.json").read_text())["cases"]
    validate_cases(cases)
    assert len(cases) == 30
    assert sum(case["answerable"] for case in cases) == 20


def test_m21b_holdout_cases_have_valid_independent_labels():
    from pathlib import Path
    from scripts.evaluate_knowledge_answer import validate_cases

    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "metadata/knowledge_answer_cases_m21b_v1.json").read_text())["cases"]
    validate_cases(cases)
    assert len(cases) == 30
    assert sum(case["answerable"] for case in cases) == 20
    old = json.loads((root / "metadata/knowledge_answer_cases_m21a_v1.json").read_text())["cases"]
    old_citations = {citation for case in old for citation in case["supporting_citation_ids"]}
    new_citations = {citation for case in cases for citation in case["supporting_citation_ids"]}
    assert not old_citations & new_citations


def test_answer_evaluator_checks_whole_answer_and_no_answer_cases():
    from scripts.evaluate_knowledge_answer import grade_response

    answerable = {"answerable": True, "supporting_citation_ids": ["target"]}
    no_answer = {"answerable": False, "supporting_citation_ids": []}
    assert grade_response(answerable, {"status": "answered", "claims": [
        {"citation_id": "target", "text": "supported"}], "unresolved": []}) == "manual_review"
    assert grade_response(answerable, {"status": "partial", "claims": [
        {"citation_id": "target"}], "unresolved": ["missing"]}) == "incomplete"
    assert grade_response(no_answer, {"status": "abstained", "claims": []}) == "correct_refusal"
    assert grade_response(no_answer, {"status": "answered", "claims": [
        {"citation_id": "target"}]}) == "false_answer"


def test_evidence_expansion_keeps_ranked_answer_chunk_beyond_old_budget():
    from scripts.knowledge_answer import expand_evidence

    nodes = []
    for index in range(12):
        nodes.append(node(f"page:{index}", f"chunk {index}", page=1))
    search = Search(nodes)
    hits = [{"citation_id": f"page:{index}", "source_id": "nba-rulebook-2025-26", "page": 1,
             "excerpt": f"chunk {index}"} for index in range(8)]
    evidence = expand_evidence(search, hits)
    assert len(evidence) == 12
    assert evidence[-1]["citation_id"] == "page:11"


def test_evidence_expansion_respects_character_budget():
    from scripts.knowledge_answer import expand_evidence

    nodes = [node(f"page:{index}", "x" * 9000, page=1) for index in range(4)]
    search = Search(nodes)
    hits = [{"citation_id": f"page:{index}", "source_id": "nba-rulebook-2025-26", "page": 1,
             "excerpt": nodes[index].text} for index in range(4)]
    evidence = expand_evidence(search, hits)
    assert len(evidence) == 2
    assert sum(len(item["excerpt"]) for item in evidence) <= 24000
