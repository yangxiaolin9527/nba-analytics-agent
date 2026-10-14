"""Publication contract tests with fixed provider outputs and artificial evidence."""
import asyncio

import pytest

from scripts.knowledge_answer import KnowledgeAnswer
from tests.test_knowledge_answer import Model, Search, node, fixture, part


CHARGING = 'NBA 规则中，什么是带球撞人（charging）？'
ENGLISH_CLAIM = 'A dribbler shall not charge into an opponent who has established a legal guarding position.'
EXCERPT = 'A dribbler shall not (1) charge into an opponent who has established a legal \nguarding position'


class RulesGuided:
    async def asearch(self, question, *, top_k):
        return {'retrieval_status': 'candidates_found', 'reason_code': None,
                'intent': 'rules', 'question_scope': 'single',
                'hits': [{'citation_id': 'charging-source', 'excerpt': EXCERPT,
                          'source_id': 'nba-rulebook-2025-26', 'page': 42,
                          'source_url': 'https://example.com/rules',
                          'source_version': '2025-26', 'sha256': 'fixture',
                          'anchor': None, 'heading': None}]}


def charging_answer(claim, strategy='baseline'):
    model = Model({'parts': [part(question=CHARGING, claim=claim,
                                 citation='charging-source', excerpt=EXCERPT)]},
                  {'verdicts': [{'index': 0, 'supported': True}]})
    answerer = KnowledgeAnswer(Search([node('charging-source', EXCERPT)]), model,
                              guided=RulesGuided(), generation_prompt_strategy=strategy)
    return answerer, model


@pytest.mark.parametrize('strategy', ['baseline', 'concise'])
def test_chinese_question_never_publishes_english_prose_as_answered(strategy):
    answerer, model = charging_answer(ENGLISH_CLAIM, strategy)
    result = asyncio.run(answerer.aanswer(CHARGING))
    assert result['status'] == 'abstained'
    assert result['reason_code'] == 'answer_language_mismatch'
    assert result['answer'] is None and result['claims'] == []
    assert CHARGING in result['unresolved']
    assert result['answer_language']['expected'] == 'zh'
    assert result['answer_language']['rejected_count'] == 1
    # Language rejection adds no model call; it happens before support checking.
    assert len(model.prompts) == 1


def test_one_chinese_label_does_not_disguise_an_english_answer():
    answerer, _ = charging_answer('定义：' + ENGLISH_CLAIM)
    result = asyncio.run(answerer.aanswer(CHARGING))
    assert result['status'] == 'abstained'
    assert result['reason_code'] == 'answer_language_mismatch'


def test_quote_marks_do_not_disguise_an_unrequested_english_answer():
    answerer, _ = charging_answer('“' + ENGLISH_CLAIM + '”')
    result = asyncio.run(answerer.aanswer(CHARGING))
    assert result['status'] == 'abstained'
    assert result['reason_code'] == 'answer_language_mismatch'


@pytest.mark.parametrize('claim', [
    '运球者不得撞向已经建立合法防守位置的对手。',
    '带球撞人（charging）涉及运球者撞向已建立合法防守位置的对手。',
    '原文写道：“A dribbler shall not charge into an opponent”。中文解释是运球者不得撞向合法防守者。',
])
def test_chinese_explanation_keeps_english_terms_and_verbatim_quote(claim):
    answerer, _ = charging_answer(claim)
    result = asyncio.run(answerer.aanswer(CHARGING))
    assert result['status'] == 'answered'
    assert result['answer'] == claim
    assert result['claims'][0]['excerpt'] == EXCERPT
    assert result['answer_language']['parts'][0]['status'] == 'script_compatible'


@pytest.mark.parametrize('question,claim,expected,status', [
    ('What is charging?', ENGLISH_CLAIM, 'en', 'answered'),
    ('什么是带球撞人？请用英文回答。', ENGLISH_CLAIM, 'en', 'answered'),
    ('什么是带球撞人？请用英文回答。', '运球者不得撞向合法防守者。', 'en', 'abstained'),
    ('What is charging? Answer in Chinese.', ENGLISH_CLAIM, 'zh', 'abstained'),
    ('请用中文解释 charging。', ENGLISH_CLAIM, 'zh', 'abstained'),
])
def test_explicit_answer_language_overrides_question_language(question, claim, expected, status):
    answerer, _ = charging_answer(claim)
    result = asyncio.run(answerer.aanswer(question))
    assert result['status'] == status
    assert result['answer_language']['expected'] == expected


@pytest.mark.parametrize('question,claim', [
    ('eFG% 的公式是什么？', '(FGM + 0.5 * 3PM) / FGA * 100'),
    ('这个指标的英文全称是什么？', 'effective field goal percentage'),
    ('只给出带球撞人的英文原文。', EXCERPT),
])
def test_requested_formula_term_or_original_quote_is_not_rejected(question, claim):
    answerer, _ = charging_answer(claim)
    result = asyncio.run(answerer.aanswer(question))
    assert result['status'] == 'answered'
    assert result['answer_language']['parts'][0]['status'] == 'unverified'


@pytest.mark.parametrize('question', [
    '请用日语解释 charging。',
    '充電ではなくバスケットボールの charging とは何ですか？',
    'Qu’est-ce que charging ?',
])
def test_unsupported_language_is_not_claimed_as_verified(question):
    answerer, _ = charging_answer(ENGLISH_CLAIM)
    result = asyncio.run(answerer.aanswer(question))
    assert result['answer_language']['expected'] is None
    assert result['answer_language']['parts'][0]['status'] == 'unverified'


@pytest.mark.parametrize('question', [
    '请用中文回答，也请用英文回答。',
    '术语 NBA charging',
])
def test_ambiguous_language_has_no_verified_language_claim(question):
    answerer, _ = charging_answer(ENGLISH_CLAIM)
    result = asyncio.run(answerer.aanswer(question))
    assert result['answer_language']['expected'] is None
    assert result['answer_language']['parts'][0]['status'] == 'unverified'


def test_wrong_language_subquestion_leaves_verified_chinese_part_partial():
    parts = [part(question='谁执行罚球？', claim='由场上球员执行罚球。'),
             part(question='球员必须在场上吗？', claim='A player in the game shoots.')]
    answerer, model = fixture({'parts': parts}, {'verdicts': [{'index': 0, 'supported': True}]})
    result = asyncio.run(answerer.aanswer('谁执行罚球？球员必须在场上吗？'))
    assert result['status'] == 'partial'
    assert result['reason_code'] == 'answer_language_mismatch'
    assert result['answer'] == '由场上球员执行罚球。'
    assert result['unresolved'] == ['球员必须在场上吗？']
    assert len(model.prompts) == 2


@pytest.mark.parametrize('question,claim', [
    ('请用中文解释原文的含义。', ENGLISH_CLAIM),
    ('带球撞人的原文是什么？请用中文解释其含义。', ENGLISH_CLAIM),
    ('请用中文解释 eFG% 的公式。', 'The formula is eFG% = FGM / FGA.'),
    ('Explain charging in English.', '带球撞人（charging）涉及运球者撞向合法防守者。'),
])
def test_literal_mentions_and_formulas_do_not_bypass_prose_language(question, claim):
    answerer, _ = charging_answer(claim)
    result = asyncio.run(answerer.aanswer(question))
    assert result['status'] == 'abstained'
    assert result['reason_code'] == 'answer_language_mismatch'


def test_original_text_alone_does_not_satisfy_requested_chinese_explanation():
    answerer, _ = charging_answer(EXCERPT)
    result = asyncio.run(answerer.aanswer('请用中文解释这段原文的含义。'))
    assert result['status'] == 'abstained'
    assert result['reason_code'] == 'answer_language_mismatch'


@pytest.mark.parametrize('claim,status', [(ENGLISH_CLAIM, 'incomplete'),
                                       ('运球者不得撞向合法防守者。', 'complete')])
def test_unified_model_agent_preserves_knowledge_language_result(tmp_path, claim, status):
    from app.agent_routes import KnowledgeGlossaryAdapter
    from tests.test_unified_agent import unified_app
    from tests.test_semantic_routing import request

    answerer, _ = charging_answer(claim)
    app, _ = unified_app(tmp_path, [('knowledge_glossary', {'question': CHARGING})],
                         knowledge=KnowledgeGlossaryAdapter(None, answerer=answerer))
    body = request(app, '/agent/answer', {'question': CHARGING, 'routing_mode': 'model'}).json()
    assert body['status'] == status
    assert body['steps'][0]['capability'] == 'knowledge_glossary'
    result = body['steps'][0]['result']
    assert result['answer_language']['expected'] == 'zh'
    if status == 'incomplete':
        assert body['reason_code'] == 'answer_language_mismatch'
        assert body['evidence'][0]['knowledge']['unresolved'] == [CHARGING]
        assert not body['answer']
    else:
        assert body['answer'] == claim


@pytest.mark.parametrize('claim,status', [(ENGLISH_CLAIM, 'abstained'),
                                       ('运球者不得撞向合法防守者。', 'answered')])
def test_direct_knowledge_endpoint_preserves_language_result(monkeypatch, claim, status):
    from app.main import create_app
    from scripts.provider_config import ProviderConfig
    from tests.test_semantic_routing import request
    import scripts.knowledge_guidance as guidance

    answerer, model = charging_answer(claim)
    monkeypatch.setattr(guidance, 'GuidedKnowledgeSearch', lambda search, model, **kwargs: RulesGuided())
    app = create_app(model=model, knowledge_search=answerer.search, provider_config=ProviderConfig.empty())
    body = request(app, '/knowledge/answer', {'question': CHARGING}).json()
    assert body['status'] == status
    assert body['answer_language']['expected'] == 'zh'
    if status == 'abstained':
        assert body['reason_code'] == 'answer_language_mismatch'
        assert body['answer'] is None and body['unresolved'] == [CHARGING]
    else:
        assert body['answer'] == claim
