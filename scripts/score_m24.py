"""Static scoring and failure attribution for the M2.4 production run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_knowledge_answer import validate_cases
from scripts.knowledge_answer import canonical_excerpt
from scripts.knowledge_search import load_nodes

RUNTIME_CODES = {
    'planner_timeout', 'embedding_timeout', 'embedding_error', 'retrieval_timeout',
    'generation_timeout', 'verification_timeout', 'total_timeout', 'runner_error',
    'run_cancelled',
}
CANDIDATE_STAGES = (
    'original_dense_top50', 'rewritten_dense_top50', 'bm25_top20', 'rrf_ranking',
    'dense_threshold', 'domain_filter', 'dedup', 'final_top_k',
)


def _outcome(case):
    return case.get('expected_outcome') or ('answer' if case['answerable'] else 'refusal')


def _location(metadata):
    location = metadata.get('page') if metadata.get('page') is not None else metadata.get('anchor')
    return metadata.get('id'), location


def _claim_provenance_valid(claim, nodes):
    node = nodes.get(claim.get('citation_id'))
    if node is None:
        return False
    metadata = node.metadata
    fields = {
        'source_id': 'id', 'source_url': 'source_url', 'source_version': 'version',
        'sha256': 'sha256', 'page': 'page', 'anchor': 'anchor',
    }
    if any(claim.get(output) != metadata.get(source) for output, source in fields.items()):
        return False
    excerpt = claim.get('excerpt')
    return isinstance(excerpt, str) and bool(excerpt) and canonical_excerpt(node.text, excerpt) is not None


def _gold_location_match(claim, case, nodes):
    node = nodes.get(claim.get('citation_id'))
    if node is None:
        return False
    gold_locations = {
        _location(nodes[citation_id].metadata)
        for citation_id in case.get('supporting_citation_ids', []) if citation_id in nodes
    }
    return _location(node.metadata) in gold_locations


def _stage_events(observation):
    return (observation or {}).get('events', {})


def _candidate_hit(events, gold_ids):
    if 'retrieval_result' not in events:
        return None
    ids = [
        item.get('citation_id')
        for stage in CANDIDATE_STAGES
        for item in events.get(stage, [])
        if item.get('citation_id')
    ]
    return bool(set(ids) & gold_ids)


def _evidence_hit(events, gold_ids):
    if 'bounded_evidence' not in events:
        return None
    return bool({item.get('citation_id') for item in events['bounded_evidence']} & gold_ids)


def _failure(result, case, events, candidate, evidence, provenance_valid,
             gold_location_match, review):
    code = result.get('error_code')
    status = result.get('status')
    claims = result.get('claims', [])
    if code in RUNTIME_CODES:
        return 'provider_runtime_failure'
    if result.get('failure_class') == 'generation_failure' or code in {'generation_error', 'invalid_generation'} or result.get('reason_code') == 'invalid_generation':
        return 'generation_failure'
    if result.get('failure_class') == 'citation_validation_failure' or code in {
        'verification_error', 'verification_failed', 'no_verified_claims',
    } or result.get('reason_code') == 'verification_failed':
        return 'citation_validation_failure'
    if status == 'error':
        return 'execution_failure'

    if _outcome(case) == 'answer':
        if candidate is None:
            return 'unobserved'
        if candidate is False:
            return 'retrieval_failure'
        if not claims:
            gold = set(case.get('supporting_citation_ids', []))
            rrf = {item.get('citation_id') for item in events.get('rrf_ranking', [])}
            final = {item.get('citation_id') for item in events.get('final_top_k', [])}
            if gold & rrf and not gold & final:
                return 'ranking_cutoff_failure'
            if gold & final and evidence is False:
                return 'evidence_assembly_failure'
            return 'generation_failure'

    if claims and provenance_valid is False:
        return 'citation_validation_failure'
    if claims and provenance_valid and gold_location_match is False:
        validation = events.get('citation_validation', {})
        if (validation.get('rejected_claim_count', 0) == 0 and review
                and review.get('facts_complete') and all(review.get('claim_support', []))):
            return 'evaluation_contract_issue'
        return 'citation_validation_failure'
    if claims and review and any(not supported for supported in review.get('claim_support', [])):
        verdicts = events.get('verification_result', {}).get('verdicts', [])
        if len(verdicts) == len(claims) and all(verdict.get('supported') is True for verdict in verdicts):
            return 'verification_failure'
    if _outcome(case) == 'answer' and result.get('unresolved'):
        return 'generation_failure'
    if _outcome(case) == 'answer' and claims and review and (
        not review.get('facts_complete') or not all(review.get('claim_support', []))
    ):
        return 'generation_failure'
    return None


def score_report(cases, report, reviews, nodes):
    validate_cases(cases)
    if not reviews.get('reviewer') or not isinstance(reviews.get('cases'), dict):
        raise ValueError('reviewer and cases required')
    rows = report.get('cases', [])
    cases_by_id = {case['id']: case for case in cases}
    if len(rows) != len(cases) or {row.get('id') for row in rows} != set(cases_by_id):
        raise ValueError('report cases do not match labels')
    expected_hash = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if report.get('case_sha256') not in (None, expected_hash):
        raise ValueError('case hash mismatch')

    review_ids = {case['id'] for case in cases if _outcome(case) != 'refusal'}
    review_complete = review_ids.issubset(reviews['cases'])
    review_provenance = reviews.get('reviewer_type', 'unverified')
    human_review_confirmed = (
        review_provenance == 'human' and reviews.get('human_review_confirmed') is True
    )
    out = {
        'schema_version': 2,
        'case_sha256': expected_hash,
        'reviewer': reviews['reviewer'],
        'review_provenance': review_provenance,
        'review_complete': review_complete,
        'human_review_confirmed': human_review_confirmed,
        'answer_total': 0,
        'answer_semantic_supported': 0,
        'refusal_total': 0,
        'refusal_correct': 0,
        'premise_denial_total': 0,
        'premise_denial_supported': 0,
        'candidate_recall_cases': 0,
        'candidate_recall_observed_cases': 0,
        'candidate_recall_unknown_cases': 0,
        'evidence_coverage_cases': 0,
        'evidence_coverage_observed_cases': 0,
        'evidence_coverage_unknown_cases': 0,
        'citation_valid': 0,
        'gold_location_match_cases': 0,
        'verification_supported': 0,
        'verification_observed_cases': 0,
        'verification_unknown_cases': 0,
        'provider_runtime_failures': 0,
        'execution_failure_count': 0,
        'unsupported_claim_count': 0,
        'metadata_invalid_count': 0,
        'failure_counts': {},
        'quality_gate_pass': False,
        'promotion_ready': False,
        'cases': {},
    }

    for row in rows:
        case = cases_by_id[row['id']]
        outcome = _outcome(case)
        out[f'{outcome}_total'] += 1
        result = row.get('result', row)
        claims = result.get('claims', [])
        events = _stage_events(row.get('observation'))
        gold = set(case.get('supporting_citation_ids', []))
        candidate = _candidate_hit(events, gold) if gold else None
        evidence = _evidence_hit(events, gold) if gold else None
        if outcome == 'answer':
            if candidate is None:
                out['candidate_recall_unknown_cases'] += 1
            else:
                out['candidate_recall_observed_cases'] += 1
                out['candidate_recall_cases'] += int(candidate)
            if evidence is None:
                out['evidence_coverage_unknown_cases'] += 1
            else:
                out['evidence_coverage_observed_cases'] += 1
                out['evidence_coverage_cases'] += int(evidence)

        review = reviews['cases'].get(row['id'])
        provenance_valid = bool(claims) and all(
            _claim_provenance_valid(claim, nodes) for claim in claims
        )
        gold_location_match = bool(claims) and all(
            _gold_location_match(claim, case, nodes) for claim in claims
        )
        if claims and outcome != 'refusal':
            if not isinstance(review, dict) or not isinstance(review.get('facts_complete'), bool):
                raise ValueError(f'facts_complete review missing for {row["id"]}')
            support = review.get('claim_support')
            if not isinstance(support, list) or len(support) != len(claims) or any(not isinstance(value, bool) for value in support):
                raise ValueError(f'claim_support review missing for {row["id"]}')
            out['unsupported_claim_count'] += sum(not value for value in support)
            out['metadata_invalid_count'] += int(not provenance_valid)
        elif review is not None and outcome == 'refusal':
            raise ValueError(f'unexpected review for {row["id"]}')

        complete = bool(claims) and result.get('status') == 'answered' and not result.get('unresolved')
        verification = events.get('verification_result')
        verdicts = verification.get('verdicts') if isinstance(verification, dict) else None
        verifier_supported = (
            bool(claims) and isinstance(verdicts, list) and len(verdicts) == len(claims)
            and all(verdict.get('supported') is True for verdict in verdicts)
        ) if verdicts is not None else None
        semantic = (
            complete and provenance_valid and gold_location_match
            and bool(review and review.get('facts_complete'))
            and all(review.get('claim_support', []))
        ) if outcome != 'refusal' else False
        failure = _failure(
            result, case, events, candidate, evidence, provenance_valid,
            gold_location_match, review,
        )
        if failure:
            out['failure_counts'][failure] = out['failure_counts'].get(failure, 0) + 1
        runtime = failure == 'provider_runtime_failure'
        execution = runtime or result.get('status') == 'error' or failure in {
            'citation_validation_failure', 'verification_failure',
        }
        out['provider_runtime_failures'] += int(runtime)
        out['execution_failure_count'] += int(execution)

        if outcome == 'answer':
            out['answer_semantic_supported'] += int(semantic)
            out['citation_valid'] += int(provenance_valid)
            out['gold_location_match_cases'] += int(gold_location_match)
            if verifier_supported is None:
                out['verification_unknown_cases'] += 1
            else:
                out['verification_observed_cases'] += 1
                out['verification_supported'] += int(verifier_supported)
        elif outcome == 'premise_denial':
            out['premise_denial_supported'] += int(semantic)
        else:
            out['refusal_correct'] += int(not execution and result.get('status') == 'abstained' and not claims)

        out['cases'][row['id']] = {
            'classification': failure or (
                'answer_semantic_supported' if semantic
                else ('answer_incomplete' if outcome != 'refusal' else 'correct_refusal')
            ),
            'candidate_recall': candidate,
            'evidence_coverage': evidence,
            'metadata_valid': provenance_valid,
            'gold_location_match': gold_location_match,
            'verification_supported': verifier_supported,
            'semantic_supported': semantic,
            'execution_failure': execution,
            'runtime_failure': runtime,
            'error_code': result.get('error_code'),
            'error_stage': result.get('error_stage'),
        }

    out['quality_gate_pass'] = (
        out['answer_total'] == 20
        and out['answer_semantic_supported'] >= 17
        and out['refusal_correct'] == out['refusal_total']
        and out['premise_denial_supported'] == out['premise_denial_total']
        and out['provider_runtime_failures'] <= 2
        and out['unsupported_claim_count'] == 0
        and out['metadata_invalid_count'] == 0
    )
    out['promotion_ready'] = out['quality_gate_pass'] and review_complete and human_review_confirmed
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())['cases']
    report = json.loads(args.report.read_text())
    reviews = json.loads(args.reviews.read_text())
    nodes = {node.metadata['citation_id']: node for node in load_nodes(args.index)}
    scored = score_report(cases, report, reviews, nodes)
    args.output.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + '\n')
    print(f'Report: {args.output}; quality_gate_pass={scored["quality_gate_pass"]}')


if __name__ == '__main__':
    main()
