"""Stable cross-run metric envelope; missing measurements remain explicit."""
from __future__ import annotations

METRICS = ('answer_total','answer_semantic_supported','facts_complete','citation_valid',
           'verification_supported','candidate_recall_cases','evidence_coverage_cases',
           'refusal_total','refusal_correct','premise_denial_total','premise_denial_supported',
           'provider_runtime_failures','execution_failure_count','unsupported_claim_count',
           'metadata_invalid_count','quality_gate_pass','promotion_ready','semantic_supported')
ORACLE_NOT_APPLICABLE = {'candidate_recall_cases','evidence_coverage_cases'}

def normalize_report(report: dict, *, kind: str, case_sha256: str | None) -> dict:
    source = report.get('metrics', report)
    metrics = {key: source.get(key) for key in METRICS}
    availability = {}
    for key, value in metrics.items():
        if value is not None:
            availability[key] = 'recorded'
        elif kind == 'oracle_context' and key in ORACLE_NOT_APPLICABLE:
            availability[key] = 'not_applicable'
        else:
            availability[key] = 'not_recorded'
    return {'schema_version': 1, 'evaluation_kind': kind, 'case_sha256': case_sha256,
            'metrics': metrics, 'availability': availability,
            'cases': report.get('cases', {}), 'source_report': report}
