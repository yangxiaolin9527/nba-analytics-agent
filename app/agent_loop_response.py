"""Convert validated loop observations into the shared answer contract."""
from dataclasses import asdict

from app.agent_routes import AgentResponse, RouteStep, StepResult, _bundle
from scripts.answer_renderer import render_tool_answer
from scripts.tracing import trace_id


def loop_response(result, task, *, request_id, tracer=None):
    parts, steps, evidence, warnings, timings = [], [], [], [], {}
    for observation in result.observations:
        step = RouteStep(id=observation.id, capability=observation.capability,
                         arguments=observation.arguments)
        output = observation.result or {'status': observation.status,
                                        'reason_code': observation.reason_code}
        bundle = _bundle(step, observation.status, output, observation.error)
        evidence.append(bundle)
        steps.append(StepResult(id=observation.id, capability=observation.capability,
                                status=observation.status, result=output,
                                error=observation.error, evidence=bundle))
        timings[f'{observation.id}_ms'] = observation.elapsed_ms
        if observation.reason_code:
            warnings.append(observation.reason_code)
        if observation.status in {'complete', 'partial'}:
            if observation.capability == 'knowledge_glossary':
                answer = output.get('answer')
                if isinstance(answer, str) and answer.strip():
                    parts.append(answer)
            else:
                rendered = render_tool_answer(observation.capability, observation.arguments,
                    output, evidence=bundle.structured.model_dump(mode='json'))
                parts.append(rendered.text)
                warnings.extend(rendered.warnings)
    from app.agent_composition import summarize_comparison
    comparison = summarize_comparison(task.comparison_step_ids, {
        o.requirement_id: {'capability': o.capability, 'arguments': o.arguments,
                          'status': o.status, 'result': o.result}
        for o in result.observations}, tracer=tracer)
    if comparison:
        parts.append(comparison['text'])
    return AgentResponse(status=result.status, request_id=request_id, trace_id=trace_id(),
        route=task.route, answer='\n'.join(parts) or None, steps=steps, evidence=evidence,
        warnings=list(dict.fromkeys(warnings)), timings=timings,
        reason_code=None if result.status == 'complete' else result.termination_reason,
        error_stage=next(((o.result or {}).get('error_stage') for o in result.observations
                          if o.status == 'failed'), None),
        execution_mode='bounded_loop', actions=list(result.actions),
        observations=[asdict(o) for o in result.observations], decisions=result.decisions,
        tool_calls=result.tool_calls, termination_reason=result.termination_reason,
        decision_failures=list(result.decision_failures), comparison=comparison)
