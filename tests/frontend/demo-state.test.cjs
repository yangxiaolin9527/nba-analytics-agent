const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = path.join(__dirname, '../../static/demo-state.js');
const api = fs.existsSync(source) ? require(source) : {};

function storage() {
  const data = new Map();
  return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
}
function store(options = {}) {
  assert.equal(typeof api.SessionHistory, 'function', 'session history must exist');
  return new api.SessionHistory(options);
}
function turn(name, index, overrides = {}) {
  return { view: 'agent', routingMode: 'model', mode: 'live', question: name,
    payload: { status: 'complete', answer: name + ' answer', conversation_id: 'server-a', turn_index: index }, ...overrides };
}

test('one session owns three turns and selected responses survive reload', () => {
  const disk = storage(); const history = store({ storage: disk });
  const session = history.record(turn('Jordan', 1));
  history.record(turn('Rodman', 2), session.id);
  history.record(turn('Malone', 3), session.id);
  assert.equal(history.sessions.length, 1);
  assert.deepEqual(history.sessions[0].turns.map(t => t.question), ['Jordan', 'Rodman', 'Malone']);
  history.select(session.id, session.turns[0].id);
  const restored = store({ storage: disk });
  assert.equal(restored.selectedTurn(session.id).payload.answer, 'Jordan answer');
  assert.equal(restored.sessions[0].turns.length, 3);
});

test('conversation turns preserve a numeric creation time for ChatGPT-like labels', () => {
  const history = store();
  const session = history.record(turn('Timed', 1, { createdAt: 1700000000000 }));
  assert.equal(session.turns[0].createdAt, 1700000000000);
});

test('new session never erases old transcript and reset server context creates a new session', () => {
  const history = store(); const first = history.record(turn('Jordan', 1));
  history.record(turn('Rodman', 2), first.id);
  const second = history.record(turn('Malone', 1, { payload: { conversation_id: 'server-b', turn_index: 1 } }));
  assert.equal(history.sessions.length, 2);
  assert.equal(history.get(first.id).turns.length, 2);
  const reset = history.record(turn('Expired', 1, { payload: { conversation_id: 'server-a', turn_index: 1, conversation_context_status: 'reset' } }), first.id);
  assert.notEqual(reset.id, first.id);
  assert.equal(history.get(first.id).turns.length, 2);
  assert.equal(second.turns[0].question, 'Malone');
});

test('view and routing mode cannot merge into another session', () => {
  const history = store(); const session = history.record(turn('Jordan', 1));
  history.record(turn('Stats', 2, { view: 'stats' }), session.id);
  history.record(turn('Rules', 3, { routingMode: 'rules' }), session.id);
  assert.equal(history.sessions.length, 3);
  const replay = history.record(turn('Replay', 1, { mode: 'replay', payload: { status: 'complete', answer: 'replay', conversation_id: null, turn_index: 1 } }));
  const live = history.record(turn('Live', 1, { mode: 'live', payload: { status: 'complete', answer: 'live', conversation_id: 'server-live', turn_index: 1 } }), replay.id);
  assert.notEqual(live.id, replay.id);
});

test('selecting a session without an explicit turn preserves its persisted selection', () => {
  const history = store(); const session = history.record(turn('Jordan', 1));
  history.record(turn('Malone', 2), session.id); history.select(session.id, session.turns[0].id);
  assert.equal(history.selectedTurn(session.id).question, 'Jordan');
});

test('delete, clear, eviction and malformed storage keep consistent ownership', () => {
  const disk = storage(); const history = store({ storage: disk, maxSessions: 2 });
  const first = history.record(turn('First', 1));
  const second = history.record(turn('Second', 1));
  history.record(turn('Third', 1));
  assert.equal(history.get(first.id), null);
  history.remove(second.id);
  assert.equal(store({ storage: disk }).sessions.length, 1);
  history.clear(); assert.equal(store({ storage: disk }).sessions.length, 0);
  disk.setItem(api.STORAGE_KEY, '{broken');
  assert.equal(store({ storage: disk }).sessions.length, 0);
  disk.setItem(api.STORAGE_KEY, JSON.stringify({ version: 1, sessions: [{ id: 'bad', turns: [{}] }] }));
  assert.equal(store({ storage: disk }).sessions.length, 0);
});

test('storage exceptions do not prevent recording or selecting a result', () => {
  const history = store({ storage: { getItem() { throw Error('disabled'); }, setItem() { throw Error('quota'); } } });
  const session = history.record(turn('Jordan', 1));
  assert.equal(history.selectedTurn(session.id).question, 'Jordan');
  assert.equal(history.persistenceError, true);
});

test('turn and byte limits evict whole oldest sessions', () => {
  const history = store(); const first = history.record(turn('First', 1));
  for (let i = 2; i <= 100; i++) history.record(turn('Turn ' + i, i), first.id);
  assert.equal(history.get(first.id).turns.length, 100);
  const second = history.record(turn('New', 1));
  assert.equal(history.get(first.id), null);
  assert.equal(history.get(second.id).turns.length, 1);
  history.record(turn('Too large', 1, { payload: { answer: 'x'.repeat(5 * 1024 * 1024) } }));
  assert.equal(history.sessions.length, 0);
});

test('nested timeout timings expose generation failure and unexecuted verification', () => {
  assert.equal(typeof api.knowledgeStages, 'function');
  const rows = api.knowledgeStages({ status: 'failed', error_stage: 'generation', reason_code: 'generation_timeout',
    timings: { retrieval_ms: 1533.455, generation_ms: 10003.758, verification_ms: 0 } });
  assert.equal(rows.find(r => r.stage === 'retrieval').status, 'complete');
  assert.equal(rows.find(r => r.stage === 'generation').status, 'failed');
  assert.equal(rows.find(r => r.stage === 'generation').meta, 10003.758);
  assert.equal(rows.find(r => r.stage === 'verification').status, 'skipped');
});

test('runtime stage statuses override timings and old abstentions do not fabricate success', () => {
  assert.equal(typeof api.knowledgeStages, 'function');
  const rows = api.knowledgeStages({ status: 'failed', error_stage: 'total', timings: { generation_ms: 1 },
    stage_statuses: { retrieval: 'complete', generation: 'failed', citation_validation: 'skipped', verification: 'skipped' } });
  assert.equal(rows.find(r => r.stage === 'generation').status, 'failed');
  const old = api.knowledgeStages({ status: 'abstained', reason_code: 'no_verified_claims', generation_ms: 1, verification_ms: 0 });
  assert.equal(old.find(r => r.stage === 'verification').status, 'skipped');
});

test('estimated knowledge progress is explicitly pending/running before runtime response', () => {
  const rows = api.estimatedKnowledgeStages('generation');
  assert.equal(rows.find(r => r.stage === 'retrieval').status, 'complete');
  assert.equal(rows.find(r => r.stage === 'generation').status, 'running');
  assert.equal(rows.find(r => r.stage === 'verification').status, 'pending');
});

test('execution event rows collapse a stage to its terminal state and keep elapsed time', () => {
  assert.equal(typeof api.executionEventRows, 'function');
  const rows = api.executionEventRows([
    { seq: 1, stage: 'planning', status: 'running', message: '模型规划' },
    { seq: 2, stage: 'planning', status: 'complete', message: '模型规划完成', elapsed_ms: 541 },
    { seq: 3, stage: 'tool', status: 'running', message: '调用 knowledge_glossary', metadata: { step_id: 's1', capability: 'knowledge_glossary' } },
    { seq: 4, stage: 'tool', status: 'failed', message: '失败 knowledge_glossary', elapsed_ms: 12000,
      metadata: { step_id: 's1', capability: 'knowledge_glossary', reason_code: 'no_verified_claims' } },
    { seq: 5, stage: 'retrieval', status: 'running', message: '知识检索' },
    { seq: 6, stage: 'retrieval', status: 'complete', message: '知识检索完成', elapsed_ms: 133 },
    { seq: 7, stage: 'verification', status: 'skipped', message: '语义验证未执行' },
    { seq: 8, stage: 'result', status: 'failed', message: '知识回答失败' },
  ]);
  assert.deepEqual(rows.map(row => row.status), ['complete', 'failed', 'complete', 'skipped']);
  assert.deepEqual(rows.map(row => row.key), ['planning', 'tool:s1', 'retrieval', 'verification']);
  assert.equal(rows[0].elapsed_ms, 541);
  assert.equal(rows[1].metadata.reason_code, 'no_verified_claims');
  assert.equal(rows.some(row => row.stage === 'result'), false);
});

test('execution event rows separate nested query planning from agent planning', () => {
  const rows = api.executionEventRows([
    { seq: 1, stage: 'planning', status: 'complete', message: '模型规划完成', metadata: { route: 'mixed' } },
    { seq: 2, stage: 'planning', status: 'running', message: '查询规划', metadata: { intent: 'rules' } },
    { seq: 3, stage: 'planning', status: 'complete', message: '查询规划完成', elapsed_ms: 222, metadata: { intent: 'rules' } },
  ]);
  assert.deepEqual(rows.map(row => row.key), ['planning']);
  assert.deepEqual(rows.map(row => row.status), ['complete']);
});

test('execution event rows omit internal evidence assembly from user-facing stages', () => {
  const rows = api.executionEventRows([
    { seq: 1, stage: 'planning', status: 'complete', message: '查询规划完成' },
    { seq: 2, stage: 'retrieval', status: 'complete', message: '知识检索完成', elapsed_ms: 80 },
    { seq: 3, stage: 'evidence', status: 'complete', message: '整理相关证据完成', elapsed_ms: 4 },
    { seq: 4, stage: 'generation', status: 'complete', message: '答案生成完成', elapsed_ms: 400 },
  ]);
  assert.deepEqual(rows.map(row => row.stage), ['planning', 'retrieval', 'generation']);
});

test('knowledge stages are children of the knowledge tool in Agent execution', () => {
  const rows = api.executionEventRows([
    { seq: 1, stage: 'planning', status: 'complete', message: '模型规划完成' },
    { seq: 2, stage: 'tool', status: 'running', message: '调用 knowledge_glossary', metadata: { step_id: 's2', capability: 'knowledge_glossary' } },
    { seq: 3, stage: 'retrieval', status: 'complete', message: '知识检索完成', elapsed_ms: 100 },
    { seq: 4, stage: 'generation', status: 'complete', message: '答案生成完成', elapsed_ms: 200 },
    { seq: 5, stage: 'tool', status: 'failed', message: '失败 knowledge_glossary', metadata: { step_id: 's2', capability: 'knowledge_glossary', reason_code: 'no_verified_claims' } },
    { seq: 6, stage: 'verification', status: 'skipped', message: '语义验证未执行' },
  ]);
  const tool = rows.find(row => row.key === 'tool:s2');
  assert.equal(tool.status, 'failed');
  assert.equal(rows.find(row => row.stage === 'retrieval').parent_key, 'tool:s2');
  assert.equal(rows.find(row => row.stage === 'verification').parent_key, 'tool:s2');
});

test('loop decisions, repeated tool attempts, observations and termination remain distinct', () => {
  const rows = api.executionEventRows([
    { stage: 'loop_decision', status: 'running', metadata: { decision_number: 1 } },
    { stage: 'loop_decision', status: 'complete', metadata: { decision_number: 1 } },
    { stage: 'tool', status: 'failed', metadata: { step_id: 's1', decision_number: 1 } },
    { stage: 'observation', status: 'failed', metadata: { observation_id: 'o1' } },
    { stage: 'loop_decision', status: 'complete', metadata: { decision_number: 2 } },
    { stage: 'tool', status: 'complete', metadata: { step_id: 's1', decision_number: 2 } },
    { stage: 'observation', status: 'complete', metadata: { observation_id: 'o2' } },
    { stage: 'termination', status: 'complete', metadata: { termination_reason: 'finish' } },
  ]);
  assert.deepEqual(rows.map(row => row.key), ['loop-decision:1', 'tool:s1:1', 'observation:o1',
    'loop-decision:2', 'tool:s1:2', 'observation:o2', 'termination']);
  assert.equal(rows[0].status, 'complete');
  assert.equal(rows[2].status, 'failed');
});
