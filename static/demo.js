const $ = (id) => document.getElementById(id);
const bundle = window.NBA_DEMO_DATA || { stats: { cases: [] }, agent: [] };
let cases = bundle.stats.cases;
let agentCases = bundle.agent;
let activeView = 'agent';
const conversations = {};
function conversationKey(view) { return view + ':' + $('routing-mode').value; }
let historyStorage;
try { historyStorage = window.localStorage; } catch { historyStorage = null; }
const sessionHistory = new SessionHistory({ storage: historyStorage });
const activeSessions = {};
let pending = null;
let requestVersion = 0;
let toastTimer;
const displayed = {};
const names = { agent: '统一 Agent', stats: '结构化统计', knowledge: '知识问答' };
const statuses = {
  complete: '完成', answered: '已回答', partial: '部分完成', clarification_required: '待补充',
  unsupported: '暂不支持', not_computable: '无法计算', incomplete: '未完成',
  abstained: '未作答', abstain: '未作答', failed: '失败', timeout: '超时',
  skipped: '未执行', unknown: '未记录', running: '执行中',
  pending: '等待',
};
const capabilities = {
  season_summary: '赛季统计', single_game_score: '单场比分', window_query: '近期比赛统计',
  season_compare: '赛季对比', player_efg: '有效命中率', knowledge_glossary: '知识问答',
  knowledge_rules: '规则检索', knowledge: '知识检索',
};
const qualityNames = { validated: '已校验', unverified: '需要进一步核对', unsupported: '不支持', invalid: '不可用' };
const limitationNames = {
  season_completeness_not_checked: '尚未确认是否覆盖该赛季全部比赛',
  capability_evidence_conditional: '该统计范围的独立核验依据不完整',
  historical_shooting_unverified: '投篮数据尚未完成独立核验',
  capability_not_profiled: '该赛季和阶段尚未建立完整的数据质量评估',
  fewer_than_requested_games: '实际场次少于请求数量',
  partial_result: '结果只覆盖已取得的记录',
  inferred_appearance_included: '结果包含根据数据推断的出场记录',
  game_count_mismatch: '数据库比赛数与参照赛程不一致',
  team_context_incomplete: '部分球队关联信息尚未完成核验',
  zero_baseline: '基线为 0，无法计算百分比变化',
};
function plainLimitations(limitations = [], capability = '') {
  const values = [...new Set(Array.isArray(limitations) ? limitations : [])];
  if (capability === 'single_game_score') {
    const seasonIndex = values.indexOf('season_completeness_not_checked');
    if (seasonIndex >= 0) values.splice(seasonIndex, 1);
  }
  if (values.includes('season_completeness_not_checked')) {
    const labels = { ...limitationNames,
      season_completeness_not_checked: capability === 'window_query'
        ? '尚未确认当前查询范围是否覆盖全部比赛'
        : capability === 'season_compare'
          ? '尚未确认两个赛季是否都覆盖全部比赛'
          : capability === 'season_summary'
            ? '尚未确认是否覆盖该赛季全部比赛'
            : '尚未确认该查询范围是否完整',
    };
    return values.filter((item) => item !== 'capability_evidence_conditional')
      .map((item) => labels[item] || '存在未展开的数据限制');
  }
  return values.map((item) => limitationNames[item] || '存在未展开的数据限制');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}
function icon(name) { return '<i data-lucide="' + name + '" aria-hidden="true"></i>'; }
function paintIcons() { window.lucide?.createIcons(); }
function statusName(value) { return statuses[value] || value || '未知'; }
function capabilityName(value) { return capabilities[value] || '工具执行'; }
function formatError(value) {
  return typeof value === 'string' ? value : JSON.stringify(value ?? '请求失败', null, 2);
}
function failureDiagnostic(response = {}) {
  const candidates = [];
  if (response.reason_code) candidates.push({ code: response.reason_code, stage: response.error_stage, timing: response.timings });
  for (const [index, step] of (response.steps || []).entries()) {
    const result = step.result || {};
    if (result.reason_code || step.error) {
      const stepTiming = response.timings?.[step.id + '_ms'] ?? response.timings?.['s' + (index + 1) + '_ms'];
      candidates.push({ code: result.reason_code || step.error, stage: result.error_stage || step.evidence?.error_stage,
        timing: { ...result, ...result.timings, elapsed_ms: stepTiming ?? result.elapsed_ms } });
    }
  }
  const item = candidates.find((candidate) => candidate.code !== 'no_usable_result' &&
    /(?:timeout|failed|error|unavailable|invalid)/i.test(candidate.code || ''))
    || candidates[0];
  if (!item) return null;
  const stages = {
    retrieval_timeout: '知识检索', planning_timeout: '查询规划', embedding_timeout: '向量生成',
    generation_timeout: '答案生成', verification_timeout: '引用验证', total_timeout: '知识回答总链路',
    agent_timeout: 'Agent 总链路', provider_failed: '模型服务', no_usable_result: '结果整理',
  };
  const timing = item.timing || {};
  const elapsed = timing[item.stage + '_ms'] ?? timing[item.code.replace(/_timeout$|_error$/, '') + '_ms']
    ?? timing.elapsed_ms ?? timing.total_ms;
  const stageLabels = { planning: '查询规划', planner: '知识规划', validation: '参数校验', execution: '执行', evidence: '证据收集', rendering: '结果整理',
    generation: '答案生成', verification: '语义验证', retrieval: '知识检索', embedding: '向量生成', total: '知识回答总链路' };
  return { code: String(item.code), stage: stageLabels[item.stage] || stages[item.code] || '执行', elapsed };
}
function responseSteps(response = {}) {
  return response.steps || (response.tool ? [{ capability: response.tool, status: response.status }] : []);
}
function executionChainHtml(response = {}) {
  if (Array.isArray(response.execution_events) && response.execution_events.length) {
    const rows = window.executionEventRows(response.execution_events);
    return rows.map((event, index) => {
      const reason = event.metadata?.reason_code || event.metadata?.error;
      const label = event.stage === 'tool' && event.metadata?.capability
        ? capabilityName(event.metadata.capability) : (event.message || event.stage);
      const depthClass = event.parent_key ? ' chain-child' : '';
      return '<div class="chain-item' + depthClass + '"><span class="chain-index">' + (index + 1) +
      '</span><div class="chain-copy"><strong>' + escapeHtml(label) + '</strong>' +
      (reason ? '<small>' + escapeHtml(reason) + '</small>' : '') + '</div><span class="chain-status">' +
      escapeHtml(statusName(event.status)) + (event.elapsed_ms != null ? ' · ' + escapeHtml(String(event.elapsed_ms)) + ' ms' : '') +
      '</span></div>';
    }).join('') || '<p class="quiet-empty">没有工具执行记录</p>';
  }
  const steps = responseSteps(response);
  const items = [];
  const planner = response.planning?.planner_type;
  if (planner) items.push({ label: planner === 'model' ? '模型规划' : '规则规划', status: 'complete', meta: response.planning?.elapsed_ms });
  for (const [index, step] of steps.entries()) {
    const elapsed = response.timings?.[step.id + '_ms'] ?? response.timings?.['s' + (index + 1) + '_ms'];
    if (step.capability !== 'knowledge_glossary') {
      items.push({ label: capabilityName(step.capability), status: step.status, meta: elapsed,
        detail: step.result?.reason || step.result?.reason_code || step.result?.answer });
    } else if (!step.result?.execution_events && !step.evidence?.execution_events) {
      items.push({ label: capabilityName(step.capability), status: step.status, meta: elapsed,
        detail: step.result?.reason || step.result?.reason_code || step.result?.answer });
      items.push(...knowledgeStages(step.result || step.evidence || {}));
    }
  }
  if (!steps.length && (response.stage_statuses || response.generation_ms != null)) items.push(...knowledgeStages(response));
  if (!items.length) return '<p class="quiet-empty">没有工具执行记录</p>';
  return items.map((item, index) => '<div class="chain-item"><span class="chain-index">' + (index + 1) +
    '</span><div class="chain-copy"><strong>' + escapeHtml(item.label) + '</strong>' +
    (item.detail ? '<small>' + escapeHtml(item.detail) + '</small>' : '') + '</div><span class="chain-status">' +
    escapeHtml(statusName(item.status)) + (item.meta != null ? ' · ' + escapeHtml(String(item.meta)) + ' ms' : '') + '</span></div>').join('');
}
function estimatedKnowledgeHtml(active = 'retrieval') {
  return window.estimatedKnowledgeStages(active).map((item, index) =>
    '<div class="chain-item"><span class="chain-index">' + (index + 1) +
    '</span><div class="chain-copy"><strong>' + escapeHtml(item.label) + '</strong>' +
    '<small>客户端估计进度</small></div><span class="chain-status">' +
    escapeHtml(statusName(item.status)) + '</span></div>').join('');
}
function turnAnswer(response) {
  const diagnostic = failureDiagnostic(response);
  return response.answer || response.reason || (diagnostic ? diagnostic.stage + '阶段未完成（' + diagnostic.code + '）。' : '未得到可展示答案');
}
function formatTurnTime(value) {
  const date = new Date(value || Date.now());
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit' }).format(date);
}
function renderTranscript(key) {
  const session = sessionHistory.get(key);
  const turns = session?.turns || [];
  const target = $('agent-transcript');
  if (!target) return;
  target.hidden = !turns.length;
  target.innerHTML = turns.map((turn) => {
    const response = turn.payload.response || turn.payload;
    return '<div class="transcript-turn' + (turn.id === session.selectedTurnId ? ' selected-turn' : '') + '"><div class="transcript-bubble transcript-user">' +
      escapeHtml(turn.question) + '<div class="transcript-meta">用户 · ' + escapeHtml(formatTurnTime(turn.createdAt)) + ' · 第 ' + escapeHtml(String(turn.turnIndex)) + ' 轮</div></div>' +
      '<div class="transcript-bubble transcript-assistant">' + escapeHtml(turnAnswer(response)) +
      '<div class="transcript-meta">Agent · ' + escapeHtml(statusName(response.status)) +
      (response.execution_mode === 'bounded_loop' ? ' · 决策循环' : '') +
      (response.termination_reason ? ' · ' + escapeHtml(response.termination_reason) : '') +
      ' · ' + escapeHtml(formatTurnTime(turn.createdAt)) +
      (response.timings?.total_ms != null ? ' · ' + escapeHtml(String(response.timings.total_ms)) + ' ms' : '') +
      '<button class="icon-button turn-select" data-turn="' + escapeHtml(turn.id) + '" title="查看本轮结果" aria-label="查看第 ' +
      escapeHtml(String(turn.turnIndex)) + ' 轮结果">' + icon('list-tree') + '</button></div></div></div>';
  }).join('');
  target.querySelectorAll('[data-turn]').forEach(button => button.addEventListener('click', () => selectHistoryTurn(key, button.dataset.turn)));
  paintIcons();
}
function currentTranscriptKey() { return activeSessions[conversationKey('agent')] || null; }
function sessionHint(response = {}, payload = {}) {
  const target = $('agent-session-hint');
  if (!target) return;
  const id = response.conversation_id || response.conversationId || payload.conversation_id;
  if (!id || payload.mode) { target.hidden = true; return; }
  target.hidden = false;
  target.textContent = response.conversation_context_status === 'reset'
    ? '上一会话上下文已失效，本次开始新会话。历史记录仍可回看。'
    : '历史记录保存在此浏览器。服务端上下文约 30 分钟或服务重启后失效，历史回看不保证上下文仍有效。';
}
function sourceLabel(type) {
  if (type === 'live_provider') return '历史在线执行回放';
  if (type?.includes('contract')) return '合成数据回放';
  if (type === 'fixture') return '快照数据回放';
  return '本次实时执行';
}
function notify(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3000);
}
function cancelPending() {
  requestVersion += 1;
  pending?.abort();
  pending = null;
  document.querySelectorAll('.primary').forEach((button) => { button.disabled = false; });
  document.querySelectorAll('.result[aria-busy="true"]').forEach((target) => {
    target.removeAttribute('aria-busy');
    target.hidden = true;
  });
}
async function request(url, options = {}, controller = new AbortController()) {
  const timeout = setTimeout(() => controller.abort(), 65000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    let body;
    try { body = await response.json(); }
    catch { throw new Error('服务返回了无法解析的响应。'); }
    if (!response.ok) {
      const error = new Error(response.status === 503 ? '相关服务尚未就绪，请检查资源配置。'
        : response.status === 409 ? '上一轮分析仍在执行，请稍后再试。'
        : '请求未完成（HTTP ' + response.status + '）。');
      error.details = body;
      error.httpStatus = response.status;
      throw error;
    }
    return body;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('请求等待超时，请稍后重试。');
    if (error instanceof TypeError) throw new Error('无法连接本地服务。离线回放仍可用。');
    throw error;
  } finally { clearTimeout(timeout); }
}

async function requestStream(url, options, controller, onEvent) {
  const response = await fetch(url, { ...options, signal: controller.signal });
  if (!response.ok || !response.body) {
    const error = new Error('实时进度不可用');
    try { error.details = await response.json(); } catch {}
    throw error;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;
  const consume = (block) => {
    let event = 'message';
    let data = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    if (!data) return;
    const payload = JSON.parse(data);
    if (event === 'stage') onEvent(payload);
    if (event === 'result') result = payload;
  };
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() || '';
    blocks.forEach(consume);
    if (done) break;
  }
  return result;
}

function populateCases() {
  const oldAgent = $('agent-case-select').value;
  const oldStats = $('case-select').value;
  $('case-select').replaceChildren(...cases.map((item) => new Option(item.label || item.question, item.case_id)));
  $('agent-case-select').replaceChildren(...agentCases.map((item) =>
    new Option(item.question + ' · ' + statusName(item.status) + ' · ' + sourceLabel(item.source_type), item.case_id)));
  if (cases.some((item) => item.case_id === oldStats)) $('case-select').value = oldStats;
  if (agentCases.some((item) => item.case_id === oldAgent)) $('agent-case-select').value = oldAgent;
  if (!$('agent-question').value) $('agent-question').value = agentCases[0]?.question || '';
  if (!$('stats-question').value) $('stats-question').value = cases[0]?.question || '';
  updateModes();
  renderSuggestions();
}
function updateModes() {
  const agentReplay = $('agent-mode').value === 'replay';
  const statsReplay = $('stats-mode').value === 'fixture';
  $('agent-question').readOnly = agentReplay;
  $('stats-question').readOnly = statsReplay;
  $('agent-case-select').closest('.example-picker').hidden = !agentReplay;
  $('case-select').closest('.example-picker').hidden = !statsReplay;
  const selected = agentCases.find((item) => item.case_id === $('agent-case-select').value);
  $('agent-hint').textContent = agentReplay ? sourceLabel(selected?.source_type)
    : conversations[conversationKey('agent')] ? '当前会话 · 可继续追问' : '本次实时执行';
  $('stats-hint').textContent = statsReplay ? '快照数据回放' : conversations[conversationKey('stats')] ? '当前会话 · 可继续追问' : '本次实时执行';
  $('routing-mode').disabled = activeView === 'knowledge' || (activeView === 'agent' ? agentReplay : statsReplay);
  if (activeView === 'agent') {
    const id = agentReplay ? null : currentTranscriptKey();
    renderTranscript(id);
    const turn = sessionHistory.selectedTurn(id);
    sessionHint(turn?.payload || { conversation_id: agentReplay ? null : conversations[conversationKey('agent')] }, {});
  }
  document.querySelectorAll('[data-mode]').forEach((button) => {
    const selectedMode = $(button.dataset.control).value === button.dataset.mode;
    button.setAttribute('aria-pressed', String(selectedMode));
  });
}
function chooseCase(view) {
  const selected = view === 'agent'
    ? agentCases.find((candidate) => candidate.case_id === $('agent-case-select').value)
    : cases.find((candidate) => candidate.case_id === $('case-select').value);
  if (selected) $(view + '-question').value = selected.question;
  updateModes();
}
function renderSuggestions() {
  const options = activeView === 'stats'
    ? cases.slice(0, 4).map((item) => ({ ...item, note: '快照数据回放', image: null }))
    : activeView === 'knowledge'
      ? [{ question: '什么是有效命中率 eFG%？', note: '统计指标', symbol: 'percent' },
        { question: 'NBA 三秒违例是如何定义的？', note: '比赛规则', symbol: 'book-open' }]
      : [agentCases[0], agentCases.find((item) => item.source_type === 'synthetic_mixed_contract'),
        agentCases.find((item) => item.status === 'partial'), agentCases.find((item) => item.status === 'clarification_required')]
        .filter(Boolean).map((item) => ({ ...item, note: sourceLabel(item.source_type) + ' · ' + statusName(item.status),
          image: null }));
  $('suggestion-list').innerHTML = options.map((item, index) =>
    '<button class="suggestion" data-suggestion="' + index + '">' +
    (item.image ? '<img src="./' + item.image + '" alt="" width="53" height="54">'
      : '<span class="suggestion-icon">' + icon(item.symbol || 'chart-no-axes-combined') + '</span>') +
    '<span class="suggestion-copy"><h3>' + escapeHtml(item.question) + '</h3><p>' +
    escapeHtml(item.note) + '</p></span>' + icon('arrow-up-right') + '</button>').join('');
  $('suggestion-list').querySelectorAll('button').forEach((button, index) => {
    button.addEventListener('click', () => {
      cancelPending();
      const item = options[index];
      if (activeView === 'knowledge') $('knowledge-question').value = item.question;
      else {
        $(activeView + '-mode').value = activeView === 'agent' ? 'replay' : 'fixture';
        $(activeView === 'agent' ? 'agent-case-select' : 'case-select').value = item.case_id;
        chooseCase(activeView);
      }
      $(activeView + '-question').focus();
    });
  });
  paintIcons();
}

async function checkConnection() {
  if (location.protocol === 'file:') {
    showConnection(false, '离线回放');
    $('resource-status').innerHTML = '<li><span>演示回放</span><span class="resource-label good">可用</span></li>' +
      '<li><span>API 服务</span><span class="resource-label warn">未连接</span></li>';
    return;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  $('refresh-connection').disabled = true;
  try {
    const data = await request('/demo/status', {}, controller);
    showConnection(true, '服务已连接');
    const r = data.resources || {};
    const rows = [
      ['演示回放', true, '可用', ''],
      ['结构化数据', data.modes?.structured, '就绪', '缺少数据'],
      ['知识索引', r.knowledge_index?.available && r.knowledge_index?.dense_available, '已检测到', '未就绪'],
      ['生成模型', r.model?.configured, '已配置', '未配置'],
      ['Embedding', r.embedding?.configured, '已配置', '未配置'],
    ];
    $('resource-status').innerHTML = rows.map(([name, good, yes, no]) =>
      '<li><span>' + name + '</span><span class="resource-label ' + (good ? 'good' : 'warn') + '">' +
      (good ? yes : no) + '</span></li>').join('');
  } catch {
    showConnection(false, '服务未连接');
    $('resource-status').innerHTML = '<li><span>演示回放</span><span class="resource-label good">可用</span></li>' +
      '<li><span>API 服务</span><span class="resource-label warn">未连接</span></li>';
  } finally {
    clearTimeout(timer);
    $('refresh-connection').disabled = false;
  }
}
function showConnection(ok, label) {
  $('connection-status').className = 'connection-pill ' + (ok ? 'connected' : 'offline');
  $('connection-status').innerHTML = '<span class="tiny-dot"></span><span>' + label + '</span>';
  $('sidebar-connection').textContent = label;
}

// Preserve each evidence record, including conditional quality and unresolved citations.
function evidenceRecords(response) {
  const evidence = Array.isArray(response.evidence) ? response.evidence
    : response.evidence ? [{ structured: response.evidence }] : [];
  const records = [];
  for (const item of evidence) {
    if (item.structured) records.push({ kind: 'structured', value: item.structured, capability: item.capability,
      executionStatus: item.execution_status || response.tool_result?.status || item.structured.tool_status });
    for (const citation of item.knowledge?.citations || []) records.push({ kind: 'knowledge', value: citation });
  }
  if (!records.some((item) => item.kind === 'knowledge')) {
    for (const claim of response.claims || []) records.push({ kind: 'knowledge', value: claim });
  }
  return records;
}
function evidenceHtml(record) {
  const e = record.value;
  if (record.kind === 'structured') {
    const samples = e.sample_count ?? e.sample_size;
    const plain = plainLimitations(e.limitations, record.capability || e.capability);
    const quality = plain.length ? '' : (qualityNames[e.quality_status] || e.quality_status || '');
    return '<div class="evidence-item"><strong>' + escapeHtml(e.entity?.name || '结构化数据') +
      '</strong>' + escapeHtml(e.snapshot_id || e.source_snapshot || '固定数据快照') +
      (samples != null ? ' · 样本 ' + escapeHtml(samples) : '') +
      '<small>' + escapeHtml(['failed', 'incomplete', 'unsupported', 'not_computable'].includes(record.executionStatus)
        ? '未生成可验证结果' : quality || '质量未标注') + '</small>' +
      (plain.length ? '<small>说明：' + escapeHtml(plain.join('；')) + '</small>' : '') +
      (e.formula ? '<blockquote>' + escapeHtml(e.formula) + '</blockquote>' : '') + '</div>';
  }
  const url = e.url || e.source_url;
  const safeUrl = typeof url === 'string' && /^https?:\/\//i.test(url) ? url : null;
  return '<div class="evidence-item"><strong>' + escapeHtml(e.source_title || e.source_id || '知识引用') + '</strong>' +
    (e.text ? escapeHtml(e.text) : '') +
    (e.excerpt ? '<blockquote>' + escapeHtml(e.excerpt) + '</blockquote>' : '') +
    '<small>' + escapeHtml([e.page != null ? '页码 ' + e.page : '', e.anchor,
      e.validation === 'validated' ? '引用已校验' : e.validation, e.source_version].filter(Boolean).join(' · ')) + '</small>' +
    (safeUrl ? '<a href="' + escapeHtml(safeUrl) + '" target="_blank" rel="noopener noreferrer">查看来源 ↗</a>' : '') + '</div>';
}
function updateContext(response = {}, payload = {}, question = '') {
  const records = evidenceRecords(response);
  const structured = records.find((item) => item.kind === 'structured')?.value || {};
  const args = response.arguments || response.task_state?.route?.steps?.[0]?.arguments || {};
  const season = structured.season_id || args.season_id;
  const phase = structured.competition_phase || args.competition_phase;
  const phaseLabel = { regular_season: '常规赛', playoffs: '季后赛', preseason: '季前赛' }[phase] || '';
  $('scope-source').textContent = payload.mode === 'fixture' ? sourceLabel('fixture')
    : payload.mode === 'replay' ? sourceLabel(payload.provenance?.source_type) : question ? '本次实时执行' : '待分析';
  $('scope-season').textContent = [season, phaseLabel].filter(Boolean).join(' · ') || '未限定';
  $('scope-entity').textContent = structured.entity?.name || '未限定';
  $('scope-steps').textContent = response.steps ? String(response.steps.length)
    : response.tool_calls_used != null ? String(response.tool_calls_used) : '—';
  $('evidence-count').textContent = String(records.length);
  $('evidence-preview').innerHTML = records.length ? records.slice(0, 3).map(evidenceHtml).join('')
    : '<div class="empty-evidence">' + icon('files') + '<p>暂无证据</p></div>';
  paintIcons();
}
function renderResult(view, payload, question) {
  const response = payload.response || payload;
  const target = $(view + '-result');
  displayed[view] = { payload, question };
  target.hidden = false;
  target.removeAttribute('aria-busy');
  const records = evidenceRecords(response);
  const steps = responseSteps(response);
  const status = response.status || 'unknown';
  const good = ['complete', 'answered'].includes(status);
  const totalMs = response.timings?.total_ms ?? response.total_ms ?? response.elapsed_ms;
  const label = payload.mode === 'fixture' ? sourceLabel('fixture')
    : payload.mode === 'replay' ? sourceLabel(payload.provenance?.source_type) : '本次实时执行';
  const diagnostic = !good ? failureDiagnostic(response) : null;
  const answer = response.answer || response.reason || (diagnostic
    ? diagnostic.stage + '阶段未完成（' + diagnostic.code + '）。'
    : '当前问题未得到可展示答案。');
  const unresolved = [...(response.unresolved || []), ...(Array.isArray(response.evidence)
    ? response.evidence.flatMap((item) => item.knowledge?.unresolved || []) : [])];
  target.innerHTML = '<div class="result-header"><div class="result-title">' + icon('sparkles') + '分析结果' +
    '<span class="status-badge ' + (good ? '' : ['partial', 'clarification_required'].includes(status) ? 'partial' : 'error') + '">' +
    escapeHtml(statusName(status)) + '</span></div><div class="result-actions">' +
    '<button class="icon-button" data-action="copy" title="复制答案" aria-label="复制答案">' + icon('copy') + '</button>' +
    '<button class="icon-button" data-action="download" title="下载完整结果" aria-label="下载完整结果">' + icon('download') + '</button></div></div>' +
    '<div class="answer-question">' + escapeHtml(question) + '</div><div class="answer">' + escapeHtml(answer) + '</div>' +
    (diagnostic ? '<div class="unresolved">失败阶段：' + escapeHtml(diagnostic.stage) + ' · 错误码：' +
      escapeHtml(diagnostic.code) + (diagnostic.elapsed != null ? ' · 已耗时：' + escapeHtml(String(diagnostic.elapsed)) + ' ms' : '') + '</div>' : '') +
    '<div class="meta"><span>' + icon('database') + escapeHtml(label) + '</span>' +
    (response.execution_mode === 'bounded_loop' ? '<span>决策循环 · ' + escapeHtml(String(response.decisions || 0)) + ' 次决策 · ' + escapeHtml(String(response.tool_calls || 0)) + ' 次工具调用</span>' : '') +
    (response.termination_reason ? '<span>终止：' + escapeHtml(response.termination_reason) + '</span>' : '') +
    (totalMs != null ? '<span>' + icon('clock-3') + escapeHtml(String(totalMs)) + ' ms</span>' : '') +
    (response.turn_index != null && !payload.mode ? '<span>第 ' + escapeHtml(response.turn_index) + ' 轮</span>' : '') + '</div>' +
    (payload.mode === 'replay' && payload.provenance?.source_type !== 'live_provider'
      ? '<div class="provenance">测试用合成数据与合同回放，数值不代表完整 NBA 赛季。</div>' : '') +
    (unresolved.length ? '<div class="unresolved">未完成：' + escapeHtml(unresolved.map(formatError).join('；')) + '</div>' : '') +
    '<div class="result-tabs" role="tablist" aria-label="结果详情">' +
    '<button id="' + view + '-execution-tab" class="result-tab active" role="tab" aria-selected="true" aria-controls="' + view + '-execution" data-detail="execution">执行记录 · ' + steps.length + '</button>' +
    '<button id="' + view + '-evidence-tab" class="result-tab" role="tab" aria-selected="false" aria-controls="' + view + '-evidence" data-detail="evidence">证据来源 · ' + records.length + '</button></div>' +
    '<div id="' + view + '-execution" class="detail-panel" role="tabpanel" aria-labelledby="' + view + '-execution-tab">' +
    (view === 'agent' || view === 'knowledge' ? '<div class="execution-chain">' + executionChainHtml(response) + '</div>' :
      (steps.length ? steps.map((step, index) => '<div class="agent-step"><span class="step-index">' + (index + 1) +
        '</span><div class="step-copy">' + escapeHtml(capabilityName(step.capability)) +
        '<small>' + escapeHtml(step.result?.reason || step.result?.answer || '') + '</small></div><span class="step-status">' +
        escapeHtml(statusName(step.status)) + '</span></div>').join('') : '<p class="quiet-empty">没有工具执行记录</p>')) + '</div>' +
    '<div id="' + view + '-evidence" class="detail-panel" role="tabpanel" aria-labelledby="' + view + '-evidence-tab" hidden>' +
    (records.length ? records.map(evidenceHtml).join('') : '<p class="quiet-empty">没有可用证据</p>') + '</div>' +
    '<details class="diagnostics"><summary>原始响应与来源记录</summary><pre>' + escapeHtml(JSON.stringify(payload, null, 2)) + '</pre></details>';
  bindTabs(target.querySelectorAll('.result-tab'), (button) => {
    target.querySelectorAll('.result-tab').forEach((item) => {
      item.classList.toggle('active', item === button);
      item.setAttribute('aria-selected', String(item === button));
    });
    $(view + '-execution').hidden = button.dataset.detail !== 'execution';
    $(view + '-evidence').hidden = button.dataset.detail !== 'evidence';
  });
  target.querySelector('[data-action="copy"]').addEventListener('click', async () => {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(answer);
      else {
        const field = document.createElement('textarea');
        field.value = answer;
        field.style.position = 'fixed';
        field.style.left = '-9999px';
        document.body.append(field);
        field.select();
        const copied = document.execCommand('copy');
        field.remove();
        if (!copied) throw new Error('copy failed');
      }
      notify('答案已复制');
    } catch { notify('无法访问剪贴板，请选中答案复制'); }
  });
  target.querySelector('[data-action="download"]').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'nba-analysis-' + view + '.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  updateContext(response, payload, question);
  if (view === 'agent') sessionHint(response, payload);
  paintIcons();
}
function renderError(view, error) {
  const target = $(view + '-result');
  target.hidden = false;
  target.removeAttribute('aria-busy');
  target.innerHTML = '<div class="error-message"><strong>分析未完成</strong>' + escapeHtml(error.message) +
    '<div class="notice-actions"><button data-retry>重试</button></div></div>' +
    (error.details ? '<details class="diagnostics"><summary>查看错误详情</summary><pre>' +
      escapeHtml(JSON.stringify(error.details, null, 2)) + '</pre></details>' : '');
  target.querySelector('[data-retry]').addEventListener('click', () => submit(view));
  delete displayed[view];
  updateContext();
}

function recordHistory(view, payload, question) {
  const response = payload.response || payload;
  const conversationId = view !== 'knowledge' && !payload.mode ? (response.conversation_id || conversations[conversationKey(view)]) : null;
  const live = view !== 'knowledge' && !payload.mode;
  const key = conversationKey(view);
  const session = sessionHistory.record({ view, payload, question, mode: view === 'knowledge' ? 'live' : $(view + '-mode').value,
    conversationId, turnIndex: response.turn_index, routingMode: $('routing-mode').value }, live ? activeSessions[key] : null);
  if (live) activeSessions[key] = session.id;
  for (const [activeKey, value] of Object.entries(activeSessions)) if (!sessionHistory.get(value)) {
    delete activeSessions[activeKey]; delete conversations[activeKey];
  }
  if (view === 'agent') renderTranscript(live ? session.id : null);
  if (sessionHistory.persistenceError) notify('浏览器存储不可用，记录仅保留在当前页面。');
  renderHistory();
}
function removeHistory(sessionId) {
  sessionHistory.remove(sessionId);
  for (const [key, value] of Object.entries(activeSessions)) if (value === sessionId) {
    delete activeSessions[key]; delete conversations[key];
    const view = key.split(':')[0];
    if (key === conversationKey(view)) {
      $(view + '-result').hidden = true; delete displayed[view];
      if (view === 'agent') { renderTranscript(null); $('agent-session-hint').hidden = true; }
    }
  }
  updateContext();
  renderHistory();
}
function selectHistoryTurn(sessionId, turnId) {
  cancelPending(); const session = sessionHistory.get(sessionId);
  if (!session) return;
  const item = sessionHistory.select(sessionId, turnId || session.selectedTurnId);
  if (!item) return;
  $('routing-mode').value = item.routingMode || 'model';
  if (item.view !== 'knowledge') {
    const key = conversationKey(item.view);
    activeSessions[key] = session.id; conversations[key] = session.conversationId;
    $(item.view + '-mode').value = item.mode;
    if (item.payload.case_id) $(item.view === 'agent' ? 'agent-case-select' : 'case-select').value = item.payload.case_id;
  }
  $(item.view + '-question').value = item.question;
  switchView(item.view); renderResult(item.view, item.payload, item.question); updateModes(); renderHistory();
}
function renderHistory() {
  $('history-count').textContent = String(sessionHistory.sessions.length);
  $('history-list').innerHTML = sessionHistory.sessions.map((item) => '<div class="history-row"><button class="history-item" data-history="' + escapeHtml(item.id) +
    '" title="' + escapeHtml(item.title) + '">' + icon('message-circle') + '<span>' + escapeHtml(item.title) +
    '<small>' + escapeHtml(names[item.view]) + ' · ' + item.turns.length + ' 轮</small></span>' +
    '</button><button class="icon-button history-remove" data-remove-history="' + escapeHtml(item.id) + '" title="删除会话" aria-label="删除会话">' + icon('x') + '</button></div>').join('') || '<p class="quiet-empty">暂无分析记录</p>';
  $('history-list').querySelectorAll('[data-remove-history]').forEach((button) => button.addEventListener('click', (event) => {
    event.stopPropagation(); removeHistory(button.dataset.removeHistory);
  }));
  $('history-list').querySelectorAll('[data-history]').forEach((button) => button.addEventListener('click', () => selectHistoryTurn(button.dataset.history)));
  paintIcons();
}
async function submit(view) {
  if (pending) return;
  const question = $(view + '-question').value.trim();
  if (!question) { $(view + '-question').focus(); notify('请输入问题'); return; }
  const mode = view === 'knowledge' ? 'live' : $(view + '-mode').value;
  const version = ++requestVersion;
  pending = new AbortController();
  const controller = pending;
  document.querySelectorAll('.primary').forEach((button) => { button.disabled = true; });
  const target = $(view + '-result');
  target.hidden = false;
  target.setAttribute('aria-busy', 'true');
  target.innerHTML = '<div class="loading" role="status">' + icon('loader-circle') + '正在分析…</div>';
  delete displayed[view];
  updateContext();
  paintIcons();
  try {
    let payload;
    if (mode === 'replay' || mode === 'fixture') {
      const item = view === 'agent'
        ? agentCases.find((candidate) => candidate.case_id === $('agent-case-select').value)
        : cases.find((candidate) => candidate.case_id === $('case-select').value);
      if (!item || item.question !== question) throw new Error('请选择一个回放示例。');
      const local = (view === 'agent' ? bundle.agent : bundle.stats.cases).find((candidate) => candidate.case_id === item.case_id);
      if (local) payload = view === 'agent' ? { mode: 'replay', case_id: item.case_id, provenance: local.provenance, response: local.response }
        : { mode: 'fixture', case_id: item.case_id, provenance: bundle.stats.provenance, response: local.response };
      else payload = await request(view === 'agent' ? '/demo/agent/replay' : '/demo/fixture',
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case_id: item.case_id }) }, controller);
    } else {
      if (location.protocol === 'file:') throw new Error('实时分析需要打开本地服务页面：http://localhost:8000/demo');
      const data = { question };
      if (view !== 'knowledge') {
        data.routing_mode = $('routing-mode').value;
        const currentConversation = conversations[conversationKey(view)];
        if (currentConversation) data.conversation_id = currentConversation;
      }
      const endpoint = { agent: '/agent/answer', stats: '/answer', knowledge: '/knowledge/answer' }[view];
      if (view === 'agent' || view === 'knowledge') {
        const events = [];
        const streamEndpoint = view === 'agent' ? '/agent/answer/stream' : '/knowledge/answer/stream';
        try {
          payload = await requestStream(streamEndpoint,
            { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) },
            controller,
            (event) => {
              events.push(event);
              if (version === requestVersion) {
                const running = [...events].reverse().find((item) => item.status === 'running');
                const progress = running?.message ? '正在执行 ' + running.message : '正在分析…';
                target.innerHTML = '<div class="loading" role="status" aria-live="polite">' + icon('loader-circle') +
                  escapeHtml(progress) + '</div>';
                paintIcons();
              }
            });
        } catch {
          payload = await request(endpoint,
            { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }, controller);
        }
      } else {
        payload = await request(endpoint,
          { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }, controller);
      }
      if (view !== 'knowledge' && version === requestVersion) {
        conversations[conversationKey(view)] = payload.conversation_id || null;
      }
    }
    if (version !== requestVersion) return;
    renderResult(view, payload, question);
    recordHistory(view, payload, question);
    updateModes();
  } catch (error) {
    if (version === requestVersion) {
      const detail = error.details?.detail;
      if (view === 'knowledge' && detail?.code && detail?.timings) {
        const failed = { status: 'failed', reason_code: detail.code, error_stage: detail.stage,
          timings: detail.timings, stage_statuses: detail.stage_statuses, model_calls: detail.model_calls };
        renderResult(view, failed, question); recordHistory(view, failed, question);
      } else renderError(view, error);
    }
  } finally {
    if (version === requestVersion) {
      pending = null;
      document.querySelectorAll('.primary').forEach((button) => { button.disabled = false; });
    }
  }
}
function switchView(view) {
  activeView = view;
  document.querySelectorAll('.tab').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === view);
    button.setAttribute('aria-selected', String(button.dataset.tab === view));
  });
  Object.keys(names).forEach((key) => { $(key + '-panel').hidden = key !== view; });
  $('current-view').textContent = names[view];
  $('page-caption').textContent = { agent: '球员、球队与赛场数据', stats: '赛季统计与比赛记录', knowledge: '比赛规则与统计术语' }[view];
  const item = displayed[view];
  updateContext(item?.payload.response || item?.payload || {}, item?.payload || {}, item?.question || '');
  renderSuggestions();
  updateModes();
}
function bindTabs(buttons, select) {
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => select(button));
    button.addEventListener('keydown', (event) => {
      let next;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + buttons.length - 1) % buttons.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = buttons.length - 1;
      if (next != null) { event.preventDefault(); buttons[next].focus(); select(buttons[next]); }
    });
  });
}

['agent', 'stats', 'knowledge'].forEach((view) => $(view + '-submit').addEventListener('click', () => submit(view)));
$('agent-case-select').addEventListener('change', () => { cancelPending(); chooseCase('agent'); });
$('case-select').addEventListener('change', () => { cancelPending(); chooseCase('stats'); });
['agent', 'stats'].forEach((view) => $(view + '-mode').addEventListener('change', () => {
  cancelPending();
  if ($(view + '-mode').value !== 'live') chooseCase(view);
  updateModes();
}));
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  $(button.dataset.control).value = button.dataset.mode;
  $(button.dataset.control).dispatchEvent(new Event('change'));
}));
bindTabs(document.querySelectorAll('.tab'), (button) => { cancelPending(); switchView(button.dataset.tab); });
$('new-session').addEventListener('click', () => {
  cancelPending();
  Object.keys(conversations).forEach((key) => delete conversations[key]);
  Object.keys(activeSessions).forEach((key) => delete activeSessions[key]);
  Object.keys(names).forEach((view) => {
    $(view + '-result').hidden = true;
    delete displayed[view];
  });
  if ($('agent-mode').value === 'live') $('agent-question').value = '';
  if ($('stats-mode').value === 'live') $('stats-question').value = '';
  updateModes();
  updateContext();
  renderTranscript(null);
  $('agent-session-hint').hidden = true;
  $(activeView + '-question').focus();
  notify('已新建分析');
});
$('clear-history').addEventListener('click', () => {
  cancelPending(); sessionHistory.clear();
  Object.keys(conversations).forEach(key => delete conversations[key]);
  Object.keys(activeSessions).forEach(key => delete activeSessions[key]);
  renderTranscript(null); $('agent-session-hint').hidden = true;
  Object.keys(names).forEach(view => { $(view + '-result').hidden = true; delete displayed[view]; });
  updateContext();
  renderHistory();
  notify('已清空此浏览器的历史记录');
});
$('routing-mode').addEventListener('change', () => {
  cancelPending();
  const id = activeSessions[conversationKey(activeView)];
  if (id && sessionHistory.get(id)) selectHistoryTurn(id);
  else { $(activeView + '-result').hidden = true; delete displayed[activeView]; updateContext(); }
  updateModes();
});
$('refresh-connection').addEventListener('click', checkConnection);
populateCases();
renderHistory();
paintIcons();
checkConnection();
if (location.protocol !== 'file:') {
  Promise.allSettled([request('/demo/cases'), request('/demo/agent/cases')]).then(([stats, agent]) => {
    if (stats.status === 'fulfilled') cases = stats.value.cases;
    if (agent.status === 'fulfilled') agentCases = agent.value.cases;
    populateCases();
  });
}
