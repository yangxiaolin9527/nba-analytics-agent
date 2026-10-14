(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis === 'object' ? globalThis : this, function () {
  const STORAGE_KEY = 'nba-analytics-demo-history-v2';
  const VERSION = 2;
  const validViews = new Set(['agent', 'stats', 'knowledge']);

  function id(prefix) {
    return prefix + '-' + (globalThis.crypto?.randomUUID?.() || Date.now() + '-' + Math.random().toString(16).slice(2));
  }
  function cleanTurn(turn) {
    if (!turn || typeof turn !== 'object' || !validViews.has(turn.view) || !turn.question || !turn.payload) return null;
    return {
      id: turn.id || id('turn'), view: turn.view, routingMode: turn.routingMode || 'model',
      mode: turn.mode || 'live', question: String(turn.question), payload: turn.payload,
      conversationId: turn.conversationId || turn.payload.conversation_id || null,
      turnIndex: turn.turnIndex || turn.payload.turn_index || 1,
      createdAt: Number.isFinite(Number(turn.createdAt)) ? Number(turn.createdAt) : Date.now(),
    };
  }
  function cleanSession(session) {
    if (!session || typeof session !== 'object' || !session.id || !Array.isArray(session.turns)) return null;
    const turns = session.turns.map(cleanTurn).filter(Boolean);
    if (!turns.length) return null;
    const first = turns[0];
    return { id: String(session.id), view: first.view, routingMode: first.routingMode,
      mode: first.mode, conversationId: session.conversationId || first.conversationId,
      title: session.title || first.question, turns, selectedTurnId: session.selectedTurnId || turns.at(-1).id,
      createdAt: session.createdAt || first.createdAt, updatedAt: session.updatedAt || turns.at(-1).createdAt };
  }

  class SessionHistory {
    constructor({ storage = globalThis.localStorage, maxSessions = 12 } = {}) {
      this.storage = storage; this.maxSessions = maxSessions; this.persistenceError = false; this.sessions = [];
      this._load();
    }
    _load() {
      try {
        const parsed = JSON.parse(this.storage?.getItem(STORAGE_KEY) || 'null');
        if (parsed?.version !== VERSION || !Array.isArray(parsed.sessions)) return;
        this.sessions = parsed.sessions.map(cleanSession).filter(Boolean).slice(0, this.maxSessions);
        this._trim();
      } catch { this.persistenceError = true; this.sessions = []; }
    }
    _save() {
      this._trim();
      try { this.storage?.setItem(STORAGE_KEY, JSON.stringify({ version: VERSION, sessions: this.sessions })); }
      catch { this.persistenceError = true; }
    }
    _trim() {
      this.sessions = this.sessions.slice(0, this.maxSessions);
      while (this.sessions.length && (this.sessions.reduce((sum, item) => sum + item.turns.length, 0) > 100
        || new TextEncoder().encode(JSON.stringify({ version: VERSION, sessions: this.sessions })).length > 5 * 1024 * 1024)) {
        this.sessions.pop();
      }
    }
    _sessionFor(turn, requestedId) {
      if (!requestedId) return null;
      const session = this.get(requestedId);
      if (!session || session.view !== turn.view || session.routingMode !== (turn.routingMode || 'model')) return null;
      if (turn.payload?.conversation_context_status === 'reset') return null;
      if (session.mode !== (turn.mode || 'live')) return null;
      if (session.conversationId && turn.conversationId && session.conversationId !== turn.conversationId) return null;
      return session;
    }
    record(turn, requestedId = null) {
      const normalized = cleanTurn(turn); if (!normalized) throw new TypeError('invalid history turn');
      let session = this._sessionFor(normalized, requestedId);
      if (!session) {
        session = { id: id('session'), view: normalized.view, routingMode: normalized.routingMode,
          mode: normalized.mode, conversationId: normalized.conversationId, title: normalized.question,
          turns: [], selectedTurnId: normalized.id, createdAt: normalized.createdAt, updatedAt: normalized.createdAt };
        this.sessions.unshift(session);
      }
      const existing = session.turns.findIndex(item => item.id === normalized.id);
      if (existing >= 0) session.turns.splice(existing, 1);
      session.turns.push(normalized); session.selectedTurnId = normalized.id; session.updatedAt = normalized.createdAt;
      session.conversationId = normalized.conversationId || session.conversationId;
      this.sessions = [session, ...this.sessions.filter(item => item.id !== session.id)].slice(0, this.maxSessions);
      this._save(); return session;
    }
    get(sessionId) { return this.sessions.find(item => item.id === sessionId) || null; }
    select(sessionId, turnId) {
      const session = this.get(sessionId); if (!session || !session.turns.some(item => item.id === turnId)) return null;
      session.selectedTurnId = turnId; this._save(); return this.selectedTurn(sessionId);
    }
    selectedTurn(sessionId) {
      const session = this.get(sessionId); return session?.turns.find(item => item.id === session.selectedTurnId) || session?.turns.at(-1) || null;
    }
    remove(sessionId) { const before = this.sessions.length; this.sessions = this.sessions.filter(item => item.id !== sessionId); if (before !== this.sessions.length) this._save(); }
    clear() { this.sessions = []; this._save(); }
    flatten() { return this.sessions.flatMap(session => session.turns.map(turn => ({ ...turn, sessionId: session.id }))); }
  }

  // The backend keeps every event for replay and diagnostics. The UI detail view
  // presents one row per logical step, using the last terminal event as its state.
  function executionEventRows(events = []) {
    const rows = new Map();
    const terminal = (status) => !['running', 'pending'].includes(status);
    const keyFor = (event) => {
      const stage = String(event.stage || '');
      const metadata = event.metadata || {};
      if (stage === 'loop_decision') return 'loop-decision:' + String(metadata.decision_number || event.seq);
      if (stage === 'observation') return 'observation:' + String(metadata.observation_id || event.seq);
      if (stage === 'tool') return 'tool:' + String(metadata.step_id || metadata.capability || event.message || 'unknown')
        + (metadata.decision_number != null ? ':' + metadata.decision_number : '');
      if (stage === 'planning') {
        const message = String(event.message || '');
        if (metadata.intent || metadata.action || /查询规划/.test(message)) return 'query-planning';
        return 'planning';
      }
      return stage;
    };
    for (const [index, raw] of (Array.isArray(events) ? events : []).entries()) {
      if (!raw || typeof raw !== 'object' || !raw.stage || raw.stage === 'result') continue;
      const event = { ...raw, metadata: { ...(raw.metadata || {}) } };
      const key = keyFor(event);
      const existing = rows.get(key);
      const isTerminal = terminal(event.status);
      if (!existing) {
        rows.set(key, { ...event, key, _index: index, _terminal: isTerminal });
        continue;
      }
      // Running events never overwrite a terminal state. A later terminal event
      // may replace it (for example complete -> failed), while elapsed time is
      // retained when a provider omits it on the terminal event.
      if (isTerminal || !existing._terminal) {
        rows.set(key, {
          ...event,
          key,
          _index: existing._index,
          _terminal: existing._terminal || isTerminal,
          elapsed_ms: event.elapsed_ms ?? existing.elapsed_ms,
        });
      } else if (event.elapsed_ms != null && existing.elapsed_ms == null) {
        existing.elapsed_ms = event.elapsed_ms;
      }
    }
    const visible = [...rows.values()];
    // Agent requests contain a top-level model planning event and a nested
    // knowledge query planning event. Keep the product-level step only.
    if (visible.some((event) => event.key === 'planning') && visible.some((event) => event.key === 'query-planning')) {
      rows.delete('query-planning');
    }
    // Evidence assembly is an internal substage of knowledge retrieval.
    rows.delete('evidence');
    const knowledgeTool = [...rows.values()].find((event) =>
      event.stage === 'tool' && event.metadata?.capability === 'knowledge_glossary');
    if (knowledgeTool) {
      for (const event of rows.values()) {
        if (['retrieval', 'generation', 'citation_validation', 'verification'].includes(event.stage)) {
          event.parent_key = knowledgeTool.key;
        }
      }
    }
    return [...rows.values()].sort((left, right) => left._index - right._index)
      .map(({ _index, _terminal, ...event }) => event);
  }

  function knowledgeStages(result = {}) {
    const timings = result.timings || result;
    const explicit = result.stage_statuses || {};
    for (const event of result.execution_events || []) {
      if (!event.stage || event.stage === 'result' || event.stage === 'tool') continue;
      explicit[event.stage] = event.status;
    }
    const failed = result.error_stage || (result.reason_code || '').replace(/_timeout$|_error$/, '');
    const status = (stage, timing) => explicit[stage] || (failed === stage ? 'failed'
      : timing > 0 ? (failed === 'total' ? 'unknown' : 'complete')
        : timing === 0 ? 'skipped' : 'unknown');
    return [
      { stage: 'retrieval', label: '知识检索', status: status('retrieval', timings.retrieval_ms), meta: timings.retrieval_ms },
      { stage: 'generation', label: '答案生成', status: status('generation', timings.generation_ms), meta: timings.generation_ms },
      { stage: 'citation_validation', label: '引用校验', status: status('citation_validation', timings.citation_validation_ms), meta: timings.citation_validation_ms },
      { stage: 'verification', label: '语义验证', status: status('verification', timings.verification_ms), meta: timings.verification_ms },
    ];
  }
  function estimatedKnowledgeStages(active = 'retrieval') {
    const order = ['retrieval', 'evidence', 'generation', 'citation_validation', 'verification'];
    const labels = { retrieval: '知识检索', evidence: '整理相关证据', generation: '答案生成',
      citation_validation: '引用校验', verification: '语义验证' };
    const index = Math.max(0, order.indexOf(active));
    return order.map((stage, position) => ({ stage, label: labels[stage],
      status: position === index ? 'running' : position < index ? 'complete' : 'pending' }));
  }
  return { STORAGE_KEY, SessionHistory, executionEventRows, knowledgeStages, estimatedKnowledgeStages };
});
