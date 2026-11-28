import { useState, type ReactNode } from 'react'
import {
  AlertTriangle, Brain, CheckCircle2, ChevronDown, ChevronRight, Code2, FileCode2,
  GitCommitHorizontal, Layers3, Terminal, Wrench, type LucideIcon,
} from 'lucide-react'
import {
  compact, duration, guardrailLabels, has, num, pct, short, words,
  type BuildSummary, type FileDiff, type Point, type ScoringPolicy, type TraceToolCall, type TraceTurn,
} from './model'

export const statusStyle = (s: string) =>
  ['valid', 'built', 'delivered', 'observed', 'complete', 'pass', 'eligible', 'active'].includes(s) ? 'good'
    : ['violation', 'build_failed', 'runtime_failure', 'fail', 'upstream_transport', 'worker_transport', 'length', 'rejected', 'generation_truncated', 'launcher_failed', 'failed_or_interrupted', 'regression', 'error'].includes(s) ? 'bad'
      : ['incomplete', 'paused', 'insufficient_evidence', 'uncertain', 'pending_review', 'reserved', 'started', 'no_terminal_receipt', 'usage_uncertain', 'inconclusive', 'needs_repeats'].includes(s) ? 'warn'
        : 'muted'

export function Badge({ children, tone = 'muted' }: { children: ReactNode; tone?: string }) {
  return <span className={`badge ${tone}`}><i />{children}</span>
}

export function Section({ title, kicker, children, className = '', action }: {
  title: string; kicker?: string; children: ReactNode; className?: string; action?: ReactNode;
}) {
  return (
    <section className={`card ${className}`}>
      <div className="section-heading">
        <div>{kicker && <div className="eyebrow">{kicker}</div>}<h2>{title}</h2></div>
        {action}
      </div>
      {children}
    </section>
  )
}

export function Empty({ title, text }: { title: string; text: string }) {
  return <div className="empty"><Layers3 size={24} /><strong>{title}</strong><span>{text}</span></div>
}

export function Drop<T extends string>({ value, onChange, options, label }: {
  value: T; onChange: (v: T) => void; options: readonly { key: T; label: string; disabled?: boolean }[]; label: string;
}) {
  return (
    <label className="select">
      <span className="sr-only">{label}</span>
      <select value={value} onChange={e => onChange(e.target.value as T)}>
        {options.map(o => <option key={o.key} value={o.key} disabled={o.disabled}>{o.label}</option>)}
      </select>
      <ChevronDown size={14} />
    </label>
  )
}

export function Stat({ icon: Icon, label, value, hint, accent }: {
  icon: LucideIcon; label: string; value: ReactNode; hint: string; accent: string;
}) {
  return (
    <div className="card stat">
      <div className="stat-top"><span className={`tile ${accent}`}><Icon size={17} /></span><span>{label}</span></div>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  )
}

export function KeyValues({ entries }: { entries: [ReactNode, ReactNode][] }) {
  return <dl className="key-values">{entries.map(([key, value], i) => <div key={i}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>
}

export function Hash({ value }: { value?: string | null }) {
  return <code className="hash" title={value ?? 'Not recorded'}>{value ?? 'Not recorded'}</code>
}

export function GuardrailStrip({ point, summary, scoring }: { point?: Point; summary?: BuildSummary; scoring?: ScoringPolicy | null }) {
  const gRatios = summary?.guardrail_ratios ?? point?.guardrail_ratios
  if (!gRatios) return null
  const floors = scoring?.guardrail_floors ?? {}
  const unfinished = point?.unfinished_count ?? 0
  return (
    <div className="guardrail-strip">
      <span className="strip-label">硬护栏 (Guardrails)</span>
      {Object.entries(gRatios).map(([k, v]) => {
        const floor = floors[k]
        const ci = summary?.ci_available ? summary.guardrail_intervals?.[k] : undefined
        const failed = summary?.below_floor.includes(k) || (floor !== undefined && v < floor)
        const tone = failed ? 'bad' : 'good'
        return (
          <div key={k} className={`guardrail-pill ${tone}`} title={floor ? `Floor: ${floor.toFixed(3)}×` : undefined}>
            <span>{guardrailLabels[k] ?? k}</span>
            <strong className="mono">{num(v, 3)}×</strong>
            {ci && <small className="mono">[{ci[0].toFixed(3)}, {ci[1].toFixed(3)}]</small>}
          </div>
        )
      })}
      <div className={`guardrail-pill ${unfinished === 0 ? 'good' : 'bad'}`}>
        <span>unfinished_count</span>
        <strong className="mono">{num(unfinished, 0)}</strong>
      </div>
    </div>
  )
}

export function W5ComparisonCard({ point }: { point: Point }) {
  const cand = point.scenes.W5
  const base = point.reference_scenes.W5
  if (!cand) return null
  const rows = [
    { label: '稳态单路 P20 解码', cand: cand.dev_score, base: base?.dev_score, unit: 'tok/s', high: true },
    { label: '首 token 延迟 (TTFT)', cand: cand.ttft_ms, base: base?.ttft_ms, unit: 'ms', high: false },
    { label: '整波输出吞吐', cand: cand.output_toks_per_s, base: base?.output_toks_per_s, unit: 'tok/s', high: true },
  ]
  return (
    <div className="w5-trio">
      <div className="w5-trio-head">
        <div>
          <span className="eyebrow">W5 · 20-SESSION BURST TRIAD</span>
          <h3>20 路突发三数并列对照（防压低并发假提速）</h3>
        </div>
        <div className="inline-badges">
          {has(cand.share_ge_40_tps) && <Badge tone={cand.share_ge_40_tps >= 0.5 ? 'good' : 'warn'}>≥40 tok/s: {pct(cand.share_ge_40_tps, 0)}</Badge>}
          {has(cand.share_ge_50_tps) && <Badge tone={cand.share_ge_50_tps > 0 ? 'good' : 'muted'}>≥50 tok/s: {pct(cand.share_ge_50_tps, 0)}</Badge>}
          {has(cand.cached_tokens_min) && <Badge>cached_min: {compact(cand.cached_tokens_min)}</Badge>}
        </div>
      </div>
      <div className="w5-trio-grid">
        {rows.map(r => {
          const better = has(r.cand) && has(r.base) && (r.high ? r.cand >= r.base : r.cand <= r.base)
          const delta = has(r.cand) && has(r.base) && r.base > 0 ? (r.high ? r.cand / r.base : r.base / r.cand) : undefined
          return (
            <div key={r.label} className="w5-cell">
              <span className="w5-cell-label">{r.label}</span>
              <strong className="mono">{has(r.cand) ? `${num(r.cand, r.unit === 'ms' ? 0 : 2)} ${r.unit}` : '—'}</strong>
              <div className="w5-cell-sub mono">
                <span>E0: {has(r.base) ? `${num(r.base, r.unit === 'ms' ? 0 : 2)} ${r.unit}` : '—'}</span>
                {delta !== undefined && <span className={better ? 'delta-good' : 'delta-bad'}>{delta.toFixed(3)}×</span>}
              </div>
            </div>
          )
        })}
      </div>
      <p className="w5-note">注：<code>ttft_ms</code> 为首 token 总延迟（含服务排队与 prefill），暂未拆分独立排队耗时。</p>
    </div>
  )
}

export function UnifiedDiffView({ files }: { files: FileDiff[] }) {
  if (!files.length) return <Empty title="No source changes" text="Identical file contents compared against reference." />
  return (
    <div className="diff-stack">
      {files.map(file => (
        <details key={file.path} className="diff-file" open={files.length <= 4}>
          <summary>
            <span className="diff-path"><FileCode2 size={14} /><code className="mono">{file.path}</code></span>
            <span className="diff-counts mono">
              <Badge tone={file.status === 'added' ? 'good' : file.status === 'deleted' ? 'bad' : 'muted'}>{file.status}</Badge>
              <span className="add-cnt">+{file.added_lines}</span>
              <span className="del-cnt">-{file.removed_lines}</span>
            </span>
          </summary>
          <pre className="diff-code mono">
            {file.diff ? file.diff.split('\n').map((line, i) => {
              const cls = line.startsWith('+++') || line.startsWith('---') ? 'diff-meta'
                : line.startsWith('@@') ? 'diff-hunk'
                  : line.startsWith('+') ? 'diff-add'
                    : line.startsWith('-') ? 'diff-del' : ''
              return <div key={i} className={`diff-line ${cls}`}>{line || ' '}</div>
            }) : <div className="diff-line diff-meta">(empty diff)</div>}
          </pre>
          {file.truncated && <p className="side-note">Diff truncated at 2,000 lines for browser responsiveness.</p>}
        </details>
      ))}
    </div>
  )
}

function ToolCallItem({ tc, runId, onInspectCandidate }: {
  tc: TraceToolCall; runId: string; onInspectCandidate?: (engine: string) => void;
}) {
  const [expanded, setExpanded] = useState(false)
  const [loadedArgs, setLoadedArgs] = useState<any>(tc.args_inline)
  const [loadedResult, setLoadedResult] = useState<any>(tc.result_inline)
  const [loading, setLoading] = useState(false)

  async function toggle() {
    const next = !expanded
    setExpanded(next)
    if (next && !loading) {
      const promises: Promise<void>[] = []
      if (!loadedArgs && tc.args_blob) {
        promises.push(fetch(`/api/runs/${encodeURIComponent(runId)}/blobs/${tc.args_blob}`).then(r => r.ok ? r.json() : null).then(d => d && setLoadedArgs(d.content)))
      }
      if (!loadedResult && tc.result_blob) {
        promises.push(fetch(`/api/runs/${encodeURIComponent(runId)}/blobs/${tc.result_blob}`).then(r => r.ok ? r.json() : null).then(d => d && setLoadedResult(d.content)))
      }
      if (promises.length) {
        setLoading(true)
        try { await Promise.all(promises) } finally { setLoading(false) }
      }
    }
  }

  const label = tc.rpc_operation ? `ee_rpc · ${tc.rpc_operation}` : (tc.tool ?? 'tool')
  const Icon = tc.tool === 'bash' ? Terminal : tc.tool === 'edit' || tc.tool === 'write' ? Code2 : tc.rpc_operation === 'candidate_submit' ? GitCommitHorizontal : Wrench
  const argsObj = loadedArgs?.args ?? loadedArgs
  const resText = (() => {
    const raw = loadedResult?.result ?? loadedResult
    if (typeof raw === 'string') return raw
    if (raw?.content && Array.isArray(raw.content)) {
      return raw.content.map((c: any) => c?.text ?? '').filter(Boolean).join('\n')
    }
    return raw ? JSON.stringify(raw, null, 2) : tc.result_preview ?? ''
  })()

  return (
    <div className={`tool-card ${tc.is_error ? 'tool-error' : ''}`}>
      <button className="tool-head" onClick={toggle}>
        <span className="tool-title">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Icon size={14} />
          <strong className="mono">{label}</strong>
          <span className="tool-preview mono">{tc.args_preview || tc.path || ''}</span>
        </span>
        <span className="tool-meta mono">
          {tc.is_error && <Badge tone="bad">error</Badge>}
          {tc.op_status && <Badge tone={statusStyle(tc.op_status)}>{words(tc.op_status)}</Badge>}
          {has(tc.duration_ms) && <span>{duration(tc.duration_ms)}</span>}
        </span>
      </button>
      {tc.engine && onInspectCandidate && (
        <div className="tool-link-bar">
          <span className="mono">Engine: {short(tc.engine)}</span>
          <button className="text-button" onClick={() => onInspectCandidate(tc.engine!)}>Open candidate detail →</button>
        </div>
      )}
      {expanded && (
        <div className="tool-body">
          {loading && <div className="side-note">Loading full blob…</div>}
          {tc.tool === 'bash' && argsObj?.command && (
            <pre className="code-block bash mono">$ {String(argsObj.command)}</pre>
          )}
          {tc.tool === 'edit' && argsObj && (argsObj.oldText !== undefined || argsObj.newText !== undefined) && (
            <div className="inline-edit-diff">
              <div className="diff-meta mono">{argsObj.path ?? tc.path}</div>
              <pre className="diff-code mono">
                {String(argsObj.oldText ?? '').split('\n').map((ln, i) => <div key={'o' + i} className="diff-line diff-del">-{ln}</div>)}
                {String(argsObj.newText ?? '').split('\n').map((ln, i) => <div key={'n' + i} className="diff-line diff-add">+{ln}</div>)}
              </pre>
            </div>
          )}
          {tc.tool !== 'bash' && tc.tool !== 'edit' && argsObj && (
            <details open>
              <summary className="mini-summary">Arguments</summary>
              <pre className="code-block mono">{typeof argsObj === 'string' ? argsObj : JSON.stringify(argsObj, null, 2)}</pre>
            </details>
          )}
          {resText && (
            <details open={resText.length <= 1200}>
              <summary className="mini-summary">Result ({compact(resText.length)} chars)</summary>
              <pre className="code-block mono">{resText.length > 16000 ? resText.slice(0, 16000) + '\n… [truncated in UI]' : resText}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

export function ConversationTurnCard({ turn, runId, onInspectCandidate }: {
  turn: TraceTurn; runId: string; onInspectCandidate?: (engine: string) => void;
}) {
  const [showReasoning, setShowReasoning] = useState(false)
  const [fullResponse, setFullResponse] = useState(turn.response_inline)
  const [loadingResp, setLoadingResp] = useState(false)

  async function loadResponse() {
    if (fullResponse || !turn.response_blob || loadingResp) return
    setLoadingResp(true)
    try {
      const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/blobs/${turn.response_blob}`)
      if (r.ok) {
        const d = await r.json()
        setFullResponse(d.content)
      }
    } finally {
      setLoadingResp(false)
    }
  }

  async function toggleReasoning() {
    const next = !showReasoning
    setShowReasoning(next)
    if (next && !fullResponse && turn.response_blob) {
      await loadResponse()
    }
  }

  const m = turn.model
  const reasoning = fullResponse?.reasoning ?? ''
  const content = fullResponse?.content ?? m?.content_preview ?? ''
  const hasReasoning = Boolean(reasoning || (m?.reasoning_chars ?? 0) > 0 || (m?.reasoning_tokens ?? 0) > 0)

  return (
    <article className={`turn-card ${turn.has_error ? 'turn-error' : ''} ${turn.is_candidate_turn ? 'turn-candidate' : ''}`}>
      <header className="turn-header">
        <div className="turn-id">
          <span className="mono turn-tag">S{turn.segment}{turn.attempt != null ? ` · #${turn.attempt}` : ''}</span>
          {turn.correlation && <Badge tone="violet">{words(turn.correlation)}</Badge>}
          {turn.is_candidate_turn && <Badge tone="good">candidate_submit</Badge>}
          {m?.finish && <Badge tone={statusStyle(m.finish)}>{words(m.finish)}</Badge>}
          {turn.has_error && <Badge tone="bad">attention</Badge>}
        </div>
        <div className="turn-metrics mono">
          {has(turn.elapsed_ms) && <span>@{duration(turn.elapsed_ms)}</span>}
          {m && <>
            <span>in {compact(m.input_tokens)}</span>
            <span>out {compact(m.output_tokens)}</span>
            {has(m.cache_hit_tokens) && <span className="cache-hit">hit {compact(m.cache_hit_tokens)}</span>}
            {has(m.wall_ms) && <span>{duration(m.wall_ms)}</span>}
          </>}
        </div>
      </header>

      {hasReasoning && (
        <div className="reasoning-box">
          <button className="reasoning-toggle" onClick={toggleReasoning}>
            <Brain size={14} />
            <span>Reasoning / CoT ({has(m?.reasoning_tokens) ? `${compact(m?.reasoning_tokens)} tok` : `${compact(m?.reasoning_chars)} chars`})</span>
            {showReasoning ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          {showReasoning && (
            <pre className="reasoning-text mono">
              {loadingResp ? 'Loading reasoning blob…' : (reasoning || '(Reasoning text unavailable)')}
            </pre>
          )}
        </div>
      )}

      {content && (
        <div className="assistant-text">
          <pre>{content}</pre>
          {!fullResponse && turn.response_blob && (m?.content_chars ?? 0) > 220 && (
            <button className="text-button" onClick={loadResponse}>{loadingResp ? 'Loading…' : 'Load full assistant message'}</button>
          )}
        </div>
      )}

      {turn.tools.length > 0 && (
        <div className="tool-list">
          {turn.tools.map((tc, idx) => (
            <ToolCallItem key={tc.call_sha256 ?? idx} tc={tc} runId={runId} onInspectCandidate={onInspectCandidate} />
          ))}
        </div>
      )}

      {turn.worker_events.length > 0 && (
        <div className="worker-strip">
          {turn.worker_events.map((w, i) => (
            <span key={i} className="worker-chip mono">
              {w.kind === 'worker_settled' ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
              {words(w.operation)} · {words(w.status ?? w.kind)} {has(w.node_ms) ? `(${duration(w.node_ms)})` : ''}
            </span>
          ))}
        </div>
      )}
    </article>
  )
}
