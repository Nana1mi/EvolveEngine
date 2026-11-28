import { useEffect, useState } from 'react'
import {
  Activity, AlertCircle, ArrowLeft, BookOpen, ChartNoAxesCombined, CheckCircle2,
  ChevronLeft, ChevronRight, CircuitBoard, Clock3, ExternalLink, FileCode2,
  GitBranch, Layers3, ListFilter, MessageSquareText, ShieldCheck, X,
} from 'lucide-react'
import { ContextGrowthChart } from './chart'
import {
  Badge, ConversationTurnCard, Drop, Empty, GuardrailStrip, Hash, KeyValues,
  Section, UnifiedDiffView, W5ComparisonCard, statusStyle,
} from './components'
import {
  compact, duration, has, num, pct, ratioKeysFor, short, words,
  type CandidateDetail, type ContextPoint, type EventRow, type HarnessData,
  type ResearchData, type Snapshot, type TraceResponse, type WaterfallData,
} from './model'

export function CandidateDetailView({ runId, engine, snapshot, onBack, onSelectEngine }: {
  runId: string; engine: string; snapshot: Snapshot; onBack: () => void; onSelectEngine: (e: string) => void;
}) {
  const [data, setData] = useState<CandidateDetail | null>(null)
  const [error, setError] = useState('')
  const [tab, setTab] = useState<'diff' | 'metrics' | 'quality' | 'conversation'>('diff')
  const [diffBase, setDiffBase] = useState<'parent' | 'e0'>('parent')

  useEffect(() => {
    let alive = true
    setData(null); setError('')
    fetch(`/api/runs/${encodeURIComponent(runId)}/candidates/${engine}/detail`)
      .then(r => { if (!r.ok) throw Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { if (alive) setData(d) })
      .catch(() => { if (alive) setError('Candidate detail is unavailable for this run (archive or unrecorded source).') })
    return () => { alive = false }
  }, [runId, engine])

  if (error) {
    return (
      <Section kicker="CANDIDATE DETAIL" title={`Engine ${short(engine)}`}
        action={<button className="button" onClick={onBack}><ArrowLeft size={14} />Back</button>}>
        <Empty title="Detail unavailable" text={error} />
      </Section>
    )
  }
  if (!data) {
    return (
      <Section kicker="CANDIDATE DETAIL" title={`Engine ${short(engine)}`}
        action={<button className="button" onClick={onBack}><ArrowLeft size={14} />Back</button>}>
        <Empty title="Loading candidate evidence…" text="Reading diffs, receipts, and ID-correlated trace turns." />
      </Section>
    )
  }

  const latestPoint = data.points.at(-1)
  const summary = data.build_summaries.at(-1)
  const diffBundle = diffBase === 'parent' ? data.diff_vs_parent : data.diff_vs_e0
  const rKeys = ratioKeysFor(snapshot)
  const latestQual = data.qualities.at(-1)

  return (
    <div className="candidate-detail">
      <div className="candidate-hero card">
        <div className="candidate-hero-top">
          <div>
            <div className="eyebrow">CANDIDATE INVESTIGATION · ID-CORRELATED EVIDENCE</div>
            <h2>Engine E{data.iteration ?? '?'} <code className="mono hash-inline">{short(data.engine)}</code></h2>
            <div className="inline-badges" style={{ marginTop: 8 }}>
              {summary ? (
                <Badge tone={statusStyle(summary.status)}>
                  {words(summary.status)} · {summary.pairs}/{summary.minimum_pairs} pairs
                </Badge>
              ) : <Badge>Unmeasured</Badge>}
              {summary && !summary.ci_available && <Badge tone="warn">单点/不足 6 组 · 仅展示比值</Badge>}
              {summary?.ci_available && <Badge tone="good">95% CI 已收敛计算</Badge>}
              {latestQual && <Badge tone={statusStyle(latestQual.status ?? '')}>Quality: {words(latestQual.status)}</Badge>}
              {data.parent && (
                <button className="pill-link mono" onClick={() => onSelectEngine(data.parent!)}>
                  Parent: {short(data.parent)}
                </button>
              )}
            </div>
          </div>
          <div className="candidate-hero-right">
            {has(summary?.weighted_score ?? latestPoint?.aggregate) && (
              <div className="big-ratio">
                <strong className="mono">{num(summary?.weighted_score ?? latestPoint?.aggregate, 3)}×</strong>
                <span>vs paired E0</span>
              </div>
            )}
            <button className="button" onClick={onBack}><ArrowLeft size={14} />Overview</button>
          </div>
        </div>
        <GuardrailStrip point={latestPoint} summary={summary} scoring={data.scoring} />
        <div className="detail-tabs">
          <button className={tab === 'diff' ? 'selected' : ''} onClick={() => setTab('diff')}><FileCode2 size={14} />代码 Diff ({diffBundle.files.length})</button>
          <button className={tab === 'metrics' ? 'selected' : ''} onClick={() => setTab('metrics')}><ChartNoAxesCombined size={14} />指标与护栏 ({data.points.length})</button>
          <button className={tab === 'quality' ? 'selected' : ''} onClick={() => setTab('quality')}><ShieldCheck size={14} />构建与质检 ({data.qualities.length})</button>
          <button className={tab === 'conversation' ? 'selected' : ''} onClick={() => setTab('conversation')}><MessageSquareText size={14} />产生该候选的对话 ({data.conversation.length})</button>
        </div>
      </div>

      {tab === 'diff' && (
        <Section kicker="SOURCE & RUNTIME DIFF" title="Candidate modifications"
          action={
            <div className="scene-toggle">
              <button className={diffBase === 'parent' ? 'selected' : ''} onClick={() => setDiffBase('parent')}>vs Parent ({short(data.parent)})</button>
              <button className={diffBase === 'e0' ? 'selected' : ''} onClick={() => setDiffBase('e0')}>vs E0 ({short(data.baseline)})</button>
            </div>
          }>
          {diffBundle.runtime_diff && diffBundle.runtime_diff.length > 0 && (
            <div className="runtime-diff-box">
              <h4><code>ee_runtime.json</code> 运行旋钮变化</h4>
              <div className="table-scroll">
                <table className="table">
                  <thead><tr><th>Option</th><th>Before</th><th>After</th></tr></thead>
                  <tbody>
                    {diffBundle.runtime_diff.map(r => (
                      <tr key={r.key}>
                        <td className="mono">{r.key}</td>
                        <td className="mono del-cnt">{r.before === undefined ? '(unset)' : JSON.stringify(r.before)}</td>
                        <td className="mono add-cnt">{r.after === undefined ? '(removed)' : JSON.stringify(r.after)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {diffBundle.build_args_diff.length > 0 && (
            <div className="runtime-diff-box">
              <h4>Build Arguments 变化</h4>
              <div className="table-scroll">
                <table className="table">
                  <thead><tr><th>Argument</th><th>Before</th><th>After</th></tr></thead>
                  <tbody>
                    {diffBundle.build_args_diff.map(r => (
                      <tr key={r.key}><td className="mono">{r.key}</td><td className="mono">{JSON.stringify(r.before)}</td><td className="mono">{JSON.stringify(r.after)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <UnifiedDiffView files={diffBundle.files} />
        </Section>
      )}

      {tab === 'metrics' && (
        <div className="stack-gap">
          {latestPoint && <W5ComparisonCard point={latestPoint} />}
          <Section kicker="SCORING BREAKDOWN" title={summary?.ci_available ? `Aggregated across ${summary.pairs} matched pairs (with 95% CI)` : `Paired development ratios (${data.points.length} pair${data.points.length === 1 ? '' : 's'})`}>
            {!latestPoint ? <Empty title="No timing measurements" text="This candidate has not completed a paired develop_measure." /> : (
              <div className="table-scroll">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Scene / Sub-metric</th>
                      <th>Weight</th>
                      <th>Ratio vs E0</th>
                      <th>95% CI (≥6 pairs)</th>
                      <th>Floor</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rKeys.map(k => {
                      const val = summary?.scene_ratios?.[k] ?? latestPoint.ratios[k]
                      const ci = summary?.ci_available ? summary.intervals?.[k] : undefined
                      const floor = data.scoring?.scene_floors?.[k]
                      const weight = data.scoring?.weight_percent?.[k]
                      const bad = summary?.below_floor.includes(k) || (floor !== undefined && has(val) && val < floor)
                      const warn = summary?.uncertain_scenes.includes(k)
                      return (
                        <tr key={k}>
                          <td className="mono"><strong>{k}</strong></td>
                          <td className="mono">{weight !== undefined ? `${weight}%` : '—'}</td>
                          <td className="mono ink">{has(val) ? `${num(val, 3)}×` : '—'}</td>
                          <td className="mono">{ci ? `[${ci[0].toFixed(3)}×, ${ci[1].toFixed(3)}×]` : `需满 ${summary?.minimum_pairs ?? 6} 组配对`}</td>
                          <td className="mono">{floor !== undefined ? `${floor.toFixed(3)}×` : '—'}</td>
                          <td><Badge tone={bad ? 'bad' : warn ? 'warn' : has(val) ? 'good' : 'muted'}>{bad ? 'below floor' : warn ? 'uncertain' : has(val) ? 'ok' : 'n/a'}</Badge></td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {latestPoint?.guardrails?.G64K && (
              <div className="g64k-bar mono">
                <strong>G64K 64K 单路冷启动护栏实测：</strong>
                <span>TTFT {num(latestPoint.guardrails.G64K.ttft_ms, 0)} ms (E0: {num(latestPoint.reference_guardrails?.G64K?.ttft_ms, 0)} ms)</span>
                <span>Decode {num(latestPoint.guardrails.G64K.decode_tokens_per_s, 2)} tok/s (E0: {num(latestPoint.reference_guardrails?.G64K?.decode_tokens_per_s, 2)} tok/s)</span>
                <span>unfinished: {num(latestPoint.guardrails.G64K.unfinished_count, 0)}</span>
              </div>
            )}
          </Section>
        </div>
      )}

      {tab === 'quality' && (
        <div className="two-panels">
          <Section kicker="BUILD RECEIPTS" title="Native builds">
            {!data.builds.length ? <Empty title="No builds recorded" text="Build receipts appear after develop_build settles." /> : (
              <div className="stack-list">
                {data.builds.map(b => (
                  <div key={b.id} className="receipt-item">
                    <div className="inline-badges"><Badge tone={statusStyle(b.status ?? '')}>{words(b.status)}</Badge><span className="mono">{duration(b.duration_ms)}</span></div>
                    <KeyValues entries={[['Evidence', <Hash value={b.evidence} />], ['Wall', duration(b.wall_ms)]]} />
                  </div>
                ))}
              </div>
            )}
          </Section>
          <Section kicker="QUALITY GATES" title="Development quality verification">
            {!data.qualities.length ? <Empty title="No quality runs" text="Quality verdicts appear after develop_quality settles." /> : (
              <div className="stack-list">
                {data.qualities.map(q => (
                  <div key={q.id} className="receipt-item">
                    <div className="inline-badges">
                      <Badge tone={statusStyle(q.status ?? '')}>{words(q.status)}</Badge>
                      <span className="mono">{duration(q.duration_ms)}</span>
                      <Hash value={q.evidence} />
                    </div>
                    {q.cases && (
                      <div className="case-grid">
                        {Object.entries(q.cases).map(([name, verdict]) => {
                          const opaque = q.quality_detail?.verdict_only_cases?.includes(name)
                          return (
                            <div key={name} className="case-row">
                              <div>
                                <strong className="mono">{name}</strong>
                                {opaque && <small>公开回执仅含结论 (Verdict only)</small>}
                              </div>
                              <Badge tone={statusStyle(verdict)}>{verdict}</Badge>
                            </div>
                          )
                        })}
                      </div>
                    )}
                    {q.quality_detail?.trajectory_screen && (
                      <div className="screen-metrics">
                        <h5>Trajectory Screen 实测最大值</h5>
                        <div className="kv-chips mono">
                          {Object.entries(q.quality_detail.trajectory_screen.maxima ?? {}).map(([k, v]) => {
                            const failed = q.quality_detail?.trajectory_screen?.failed?.includes(k)
                            return <span key={k} className={failed ? 'chip-bad' : 'chip-ok'}>{k}: {v.toExponential(2)}</span>
                          })}
                        </div>
                      </div>
                    )}
                    {q.quality_detail?.sampling_screen && (
                      <div className="screen-metrics">
                        <h5>Sampling Screen</h5>
                        <div className="kv-chips mono">
                          <span>status: {q.quality_detail.sampling_screen.status ?? '—'}</span>
                          {has(q.quality_detail.sampling_screen.max_tv) && <span>max_tv: {q.quality_detail.sampling_screen.max_tv.toFixed(4)}</span>}
                          {has(q.quality_detail.sampling_screen.threshold) && <span>threshold: {q.quality_detail.sampling_screen.threshold.toFixed(4)}</span>}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Section>
        </div>
      )}

      {tab === 'conversation' && (
        <Section kicker="ID-CORRELATED TRACE" title="How the agent produced and tested this candidate">
          {!data.has_trace_content ? (
            <div className="unrecorded-banner"><AlertCircle size={16} /><div><strong>内容未记录（历史运行仅保存元数据）</strong><p>本轮探索未采集 Pi 工具参数/结果及模型推理原文。</p></div></div>
          ) : !data.conversation.length ? (
            <Empty title="No linked turns found" text="No trace turns explicitly referenced this candidate digest." />
          ) : (
            <div className="turn-stack">
              {data.conversation.map(t => <ConversationTurnCard key={t.id} turn={t} runId={runId} onInspectCandidate={onSelectEngine} />)}
            </div>
          )}
        </Section>
      )}
    </div>
  )
}

const LANE_COLORS: Record<string, string> = {
  'Model TTFT': '#f2c180',
  'Model Stream': '#5ee6cf',
  'Pi Tools': '#7ab8ff',
  'Research': '#c9b6ff',
  'Worker Build': '#f486a4',
  'Worker Quality': '#a3e635',
  'Worker Measure': '#38bdf8',
  'Worker Other': '#94a3b8',
}

export function TracePageView({ runId, snapshot, scopeEngine, onClearScope, onInspectCandidate, rawEventList }: {
  runId: string; snapshot: Snapshot; scopeEngine?: string; onClearScope: () => void;
  onInspectCandidate: (engine: string) => void; rawEventList: React.ReactNode;
}) {
  const [mode, setMode] = useState<'conversation' | 'waterfall' | 'context' | 'raw'>('conversation')
  const [trace, setTrace] = useState<TraceResponse | null>(null)
  const [waterfall, setWaterfall] = useState<WaterfallData | null>(null)
  const [contextPts, setContextPts] = useState<ContextPoint[]>([])
  const [query, setQuery] = useState('')
  const [tool, setTool] = useState('')
  const [segment, setSegment] = useState('')
  const [errorsOnly, setErrorsOnly] = useState(false)
  const [onlyCandidates, setOnlyCandidates] = useState(false)
  const [offset, setOffset] = useState(0)
  const limit = 25

  useEffect(() => {
    if (mode !== 'conversation') return
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) })
    if (query.trim()) params.set('q', query.trim())
    if (tool) params.set('tool', tool)
    if (segment) params.set('segment', segment)
    if (scopeEngine) params.set('engine', scopeEngine)
    if (errorsOnly) params.set('errors', '1')
    if (onlyCandidates) params.set('candidates', '1')
    let alive = true
    fetch(`/api/runs/${encodeURIComponent(runId)}/trace?${params}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (alive && d) setTrace(d) })
      .catch(() => {})
    return () => { alive = false }
  }, [runId, mode, query, tool, segment, scopeEngine, errorsOnly, onlyCandidates, offset, snapshot.revision])

  useEffect(() => {
    if (mode !== 'waterfall') return
    let alive = true
    fetch(`/api/runs/${encodeURIComponent(runId)}/waterfall`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (alive && d) setWaterfall(d) })
      .catch(() => {})
    return () => { alive = false }
  }, [runId, mode, snapshot.revision])

  useEffect(() => {
    if (mode !== 'context') return
    let alive = true
    fetch(`/api/runs/${encodeURIComponent(runId)}/context`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (alive && d) setContextPts(d.points ?? []) })
      .catch(() => {})
    return () => { alive = false }
  }, [runId, mode, snapshot.revision])

  const hasContent = trace?.has_trace_content ?? snapshot.has_trace_content ?? false

  return (
    <div className="stack-gap">
      <div className="trace-toolbar card">
        <div className="scene-toggle">
          <button className={mode === 'conversation' ? 'selected' : ''} onClick={() => setMode('conversation')}><MessageSquareText size={13} /> 对话视图</button>
          <button className={mode === 'waterfall' ? 'selected' : ''} onClick={() => setMode('waterfall')}><Clock3 size={13} /> 时间瀑布流</button>
          <button className={mode === 'context' ? 'selected' : ''} onClick={() => setMode('context')}><ChartNoAxesCombined size={13} /> 上下文增长图</button>
          <button className={mode === 'raw' ? 'selected' : ''} onClick={() => setMode('raw')}><Activity size={13} /> 原始事件流</button>
        </div>
        {scopeEngine && (
          <button className="button pressed" onClick={onClearScope}>
            Engine {short(scopeEngine)} <X size={13} />
          </button>
        )}
      </div>

      {mode === 'conversation' && (
        <Section kicker="CONVERSATION TRACE" title="Agent reasoning, tool calls & worker outcomes">
          {!hasContent && (
            <div className="unrecorded-banner">
              <AlertCircle size={17} />
              <div>
                <strong>内容未记录（Metadata only）</strong>
                <p>该历史运行在采集阶段未保存模型推理、正文和工具参数/结果。请切换「时间瀑布流」「上下文增长图」或「原始事件流」查看元数据。</p>
              </div>
            </div>
          )}
          <div className="filters wrap">
            <label className="search">
              <ListFilter size={14} />
              <input placeholder="搜索工具、文件名、命令、候选哈希…" value={query} onChange={e => { setQuery(e.target.value); setOffset(0) }} />
            </label>
            <Drop label="Segment" value={segment} onChange={v => { setSegment(v); setOffset(0) }}
              options={[{ key: '', label: 'All segments' }, ...(trace?.segments ?? []).map(s => ({ key: String(s), label: `Segment ${s}` }))]} />
            <Drop label="Tool" value={tool} onChange={v => { setTool(v); setOffset(0) }}
              options={[{ key: '', label: 'All tools' }, ...(trace?.tools ?? []).map(t => ({ key: t, label: t }))]} />
            <button className={`button ${onlyCandidates ? 'pressed' : ''}`} onClick={() => { setOnlyCandidates(!onlyCandidates); setOffset(0) }}>只看提交候选</button>
            <button className={`button ${errorsOnly ? 'pressed' : ''}`} onClick={() => { setErrorsOnly(!errorsOnly); setOffset(0) }}>只看错误</button>
          </div>
          {trace && trace.turns.length > 0 ? (
            <>
              <div className="turn-stack">
                {trace.turns.map(t => <ConversationTurnCard key={t.id} turn={t} runId={runId} onInspectCandidate={onInspectCandidate} />)}
              </div>
              {trace.matched_turns > limit && (
                <div className="pager">
                  <span>{offset + 1}–{Math.min(offset + limit, trace.matched_turns)} of {trace.matched_turns} turns</span>
                  <button className="icon-button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}><ChevronLeft size={16} /></button>
                  <button className="icon-button" disabled={offset + limit >= trace.matched_turns} onClick={() => setOffset(offset + limit)}><ChevronRight size={16} /></button>
                </div>
              )}
            </>
          ) : hasContent ? (
            <Empty title="No matching turns" text="Adjust search or filter criteria." />
          ) : null}
        </Section>
      )}

      {mode === 'waterfall' && (
        <Section kicker="TIME WATERFALL" title="Where wall-clock time was spent">
          {!waterfall || !waterfall.spans.length ? (
            <Empty title="No timed spans available" text="Spans populate live as model turns, tools, and Worker operations run." />
          ) : (
            <div className="waterfall-wrap">
              <div className="breakdown-bar">
                {[
                  ['Model TTFT', waterfall.breakdown.model_ttft_ms, LANE_COLORS['Model TTFT']],
                  ['Model Stream', waterfall.breakdown.model_stream_ms, LANE_COLORS['Model Stream']],
                  ['Pi Tools', waterfall.breakdown.pi_tool_ms, LANE_COLORS['Pi Tools']],
                  ['Research', waterfall.breakdown.research_ms, LANE_COLORS['Research']],
                  ['Worker Build', waterfall.breakdown.build_ms, LANE_COLORS['Worker Build']],
                  ['Worker Quality', waterfall.breakdown.quality_ms, LANE_COLORS['Worker Quality']],
                  ['Worker Measure', waterfall.breakdown.measure_ms, LANE_COLORS['Worker Measure']],
                ].filter(([, ms]) => (ms as number) > 0).map(([label, ms, color]) => (
                  <div key={label as string} className="breakdown-chip">
                    <i style={{ background: color as string }} />
                    <span>{label as string}</span>
                    <strong className="mono">{duration(ms)}</strong>
                  </div>
                ))}
              </div>
              <div className="waterfall-lanes">
                {waterfall.lanes.filter(l => waterfall.spans.some(sp => sp.lane === l)).map(lane => {
                  const laneSpans = waterfall.spans.filter(sp => sp.lane === lane)
                  const total = Math.max(1, waterfall.total_wall_ms)
                  return (
                    <div key={lane} className="wf-row">
                      <div className="wf-label">
                        <i style={{ background: LANE_COLORS[lane] ?? '#7aa' }} />
                        <span>{lane}</span>
                        <small className="mono">{laneSpans.length}</small>
                      </div>
                      <div className="wf-rail">
                        {laneSpans.map((sp, i) => {
                          const left = (sp.start_ms / total) * 100
                          const width = Math.max(0.35, (sp.duration_ms / total) * 100)
                          return (
                            <span
                              key={i}
                              className="wf-bar"
                              style={{ left: `${left}%`, width: `${width}%`, background: LANE_COLORS[lane] ?? '#5ee6cf' }}
                              title={`${sp.label} · ${duration(sp.duration_ms)} (@${duration(sp.start_ms)})`}
                              onClick={() => sp.engine && onInspectCandidate(sp.engine)}
                            />
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
              <p className="side-note">Total span window: {duration(waterfall.total_wall_ms)}. Click any Worker bar to open Candidate Detail.</p>
            </div>
          )}
        </Section>
      )}

      {mode === 'context' && (
        <Section kicker="CONTEXT & PREFIX CACHE" title="Per-request input tokens & cache hit rate">
          <ContextGrowthChart points={contextPts} />
          {contextPts.length > 0 && (
            <div className="breakdown-bar" style={{ padding: '0 16px 14px' }}>
              <div className="breakdown-chip"><span>Total requests</span><strong className="mono">{contextPts.length}</strong></div>
              <div className="breakdown-chip"><span>Peak input</span><strong className="mono">{compact(Math.max(...contextPts.map(p => p.input_tokens ?? 0)))}</strong></div>
              <div className="breakdown-chip"><span>Avg cache hit</span><strong className="mono">{pct(contextPts.filter(p => has(p.cache_hit_rate)).reduce((a, b, _, arr) => a + (b.cache_hit_rate ?? 0) / arr.length, 0), 1)}</strong></div>
            </div>
          )}
        </Section>
      )}

      {mode === 'raw' && rawEventList}
    </div>
  )
}

export function ResearchPageView({ runId, onInspectCandidate }: { runId: string; onInspectCandidate: (e: string) => void }) {
  const [data, setData] = useState<ResearchData | null>(null)
  const [activePage, setActivePage] = useState<{ id: string; title?: string; url?: string; content?: string; truncated?: boolean } | null>(null)

  useEffect(() => {
    let alive = true
    fetch(`/api/runs/${encodeURIComponent(runId)}/research`).then(r => r.ok ? r.json() : null).then(d => alive && setData(d)).catch(() => {})
    return () => { alive = false }
  }, [runId])

  async function openPage(id: string) {
    const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/research/pages/${id}`)
    if (r.ok) setActivePage(await r.json())
  }

  if (!data || !data.available) {
    return <Section kicker="WEB RESEARCH" title="Research provenance"><Empty title="No web research recorded" text="Search queries, fetched pages, and research_reuse receipts appear here when open web research is used." /></Section>
  }

  return (
    <div className="stack-gap">
      {data.reuses.length > 0 && (
        <Section kicker="RESEARCH REUSE PROVENANCE" title="Attributions linked to candidates">
          <div className="table-scroll">
            <table className="table">
              <thead><tr><th>Kind</th><th>Paths</th><th>Source URL / License</th><th>Linked Candidates</th></tr></thead>
              <tbody>
                {data.reuses.map(r => (
                  <tr key={r.receipt}>
                    <td><Badge tone="violet">{words(r.kind)}</Badge></td>
                    <td className="mono">{r.paths.join(', ')}</td>
                    <td>{r.url ? <a className="ext-link" href={r.url} target="_blank" rel="noreferrer">{r.url} <ExternalLink size={11} /></a> : 'Original'} <small>({r.license ?? '—'})</small></td>
                    <td>{r.matched_engines.length ? r.matched_engines.map(e => <button key={e} className="pill-link mono" onClick={() => onInspectCandidate(e)}>{short(e)}</button>) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
      <div className="two-panels">
        <Section kicker="SEARCH QUERIES" title={`Web searches (${data.searches.length})`}>
          {!data.searches.length ? <Empty title="No searches" text="No live web searches recorded." /> : (
            <div className="stack-list">
              {data.searches.map((s, i) => (
                <details key={i} className="receipt-item" open={i === 0}>
                  <summary className="mini-summary">
                    <strong>"{s.query}"</strong> · <span className="mono">{s.results.length} results · {duration(s.elapsed_ms)}</span>
                  </summary>
                  <ul className="search-results">
                    {s.results.map((r, j) => (
                      <li key={j}>
                        <strong>{r.title || r.url}</strong>
                        <small className="mono">{r.url}</small>
                        {r.snippet && <p>{r.snippet}</p>}
                      </li>
                    ))}
                  </ul>
                </details>
              ))}
            </div>
          )}
        </Section>
        <Section kicker="FETCHED PAGES" title={`Archived web pages (${data.pages.length})`}>
          {!data.pages.length ? <Empty title="No fetched pages" text="Fetched pages appear here." /> : (
            <div className="stack-list">
              {data.pages.map(p => (
                <div key={p.id} className="receipt-item">
                  <div className="row-between">
                    <strong>{p.title || short(p.id)}</strong>
                    <button className="text-button" onClick={() => openPage(p.id)}><BookOpen size={13} /> Read ({compact(p.bytes)} B)</button>
                  </div>
                  <small className="mono">{p.url}</small>
                  <p className="preview-p">{p.preview}…</p>
                </div>
              ))}
            </div>
          )}
        </Section>
      </div>
      {activePage && (
        <Section kicker="ARCHIVED PAGE READER" title={activePage.title || short(activePage.id)}
          action={<button className="icon-button" onClick={() => setActivePage(null)}><X size={16} /></button>}>
          <p className="side-note mono">{activePage.url}</p>
          <pre className="page-reader mono">{activePage.content}</pre>
          {activePage.truncated && <p className="side-note">Truncated at 200,000 characters.</p>}
        </Section>
      )}
    </div>
  )
}

export function HarnessPageView({ runId, snapshot }: { runId: string; snapshot: Snapshot }) {
  const [data, setData] = useState<HarnessData | null>(null)
  const [left, setLeft] = useState('')
  const [right, setRight] = useState('')

  useEffect(() => {
    const params = new URLSearchParams()
    if (left) params.set('left', left)
    if (right) params.set('right', right)
    const qs = params.toString() ? `?${params}` : ''
    let alive = true
    fetch(`/api/runs/${encodeURIComponent(runId)}/harness${qs}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!alive || !d) return
        setData(d)
        if (!left && d.diff_left) setLeft(d.diff_left)
        if (!right && d.diff_right) setRight(d.diff_right)
      })
      .catch(() => {})
    return () => { alive = false }
  }, [runId, left, right, snapshot.revision])

  return (
    <div className="stack-gap">
      <Section kicker="FOUR-STAGE HARNESS EVIDENCE" title="Submit · Activate · Attach · Verified Load">
        {!data || !data.versions.length ? (
          <Empty title="No Harness versions stored" text="Harness lineage versions appear once initialized by the controller." />
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead><tr><th>Version</th><th>Files</th><th>Submitted</th><th>Activated</th><th>Attached Sessions</th><th>Prompt Load Proof</th></tr></thead>
              <tbody>
                {data.versions.map(v => (
                  <tr key={v.id} className={v.id === data.active_harness ? 'row-selected' : ''}>
                    <td><Hash value={v.id} /> {v.id === data.active_harness && <Badge tone="good">active</Badge>}</td>
                    <td className="mono">{v.files.length ? v.files.join(', ') : '(empty H0)'}</td>
                    <td><Badge tone={v.evidence.submitted ? 'good' : 'muted'}>{v.evidence.submitted ? 'yes' : 'init'}</Badge></td>
                    <td><Badge tone={v.evidence.activated ? 'good' : 'muted'}>{v.evidence.activated ? 'yes' : 'no'}</Badge></td>
                    <td className="mono">{v.evidence.attached_sessions.length ? v.evidence.attached_sessions.map(s => `S${s}`).join(', ') : '—'}</td>
                    <td className="mono">{v.evidence.prompt_loaded_turns > 0 ? <Badge tone="good">{v.evidence.prompt_loaded_turns} turns</Badge> : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {data && data.versions.length > 0 && (
        <Section kicker="HARNESS VERSION DIFF" title="Compare Harness files"
          action={
            <div className="chart-actions">
              <Drop label="Base version" value={left} onChange={setLeft}
                options={data.versions.map(v => ({ key: v.id, label: `Base: ${short(v.id)}` }))} />
              <Drop label="Target version" value={right} onChange={setRight}
                options={data.versions.map(v => ({ key: v.id, label: `Target: ${short(v.id)}` }))} />
            </div>
          }>
          <UnifiedDiffView files={data.diff} />
        </Section>
      )}
    </div>
  )
}
