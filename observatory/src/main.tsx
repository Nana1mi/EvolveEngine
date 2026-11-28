import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, ArrowDownToLine, ArrowRight, AudioLines, BookOpen, ChartNoAxesCombined,
  ChevronLeft, ChevronRight, CircleDot, CircuitBoard, Clock3, Cpu, FileDiff, Fingerprint,
  Flag, GitBranch, History, Layers3, LayoutDashboard, ListFilter, Menu, Radio, ShieldCheck,
  Waypoints, X,
} from 'lucide-react'
import { Curve, CurveLegend } from './chart'
import { Badge, Drop, Empty, GuardrailStrip, Hash, KeyValues, Section, Stat, W5ComparisonCard, statusStyle } from './components'
import {
  axisList, compact, duration, eventLabel, has, isWorkBuddyV2, metricList, num, pct,
  ratioKeysFor, sceneKeysFor, short, sourceName, timestamp, words, xvalue, yuan, yvalue,
  type Axis, type EventRow, type Metric, type Point, type RunInfo, type Snapshot,
} from './model'
import { CandidateDetailView, HarnessPageView, ResearchPageView, TracePageView } from './views'
import './styles.css'

const nav = [
  {
    group: 'OBSERVE',
    items: [
      ['overview', 'Overview', LayoutDashboard],
      ['candidate', 'Candidate detail', FileDiff],
      ['trace', 'Live trace', Activity],
      ['history', 'Update history', History],
      ['experiments', 'Experiments', Layers3],
    ],
  },
  {
    group: 'LINEAGE & EVIDENCE',
    items: [
      ['engine', 'Engine', GitBranch],
      ['harness', 'Harness', CircuitBoard],
      ['evaluation', 'Evaluation', ShieldCheck],
      ['research', 'Research', BookOpen],
      ['resources', 'Resources', Cpu],
    ],
  },
] as const
type Page = typeof nav[number]['items'][number][0]
const pages = [...nav[0].items, ...nav[1].items]
const isPage = (s: string): s is Page => pages.some(([p]) => p === s)

function useRuns() {
  const [runs, setRuns] = useState<RunInfo[]>([]), [run, setRun] = useState(''), [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/runs', { signal: controller.signal }).then(r => { if (!r.ok) throw Error(); return r.json() }).then(r => {
      setRuns(r.runs)
      const id = new URLSearchParams(location.search).get('run')
      setRun(r.runs.find((v: RunInfo) => v.id === id)?.id ?? r.runs[0]?.id ?? '')
    }).catch(() => { if (!controller.signal.aborted) setError('Could not list runs. Check the local Observatory process.') })
    return () => controller.abort()
  }, [])
  return {
    runs, run, error,
    choose: (id: string) => { setRun(id); const u = new URL(location.href); u.searchParams.set('run', id); history.replaceState(null, '', u) },
  }
}

function useSnapshot(run: RunInfo | undefined) {
  const [snapshot, setSnapshot] = useState<Snapshot>(), [connection, setConnection] = useState('Connecting'), [error, setError] = useState('')
  const id = run?.id, mode = run?.mode
  useEffect(() => {
    setSnapshot(undefined); setConnection(mode === 'archive' ? 'Archive' : 'Connecting'); setError('')
    if (!id) return
    const controller = new AbortController()
    const base = `/api/runs/${encodeURIComponent(id)}`
    function update(v: Snapshot) {
      if (controller.signal.aborted || v.schema_version !== 1 || v.id !== id) return
      setSnapshot(old => old?.revision === v.revision ? old : v)
      setError('')
    }
    if (mode === 'archive') {
      fetch(base + '/snapshot', { signal: controller.signal }).then(r => { if (!r.ok) throw Error(); return r.json() }).then(update)
        .catch(() => { if (!controller.signal.aborted) setError('Archive is unavailable.') })
      return () => controller.abort()
    }
    const stream = new EventSource(base + '/events')
    stream.onopen = () => setConnection('Following receipts')
    stream.onerror = () => setConnection('Reconnecting')
    stream.addEventListener('snapshot', e => {
      try { update(JSON.parse(e.data)); setConnection('Following receipts') } catch { setError('Snapshot could not be decoded.') }
    })
    return () => { controller.abort(); stream.close() }
  }, [id, mode])
  return { snapshot: snapshot?.id === id ? snapshot : undefined, connection, error }
}

function EventList({ rows, selected, onClick, filter = false }: {
  rows: EventRow[]; selected?: string; onClick: (e: EventRow) => void; filter?: boolean;
}) {
  const [query, setQuery] = useState(''), [group, setGroup] = useState('all'), [offset, setOffset] = useState(0)
  const stream = (e: EventRow) => e.kind === 'model_request' ? 1 : e.kind === 'pi_tool' || e.kind === 'session_harness_attached' ? 2 : e.id.startsWith('annotations/') ? 3 : 0
  const sorted = useMemo(() => [0, 1, 2, 3].flatMap(k => rows.filter(e => stream(e) === k).reverse()), [rows])
  const subset = sorted.filter(e => (group === 'all' || (group === 'failed' ? e.failure : group === e.source)) &&
    (!query || `${eventLabel(e)} ${e.status ?? ''} ${e.continuation ?? ''} ${e.cause ?? ''} ${e.operation ?? ''} ${e.engine ?? ''} ${e.evidence ?? ''}`.toLowerCase().includes(query.toLowerCase())))
  const size = filter ? 25 : 7, start = Math.min(offset, Math.max(0, Math.ceil(subset.length / size) - 1))
  return <>
    {filter && (
      <div className="filters">
        <label className="search"><ListFilter size={15} /><input aria-label="Filter events" placeholder="Filter by event, status, or hash…" value={query} onChange={e => { setQuery(e.target.value); setOffset(0) }} /></label>
        <Drop label="Event group" value={group} onChange={v => { setGroup(v); setOffset(0) }} options={[{ key: 'all', label: 'All sources' }, { key: 'verified', label: 'Verified' }, { key: 'host', label: 'Host' }, { key: 'agent', label: 'Agent' }, { key: 'failed', label: 'Failures only' }]} />
      </div>
    )}
    {subset.length === 0 ? <Empty title="No matching entries" text="Only recorded events are shown." /> : (
      <div className="table-scroll"><table className="table"><thead><tr><th>Event</th><th>Engine / session</th><th>Status</th><th>Time</th></tr></thead><tbody>
        {subset.slice(start * size, (start + 1) * size).map(e => (
          <tr key={e.id} className={selected === e.id ? 'row-selected' : ''}>
            <td><button className="event-link" onClick={() => onClick(e)}><span className={`event-mark ${e.failure ? 'danger' : e.source}`}><CircleDot size={13} /></span><span><strong>{eventLabel(e)}</strong><small>{e.operation ? words(e.operation) : sourceName[e.source]}</small></span></button></td>
            <td><span className="mono">{e.engine ? (e.iteration == null ? short(e.engine) : `E${e.iteration} · ${short(e.engine)}`) : e.session ? `S${e.session}` : '—'}</span></td>
            <td><Badge tone={e.failure ? 'bad' : statusStyle(e.status ?? '')}>{words(e.finish === 'length' ? 'generation_truncated' : e.status ?? e.continuation ?? e.cause ?? e.source)}</Badge></td>
            <td className="mono">{has(e.wall_ms) ? duration(e.wall_ms) : has(e.at_ms) ? timestamp(e.at_ms) : '—'}</td>
          </tr>
        ))}
      </tbody></table></div>
    )}
    {filter && subset.length > size && (
      <div className="pager"><span>{start * size + 1}–{Math.min((start + 1) * size, subset.length)} of {subset.length}</span>
        <button className="icon-button" disabled={start === 0} onClick={() => setOffset(start - 1)}><ChevronLeft size={17} /></button>
        <button className="icon-button" disabled={(start + 1) * size >= subset.length} onClick={() => setOffset(start + 1)}><ChevronRight size={17} /></button>
      </div>
    )}
  </>
}

function PointTable({ data, selected, onClick, onOpenCandidate }: {
  data: Snapshot; selected?: string; onClick: (id: string) => void; onOpenCandidate: (engine: string) => void;
}) {
  if (!data.points.length) return <Empty title="No measured candidates" text="Failed or unmeasured attempts stay in the event log." />
  const rKeys = ratioKeysFor(data)
  return (
    <div className="table-scroll">
      <table className="table">
        <thead><tr><th>Engine / measurement</th><th>Score</th>{rKeys.map(k => <th key={k}>{k}</th>)}<th>Wall</th><th>Detail</th></tr></thead>
        <tbody>
          {[...data.points].reverse().map(p => (
            <tr key={p.id} className={selected === p.id ? 'row-selected' : ''}>
              <td><button className="event-link" onClick={() => onClick(p.id)}><span className="event-mark verified"><GitBranch size={14} /></span><span><strong>E{p.iteration} · {p.origin === 'operator_replay' ? 'Replay' : `Pair ${p.pair ?? '—'}`}</strong><small className="mono">{short(p.engine)}</small></span></button></td>
              <td className="mono ink">{has(p.aggregate) ? `${num(p.aggregate, 3)}×` : '—'}</td>
              {rKeys.map(k => <td className="mono" key={k}>{has(p.ratios[k]) ? `${num(p.ratios[k], 3)}×` : '—'}</td>)}
              <td className="mono">{duration(p.resources?.wall_ms)}</td>
              <td><button className="text-button" onClick={() => onOpenCandidate(p.engine)}>Diff & Trace <ArrowRight size={13} /></button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Inspector({ point, row, snapshot, onNavigate, onOpenCandidate, dismiss }: {
  point?: Point; row?: EventRow; snapshot: Snapshot; onNavigate: (p: Page) => void;
  onOpenCandidate: (engine: string) => void; dismiss?: () => void;
}) {
  const [tab, setTab] = useState<'evidence' | 'scenes'>('evidence')
  const quality = point && snapshot.events.filter(e => e.engine === point.engine && e.build === point.build && e.operation === 'develop_quality' && e.cases).at(-1)
  const summary = point && snapshot.build_summaries?.find(b => b.engine === point.engine && (!b.build || b.build === point.build))
  const rKeys = ratioKeysFor(snapshot)
  return (
    <Section className="inspector" kicker="EVIDENCE INSPECTOR" title={row ? eventLabel(row) : point ? `Engine E${point.iteration}` : 'Nothing selected'}
      action={dismiss && <button className="icon-button" aria-label="Close inspector" onClick={dismiss}><X size={17} /></button>}>
      {row ? (
        <div className="inspector-content">
          <div className="inline-badges"><Badge tone={row.failure ? 'bad' : 'muted'}>{words(row.status ?? row.cause ?? row.kind)}</Badge><Badge tone={row.source === 'verified' ? 'good' : 'muted'}>{sourceName[row.source]}</Badge></div>
          <KeyValues entries={[
            ['Engine', <Hash value={row.engine} />], ['Evidence', <Hash value={row.evidence} />], ['Session', row.session ?? '—'],
            ['Operation', words(row.operation)], ['Duration', duration(row.duration_ms)],
            ...(row.harness ? [['Harness', <Hash value={row.harness} />] as [string, ReactNode]] : []),
            ...(row.kind === 'model_request' ? [
              ['Finish', words(row.finish)], ['Input / Output', `${compact(row.input_tokens)} / ${compact(row.output_tokens)}`],
              ['Reasoning', compact(row.reasoning_tokens)], ['Cache hit / miss', `${compact(row.cache_hit_tokens)} / ${compact(row.cache_miss_tokens)}`],
              ['First delta', duration(row.first_delta_ms)],
            ] as [ReactNode, ReactNode][] : []),
            ...(row.cases ? Object.entries(row.cases).map(([k, v]) => [words(k), <Badge tone={statusStyle(v)}>{words(v)}</Badge>] as [string, ReactNode]) : []),
          ]} />
          {row.engine && <div className="inspector-links"><button onClick={() => onOpenCandidate(row.engine!)}><FileDiff size={14} />Open Candidate Detail <ArrowRight size={13} /></button></div>}
        </div>
      ) : point ? (
        <div className="inspector-content">
          <div className="inline-badges">
            <Badge tone={point.provisional ? 'warn' : 'good'}>{point.provisional ? 'Weight audit pending' : 'Development receipt'}</Badge>
            {summary && <Badge tone={statusStyle(summary.status)}>{summary.pairs}/{summary.minimum_pairs} pairs</Badge>}
            {snapshot.delivery?.engine === point.engine && <Badge tone="violet">Agent selected</Badge>}
          </div>
          <div className="inspector-value">
            <strong>{num(point.aggregate, 3)}<em>{has(point.aggregate) ? '×' : ''}</em></strong>
            <span>vs. paired E0 · {summary?.ci_available ? '6+ pairs (95% CI available)' : 'single paired point ratio'}</span>
          </div>
          <button className="primary-cta" onClick={() => onOpenCandidate(point.engine)}>
            <FileDiff size={15} /><span>查看代码 Diff · 质检 · 对话</span><ArrowRight size={14} />
          </button>
          <div className="tabs">
            <button className={tab === 'evidence' ? 'selected' : ''} onClick={() => setTab('evidence')}>Evidence</button>
            <button className={tab === 'scenes' ? 'selected' : ''} onClick={() => setTab('scenes')}>{isWorkBuddyV2(snapshot) ? 'W1–W5 (6)' : 'W1–W4'}</button>
          </div>
          {tab === 'evidence' ? (
            <KeyValues entries={[
              ['Engine', <Hash value={point.engine} />], ['Build', <Hash value={point.build} />],
              ['Measurement', <Hash value={point.id} />],
              ['Quality', quality ? <Badge tone={statusStyle(quality.status ?? '')}>{words(quality.status)}</Badge> : point.known_quality ? 'Development qualified' : 'Not recorded'],
              ['Wall time', duration(point.resources?.wall_ms)], ['Session', point.session ?? '—'],
            ]} />
          ) : (
            <div className="scenes">
              {rKeys.map(s => {
                const ci = summary?.ci_available ? summary.intervals?.[s] : undefined
                return (
                  <div key={s}>
                    <div><b>{s}</b><small className="mono">{ci ? `[${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : has(point.scenes[s]?.output_toks_per_s) ? `${num(point.scenes[s].output_toks_per_s, 1)} tok/s` : ''}</small></div>
                    <span className="mono">{has(point.ratios[s]) ? `${num(point.ratios[s], 3)}×` : '—'}</span>
                  </div>
                )
              })}
            </div>
          )}
          <div className="inspector-links">
            <button onClick={() => onNavigate('trace')}><Activity size={14} />Trace <ArrowRight size={12} /></button>
            <button onClick={() => onNavigate('evaluation')}><ShieldCheck size={14} />Evaluation <ArrowRight size={12} /></button>
            <button onClick={() => onNavigate('engine')}><GitBranch size={14} />Lineage <ArrowRight size={12} /></button>
          </div>
        </div>
      ) : <Empty title="Select a performance point" text="Click any point on the curve to inspect code diffs, quality gates, and linked trace turns." />}
    </Section>
  )
}

function Tracks({ rows, data, axis, onClick }: { rows: EventRow[]; data: Point[]; axis: Axis; onClick: (e: EventRow) => void }) {
  const shown = ['Failed or uncertain', 'Harness / session', 'Pause / infrastructure'].map((label, i) => ({
    label,
    rows: rows.filter(e => i === 0 ? e.failure : i === 1 ? e.kind.startsWith('harness_') || e.kind === 'session_start' || e.kind === 'segment_started' : ['pause', 'resume', 'infrastructure_change', 'infrastructure_fault'].includes(e.kind)),
  }))
  const xs = data.map(p => xvalue(p, axis)).filter(has)
  return (
    <div className="tracks">
      {shown.map(({ label, rows: r }, i) => {
        const time = (e: EventRow) => axis === 'effective_ms' ? e.effective_ms : e.wall_ms
        const aligned = (axis === 'wall_ms' || axis === 'effective_ms') ? r.filter(e => { const n = time(e); return has(n) && xs.length && n >= Math.min(...xs) && n <= Math.max(...xs) }) : []
        const omitted = r.length - aligned.length
        return (
          <div className="track" key={label}>
            <span><i className={i ? 'violet-dot' : 'pink-dot'} />{label}</span>
            <div className="track-rail">
              {aligned.map(e => <button title={`${eventLabel(e)} · ${duration(time(e))}`} className={i ? 'violet-dot' : 'pink-dot'} key={e.id} style={{ left: `${100 * ((time(e) ?? 0) - Math.min(...xs)) / Math.max(1, Math.max(...xs) - Math.min(...xs))}%` }} onClick={() => onClick(e)} />)}
            </div>
            <button className="count-link" disabled={!r.length} onClick={() => r.length && onClick(r.at(-1)!)}>{r.length} {omitted ? '· ' + omitted + ' unplaced' : ''}</button>
          </div>
        )
      })}
    </div>
  )
}

function Resources({ s }: { s: Snapshot }) {
  const r = s.resources
  return (
    <div className="two-panels">
      <Section kicker="RUN ACCOUNTING" title="Resource settlement"><KeyValues entries={[
        ['Wall time', duration(r.wall_ms)], ['Effective exploration', duration(r.effective_ms)], ['Charged node time', duration(r.node_ms)],
        ['Settled node time', duration(r.settled_node_ms)], ['Unsettled node reservation', duration(r.reserved_node_ms)],
        ['Settled model requests', num(r.requests, 0)], ['Unresolved requests', num(r.unresolved_requests, 0)],
      ]} /></Section>
      <Section kicker="BROKER ACCOUNTING" title="Model and cost"><KeyValues entries={[
        ['Input tokens', compact(r.input_tokens)], ['Output incl. reasoning', compact(r.output_tokens)], ['Reported reasoning', compact(r.reasoning_tokens)],
        ['Cache hit / miss', `${compact(r.cache_hit_tokens)} / ${compact(r.cache_miss_tokens)}`], ['Conservative local ledger', yuan(r.ledger_rmb)],
        ['Usage fee estimate', `${yuan(r.fee_idle)} – ${yuan(r.fee_peak)}`],
      ]} /></Section>
      <Section kicker="REGISTERED CEILINGS" title="Run limits"><KeyValues entries={[
        ['Input / output', `${compact(s.limits.input)} / ${compact(s.limits.output)}`], ['Per request output', compact(s.limits.per_turn_output)],
        ['Model turns', num(s.limits.turns, 0)], ['Wall clock', duration(s.limits.wall_ms)],
      ]} /></Section>
      <Section kicker="TRACE COVERAGE" title="Observability status"><KeyValues entries={[
        ['Projected request records', num(r.model_records, 0)],
        ['Full trace capture', s.has_trace_content ? <Badge tone="good">Recorded</Badge> : <Badge tone="warn">Metadata only</Badge>],
        ['Scoring policy', s.scoring?.version ?? 'Legacy'],
      ]} /></Section>
    </div>
  )
}

function Export({ s }: { s: Snapshot }) {
  function download() {
    const rKeys = ratioKeysFor(s)
    const wb2 = isWorkBuddyV2(s)
    const columns = [
      'measurement', 'engine', 'build', 'paired_reference', 'pair', 'metric_method', 'development_ratio',
      ...rKeys.map(k => `${k}_ratio`),
      ...(wb2 ? ['w4_cold_e2e_ratio', 'g64k_ttft_ratio', 'g64k_decode_ratio', 'p95_itl_ratio', 'unfinished_count',
        'W5_p20_tps', 'W5_ttft_ms', 'W5_wave_tps', 'W5_share_ge_40', 'W5_share_ge_50', 'W3_cached_min', 'W5_cached_min'] : []),
      'wall_ms', 'effective_ms', 'node_ms', 'settled_input_at_dispatch', 'settled_output_at_dispatch', 'ledger_rmb_at_dispatch', 'provisional', 'origin',
    ]
    const rows = s.points.map(p => [
      p.id, p.engine, p.build, p.reference, p.pair, p.aggregate_kind, p.aggregate,
      ...rKeys.map(k => p.ratios[k]),
      ...(wb2 ? [
        p.guardrail_ratios?.w4_cold_e2e, p.guardrail_ratios?.g64k_ttft, p.guardrail_ratios?.g64k_decode, p.guardrail_ratios?.p95_itl,
        p.unfinished_count ?? '', p.scenes.W5?.dev_score, p.scenes.W5?.ttft_ms, p.scenes.W5?.output_toks_per_s,
        p.scenes.W5?.share_ge_40_tps, p.scenes.W5?.share_ge_50_tps, p.scenes.W3?.cached_tokens_min, p.scenes.W5?.cached_tokens_min,
      ] : []),
      p.resources?.wall_ms, p.resources?.effective_ms, p.resources?.node_ms,
      p.resources?.input_tokens, p.resources?.output_tokens, p.resources?.ledger_rmb, !!p.provisional, p.origin ?? 'controller',
    ])
    const url = URL.createObjectURL(new Blob([[columns, ...rows].map(x => x.join(',')).join('\r\n') + '\r\n'], { type: 'text/csv' }))
    const a = document.createElement('a'); a.href = url; a.download = `${s.id}-development-measurements.csv`; a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <button className="button" onClick={download} disabled={!s.points.length}><ArrowDownToLine size={15} /> <span>Export CSV</span></button>
}

const pageText: Partial<Record<Page, { title: string; subtitle: string }>> = {
  candidate: { title: 'Candidate investigation', subtitle: 'Inspect exact source & ee_runtime diffs, quality gates, WorkBuddy v2 metrics, and ID-linked reasoning.' },
  trace: { title: 'Live trace', subtitle: 'Switch between Conversation View, Time Waterfall, Context Growth, and Raw Event Stream.' },
  history: { title: 'Update history', subtitle: 'Compare measured candidates across all registered scoring scenes and guardrails.' },
  experiments: { title: 'Experiments', subtitle: 'Switch between explicitly registered run roots and historical archives.' },
  engine: { title: 'Engine lineage', subtitle: 'Trace registered engines, parent identities, and open full candidate diffs.' },
  harness: { title: 'Harness lineage', subtitle: 'Verify submit, activate, launch attachment, and prompt-load proofs alongside Harness file diffs.' },
  evaluation: { title: 'Evaluation', subtitle: 'Development quality gates (including 16K/64K long-context) and paired measurement receipts.' },
  research: { title: 'Research', subtitle: 'Inspect live web searches, archived pages, and research_reuse provenance linked to candidates.' },
  resources: { title: 'Resources', subtitle: 'Observe separate wall time, GPU node time, and broker token ledgers.' },
}

function App() {
  const { runs, run, error: listError, choose } = useRuns()
  const info = runs.find(v => v.id === run)
  const { snapshot: s, connection, error } = useSnapshot(info)
  const initial = location.hash.slice(1)
  const [page, setPage] = useState<Page>(isPage(initial) ? initial : 'overview')
  const [mobile, setMobile] = useState(false), [selected, setSelected] = useState<string>(), [selectedRow, setSelectedRow] = useState<EventRow>()
  const [candidateEngine, setCandidateEngine] = useState<string>()
  const [scope, setScope] = useState(false), [replay, setReplay] = useState<number>(), [axis, setAxis] = useState<Axis>()
  const [metric, setMetric] = useState<Metric>('aggregate'), [scene, setScene] = useState('W1')

  useEffect(() => {
    const handler = () => { const p = location.hash.slice(1); if (isPage(p)) setPage(p) }
    addEventListener('hashchange', handler)
    return () => removeEventListener('hashchange', handler)
  }, [])
  useEffect(() => {
    setSelected(undefined); setSelectedRow(undefined); setCandidateEngine(undefined)
    setAxis(undefined); setReplay(undefined); setMetric('aggregate'); setScope(false)
  }, [run])

  const navigate = (p: Page) => { location.hash = p; setPage(p); setMobile(false); if (p === 'trace' || p === 'evaluation') setScope(!!selected || !!selectedRow?.engine) }
  const select = (id: string) => { setSelected(id); setSelectedRow(undefined) }
  const openCandidate = (eng: string) => { setCandidateEngine(eng); navigate('candidate') }
  const view = (e: EventRow) => { if (e.point_id) { select(e.point_id); return } setSelectedRow(e) }

  const visible = s?.points.filter(p => !replay || p.index <= replay) ?? []
  const aggregateKind = visible.findLast(p => has(p.aggregate))?.aggregate_kind
  const chartData = metric === 'aggregate' ? visible.filter(p => p.aggregate_kind === aggregateKind) : visible
  const latest = chartData.at(-1)
  const currentAxis = axis ?? (latest?.origin === 'operator_replay' ? 'index' :
    visible.some(p => has(xvalue(p, 'effective_ms'))) ? 'effective_ms' : visible.some(p => has(xvalue(p, 'wall_ms'))) ? 'wall_ms' : 'index')
  const p = visible.find(p => p.id === selected) ?? visible.findLast(p => s?.delivery?.point_id === p.id) ?? visible.at(-1)
  const m = metricList.find(v => v.key === metric)!
  const activeScenes = metric === 'ratio' ? ratioKeysFor(s) : sceneKeysFor(s)
  const effectiveScene = activeScenes.includes(scene) ? scene : activeScenes[0]
  const best = chartData.filter(pt => has(yvalue(pt, metric, effectiveScene))).reduce<Point | undefined>((acc, pt) =>
    !acc || (m.high ? Number(yvalue(pt, metric, effectiveScene)) > Number(yvalue(acc, metric, effectiveScene)) : Number(yvalue(pt, metric, effectiveScene)) < Number(yvalue(acc, metric, effectiveScene))) ? pt : acc, undefined)
  const events = s?.events ?? []
  const sessions = new Set(events.map(e => e.session).filter(has))
  const summaryEvents = events.filter(e => ['comparison', 'operator_measurement', 'engine_registered', 'harness_submit', 'harness_activate', 'delivery', 'run_settlement', 'checkpoint_saved', 'pause', 'resume', 'infrastructure_change'].includes(e.kind) || e.failure)
  const traceEngine = selectedRow?.engine ?? p?.engine ?? candidateEngine
  const scoped = (rows: EventRow[]) => scope && traceEngine ? rows.filter(e => e.engine === traceEngine) : rows
  const chartAxisOptions = axisList.map(a => ({ ...a, disabled: a.key !== 'index' && !visible.some(pt => has(xvalue(pt, a.key))) }))
  const chartMetricOptions = metricList.map(mt => ({ ...mt, disabled: mt.key !== 'aggregate' && !visible.some(pt => activeScenes.some(sc => has(yvalue(pt, mt.key, sc)))) }))
  const text = pageText[page] ?? { title: 'Evolution overview', subtitle: isWorkBuddyV2(s) ? 'WorkBuddy v2 Agent Serving (W1–W5 + 64K/ITL guardrails) with full ID-correlated trace & diff drill-down.' : 'Track engine performance, harness changes, and the evidence behind each upgrade.' }
  const trail = page === 'overview' ? 'Overview' : pages.find(([key]) => key === page)?.[1]
  const panelNav = (pg: Page) => { navigate(pg); if (traceEngine) setScope(true) }
  const activeCandidate = candidateEngine ?? p?.engine ?? s?.versions.at(-1)?.id
  const latestSummary = p && s?.build_summaries?.find(b => b.engine === p.engine && (!b.build || b.build === p.build))

  return (
    <div className="app">
      {mobile && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobile(false)} />}
      <aside className={`sidebar ${mobile ? 'open' : ''}`}>
        <a className="brand" href="#overview" onClick={() => navigate('overview')}><span className="logo"><Waypoints size={22} strokeWidth={2.1} /></span><span><strong>EvolveEngine</strong><small>RSI OBSERVATORY</small></span></a>
        <div className="space-card"><span className="space-icon"><CircuitBoard size={18} /></span><span><strong>Inference Lab</strong><small>{s?.scoring?.version ?? 'Observability workspace'}</small></span></div>
        <nav aria-label="Main navigation">
          {nav.map(group => (
            <div className="nav-group" key={group.group}>
              <p>{group.group}</p>
              {group.items.map(([key, label, Icon]) => (
                <button key={key} onClick={() => { navigate(key); setScope(false) }} className={page === key ? 'nav-active' : ''} aria-current={page === key ? 'page' : undefined}>
                  <Icon size={17} /><span>{label}</span>{page === key && <i />}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot"><div className="foot-indicator"><ShieldCheck size={17} /></div><span><strong>Read-only mode</strong><small>Loopback / SSH tunnel</small></span></div>
      </aside>
      <div className="body">
        <header className="topbar">
          <div className="breadcrumb">
            <button onClick={() => setMobile(true)} className="icon-button mobile-menu" aria-label="Open navigation"><Menu size={19} /></button>
            <Waypoints size={16} /><span>Workspace</span><ChevronRight size={14} /><b>{trail}</b>
          </div>
          <div className="top-actions">
            {s && !s.has_trace_content && <Badge tone="warn">Trace 内容未记录</Badge>}
            <span className={`connection ${connection === 'Following receipts' ? 'following' : ''}`}><Radio size={14} />{connection}</span>
            {runs.length > 0 && <Drop label="Choose run" value={run} onChange={choose} options={runs.map(v => ({ key: v.id, label: v.label }))} />}
          </div>
        </header>
        <main id="main-content">
          <div className="hero">
            <div>
              <div className="eyebrow hero-kicker"><i /> LEVEL 3 <span>/</span> {isWorkBuddyV2(s) ? 'WORKBUDDY V2 AGENT SERVING' : 'ENGINEERING EXPLORATION'}</div>
              <h1>{text.title}</h1>
              <p>{text.subtitle}</p>
            </div>
            {s && (
              <div className="hero-actions">
                <Badge tone={statusStyle(s.status)}>{s.status === 'no_terminal_receipt' ? 'No settlement yet' : words(s.status)}</Badge>
                <Export s={s} />
              </div>
            )}
          </div>
          {!s ? (
            <section className="card"><Empty title={error || listError || (!info && runs.length === 0 ? 'Waiting for run data' : 'Reading run evidence')} text={info ? 'The UI does not request or start experiments.' : 'Register a controller run or load the local archive if available.'} /></section>
          ) : (
            <>
              {page === 'overview' && <>
                <div className="stats">
                  <Stat icon={Cpu} label="Engine versions" value={s.versions.length} hint={s.delivery ? `Agent decision: ${short(s.delivery.engine)}` : 'No explicit delivery recorded'} accent="teal" />
                  <Stat icon={ChartNoAxesCombined} label={metric === 'aggregate' ? 'Best score ratio' : `Best ${m.label}`} value={best ? `${num(yvalue(best, metric, effectiveScene), m.unit === '×' ? 3 : 2)}${m.unit}` : '—'} hint={best ? `E${best.iteration} · Pair ${best.pair ?? '—'} · ${s.scoring?.version ?? 'dev'}` : 'No comparable measurement'} accent="violet" />
                  <Stat icon={Clock3} label="Wall time" value={duration(s.resources.wall_ms)} hint={`Node time: ${duration(s.resources.node_ms)}`} accent="blue" />
                  <Stat icon={AudioLines} label="Model requests" value={num(s.resources.requests, 0)} hint={`${compact(s.resources.input_tokens)} in · ${compact(s.resources.output_tokens)} out`} accent="amber" />
                </div>

                {p && <GuardrailStrip point={p} summary={latestSummary} scoring={s.scoring} />}
                {p && p.scenes.W5 && <W5ComparisonCard point={p} />}

                <div className="overview-grid">
                  <Section title="Evolution performance curve" kicker="MEASURED HISTORY" className="curve-card"
                    action={<div className="chart-actions"><Drop label="Chart metric" value={metric} onChange={setMetric} options={chartMetricOptions} /><Drop label="Horizontal axis" value={currentAxis} onChange={setAxis} options={chartAxisOptions} /></div>}>
                    <div className="chart-subline">
                      <span>{m.label} {metric === 'aggregate' ? `· ${s.scoring?.version ?? (aggregateKind === 'weighted_score' ? 'Registered weighted score' : 'Equal-scene geomean')}` : ''}</span>
                      {metric !== 'aggregate' && (
                        <div className="scene-toggle" aria-label="Workload">
                          {activeScenes.map(a => <button className={effectiveScene === a ? 'selected' : ''} key={a} onClick={() => setScene(a)}>{a}</button>)}
                        </div>
                      )}
                    </div>
                    <Curve points={chartData} selected={selectedRow ? undefined : p?.id} metric={metric} axis={currentAxis} scene={effectiveScene} onSelect={select} delivered={s.delivery?.point_id}
                      infrastructure={events.flatMap(e => { const x = currentAxis === 'effective_ms' ? e.effective_ms : currentAxis === 'wall_ms' ? e.wall_ms : undefined; return e.kind === 'infrastructure_change' && has(x) ? [{ x, label: `Infra ${short(e.evidence)}` }] : [] })} />
                    <CurveLegend />
                    <Tracks rows={events} data={chartData} axis={currentAxis} onClick={view} />
                    {s.points.length > 1 && (
                      <div className="replay"><History size={15} /><label htmlFor="replay">Measure replay</label><input id="replay" type="range" min={1} max={s.points.length} value={replay ?? s.points.length} onChange={e => { const n = Number(e.target.value); setReplay(n === s.points.length ? undefined : n); setSelectedRow(undefined) }} /><span>{visible.length}/{s.points.length}</span><button className="text-button" disabled={!replay} onClick={() => setReplay(undefined)}>Latest</button></div>
                    )}
                  </Section>
                  <Inspector key={selectedRow?.id ?? p?.id ?? 'none'} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav} onOpenCandidate={openCandidate} dismiss={selectedRow ? () => setSelectedRow(undefined) : undefined} />
                </div>

                <div className="overview-grid lower">
                  <Section kicker="EVENT JOURNAL" title="Recent updates" action={<button className="text-button" onClick={() => navigate('history')}>View history <ArrowRight size={14} /></button>}>
                    <EventList rows={summaryEvents} selected={selectedRow?.id} onClick={view} />
                  </Section>
                  <Section kicker="TRUST BOUNDARY" title="Evidence provenance">
                    <div className="provenance">
                      <div><ShieldCheck /><span><b>Verified receipts</b><small>Worker builds, quality gates & paired timing</small></span></div>
                      <div><Fingerprint /><span><b>Host observation</b><small>Redacted Pi stream, broker replies & clocks</small></span></div>
                      <div><Flag /><span><b>Agent decisions</b><small>Candidate submissions, Harness & delivery</small></span></div>
                    </div>
                    <p className="side-note">Unmeasured failures stay in event tracks. No holdout or production claim is implied.</p>
                  </Section>
                </div>
              </>}

              {page === 'candidate' && (activeCandidate ? (
                <CandidateDetailView runId={s.id} engine={activeCandidate} snapshot={s} onBack={() => navigate('overview')} onSelectEngine={openCandidate} />
              ) : (
                <Section kicker="CANDIDATE DETAIL" title="No candidate selected"><Empty title="No registered engines yet" text="Candidates appear after engine registration." /></Section>
              ))}

              {page === 'trace' && (
                <TracePageView
                  runId={s.id} snapshot={s} scopeEngine={scope ? traceEngine : undefined}
                  onClearScope={() => setScope(false)} onInspectCandidate={openCandidate}
                  rawEventList={
                    <div className="detail-grid">
                      <Section kicker="SANITIZED EVENTS" title="Recorded events"><EventList key={run + ':trace'} rows={scoped(events)} selected={selectedRow?.id} onClick={view} filter /></Section>
                      <Inspector key={selectedRow?.id ?? p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav} onOpenCandidate={openCandidate} />
                    </div>
                  }
                />
              )}

              {page === 'history' && <>
                <div className="detail-grid">
                  <Section kicker="PAIRED DEVELOPMENT MEASUREMENTS" title="Candidate updates"><PointTable data={s} selected={p?.id} onClick={select} onOpenCandidate={openCandidate} /></Section>
                  <Inspector key={selectedRow?.id ?? p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav} onOpenCandidate={openCandidate} />
                </div>
                <Section kicker="ALL RECORDED OUTCOMES" title="Attempt and update log"><EventList key={run + ':history'} rows={summaryEvents} onClick={view} selected={selectedRow?.id} filter /></Section>
              </>}

              {page === 'experiments' && (
                <Section kicker="RUN REGISTRY" title="Available experiments">
                  <div className="run-list">{runs.map(r => (
                    <button key={r.id} onClick={() => choose(r.id)} className={r.id === s.id ? 'run-selected' : ''}>
                      <Layers3 size={21} /><span><strong>{r.label}</strong><small>{r.mode === 'archive' ? 'Local archive summary (metadata only)' : 'Explicit controller evidence root'}</small></span>
                      <Badge tone={r.id === s.id ? 'good' : 'muted'}>{r.id === s.id ? 'Selected' : r.mode}</Badge><ArrowRight size={16} />
                    </button>
                  ))}</div>
                </Section>
              )}

              {page === 'engine' && (
                <Section kicker="ENGINE IDENTITY" title="Candidate lineage">
                  <div className="table-scroll">
                    <table className="table">
                      <thead><tr><th>Version</th><th>Engine</th><th>Registered parent</th><th>Score / Pairs</th><th>Action</th></tr></thead>
                      <tbody>
                        {s.versions.map(v => {
                          const pt = s.points.findLast(x => x.engine === v.id)
                          const sum = s.build_summaries?.find(b => b.engine === v.id)
                          return (
                            <tr key={v.id}>
                              <td><strong>E{v.iteration}</strong></td>
                              <td><Hash value={v.id} /></td>
                              <td><Hash value={v.parent} /></td>
                              <td className="mono">{sum && has(sum.weighted_score) ? `${num(sum.weighted_score, 3)}× (${sum.pairs}/${sum.minimum_pairs})` : pt && has(pt.aggregate) ? `${num(pt.aggregate, 3)}×` : <Badge>Not measured</Badge>}</td>
                              <td><button className="text-button" onClick={() => openCandidate(v.id)}>Diff & Trace <ArrowRight size={14} /></button></td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  {!s.versions.length && <Empty title="No engines recorded" text="Candidate identities appear after registration." />}
                </Section>
              )}

              {page === 'harness' && <HarnessPageView runId={s.id} snapshot={s} />}

              {page === 'evaluation' && (
                <div className="detail-grid">
                  <Section kicker="DEVELOPMENT GATE" title="Verified observations"
                    action={traceEngine && <button className={`button ${scope ? 'pressed' : ''}`} onClick={() => setScope(!scope)}><ListFilter size={15} />{scope ? `Engine ${short(traceEngine)}` : 'All engines'}</button>}>
                    <EventList key={run + ':quality'} rows={scoped(events.filter(e => e.source === 'verified' && e.kind !== 'reference'))} selected={selectedRow?.id} onClick={view} filter />
                  </Section>
                  <Inspector key={selectedRow?.id ?? p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav} onOpenCandidate={openCandidate} />
                </div>
              )}

              {page === 'research' && <ResearchPageView runId={s.id} onInspectCandidate={openCandidate} />}

              {page === 'resources' && <Resources s={s} />}

              <footer className="main-footer">
                <span><ShieldCheck size={15} /> Development evidence only · {s.mode === 'archive' ? 'Historical summary' : 'Explicit local run'}</span>
                <span>{sessions.size ? sessions.size + ' recorded sessions' : 'Sessions not available'} · <span className="mono">{short(s.source_ref)}</span></span>
              </footer>
              {s.notices.length > 0 && (
                <details className="notices"><summary>Evidence coverage · {s.notices.length} {s.notices.length === 1 ? 'note' : 'notes'}</summary><ul>{s.notices.map((n, i) => <li key={i}>{n}</li>)}</ul></details>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
