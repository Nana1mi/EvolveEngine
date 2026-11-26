import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, ArrowDownToLine, ArrowRight, AudioLines, BookOpen, ChartNoAxesCombined,
  ChevronDown, ChevronLeft, ChevronRight, CircleDot, CircuitBoard, Clock3, Cpu, Fingerprint, Flag, GitBranch, History, Layers3,
  LayoutDashboard, ListFilter, Menu, Radio, ShieldCheck, Waypoints, X, type LucideIcon } from 'lucide-react'
import { Curve, CurveLegend } from './chart'
import { axisList, metricList, xvalue, yvalue, xformat, has, short, num, compact, duration, yuan, timestamp, words, eventLabel,
  sourceName, type Axis, type Metric, type Point, type Snapshot, type RunInfo, type EventRow } from './model'
import './styles.css'

const nav = [
  {group:'OBSERVE', items:[['overview','Overview',LayoutDashboard],['trace','Live trace',Activity],['history','Update history',History],['experiments','Experiments',Layers3]]},
  {group:'LINEAGE & EVIDENCE',items:[['engine','Engine',GitBranch],['harness','Harness',CircuitBoard],['evaluation','Evaluation',ShieldCheck],['research','Research',BookOpen],['resources','Resources',Cpu]]},
] as const
type Page = typeof nav[number]['items'][number][0]
const pages = [...nav[0].items, ...nav[1].items]
const isPage = (s:string):s is Page => pages.some(([p])=>p===s)
const statusStyle = (s:string) => ['valid','built','delivered','observed','complete','pass'].includes(s) ? 'good' :
  ['violation','build_failed','runtime_failure','fail','upstream_transport','worker_transport','length','rejected','generation_truncated','launcher_failed','failed_or_interrupted'].includes(s) ? 'bad' :
  ['incomplete','paused','insufficient_evidence','uncertain','pending_review','reserved','started','no_terminal_receipt','usage_uncertain'].includes(s) ? 'warn' : 'muted'

function Badge({ children, tone='muted' }: { children:ReactNode; tone?:string }) {
  return <span className={`badge ${tone}`}><i/>{children}</span>
}
function Section({ title, kicker, children, className='', action }: { title:string; kicker?:string; children:ReactNode; className?:string; action?:ReactNode }) {
  return <section className={`card ${className}`}><div className="section-heading"><div>{kicker && <div className="eyebrow">{kicker}</div>}<h2>{title}</h2></div>{action}</div>{children}</section>
}
function Empty({ title, text }: {title:string; text:string}) {
  return <div className="empty"><Layers3 size={24}/><strong>{title}</strong><span>{text}</span></div>
}
function Drop<T extends string>({ value, onChange, options, label }: {value:T; onChange:(v:T)=>void; options:readonly {key:T;label:string; disabled?:boolean}[];label:string}) {
  return <label className="select"><span className="sr-only">{label}</span><select value={value} onChange={e=>onChange(e.target.value as T)}>{options.map(o=><option key={o.key} value={o.key} disabled={o.disabled}>{o.label}</option>)}</select><ChevronDown size={14}/></label>
}
function Stat({ icon: Icon, label, value, hint, accent }: {icon:LucideIcon;label:string;value:ReactNode;hint:string;accent:string}) {
  return <div className="card stat"><div className="stat-top"><span className={`tile ${accent}`}><Icon size={17}/></span><span>{label}</span></div><strong>{value}</strong><small>{hint}</small></div>
}

function useRuns() {
  const [runs,setRuns]=useState<RunInfo[]>([]), [run,setRun]=useState(''), [error,setError]=useState('')
  useEffect(()=>{
    const controller = new AbortController()
    fetch('/api/runs',{signal:controller.signal}).then(r=>{if(!r.ok)throw Error();return r.json()}).then(r=>{
      setRuns(r.runs);const id=new URLSearchParams(location.search).get('run');setRun(r.runs.find((v:RunInfo)=>v.id===id)?.id ?? r.runs[0]?.id ?? '')
    }).catch(()=>{if(!controller.signal.aborted)setError('Could not list runs. Check the local Observatory process.')})
    return ()=>controller.abort()
  },[])
  return {runs,run,error,choose:(id:string)=>{setRun(id);const u=new URL(location.href);u.searchParams.set('run',id);history.replaceState(null,'',u)}}
}
function useSnapshot(run:RunInfo|undefined) {
  const [snapshot,setSnapshot]=useState<Snapshot>(), [connection,setConnection]=useState('Connecting'), [error,setError]=useState('')
  const id=run?.id, mode=run?.mode
  useEffect(()=>{
    setSnapshot(undefined);setConnection(mode === 'archive' ? 'Archive' : 'Connecting');setError('')
    if(!id)return
    const controller=new AbortController()
    const base=`/api/runs/${encodeURIComponent(id)}`
    function update(v:Snapshot){ if(controller.signal.aborted||v.schema_version!==1||v.id!==id)return;setSnapshot(old=>old?.revision===v.revision?old:v);setError('') }
    if(mode==='archive'){
      fetch(base+'/snapshot',{signal:controller.signal}).then(r=>{if(!r.ok)throw Error();return r.json()}).then(update)
        .catch(()=>{if(!controller.signal.aborted)setError('Archive is unavailable.')})
      return ()=>controller.abort()
    }
    const stream=new EventSource(base+'/events')
    stream.onopen=()=>setConnection('Following receipts')
    stream.onerror=()=>setConnection('Reconnecting')
    stream.addEventListener('snapshot',e=>{try {update(JSON.parse(e.data));setConnection('Following receipts')}catch {setError('Snapshot could not be decoded.')}})
    return ()=>{controller.abort();stream.close()}
  },[id,mode])
  return {snapshot:snapshot?.id===id?snapshot:undefined,connection,error}
}

function EventList({ rows, selected, onClick, filter=false }: {rows:EventRow[]; selected?:string;onClick:(e:EventRow)=>void;filter?:boolean}) {
  const [query,setQuery]=useState(''),[group,setGroup]=useState('all'),[offset,setOffset]=useState(0)
  // The Journal, broker and Pi do not share one sequence or clock. Preserve
  // their separate recorded orders rather than treating concatenation as time.
  const stream = (e:EventRow) => e.kind==='model_request'?1:e.kind==='pi_tool'||e.kind==='session_harness_attached'?2:e.id.startsWith('annotations/')?3:0
  const sorted = useMemo(()=>[0,1,2,3].flatMap(k=>rows.filter(e=>stream(e)===k).reverse()),[rows])
  const mixed = new Set(rows.map(stream)).size>1
  const subset = sorted.filter(e => (group==='all' || (group==='failed' ? e.failure : group===e.source)) &&
    (!query || `${eventLabel(e)} ${e.status??''} ${e.continuation??''} ${e.cause??''} ${e.operation??''} ${e.engine??''} ${e.evidence??''}`.toLowerCase().includes(query.toLowerCase())))
  const size=filter?25:7, start=Math.min(offset,Math.max(0,Math.ceil(subset.length/size)-1))
  return <>
    {filter&&<div className="filters"><label className="search"><ListFilter size={15}/><input aria-label="Filter events" placeholder="Filter by event, status, or hash…" value={query} onChange={e=>{setQuery(e.target.value);setOffset(0)}}/></label>
      <Drop label="Event group" value={group} onChange={v=>{setGroup(v);setOffset(0)}} options={[{key:'all',label:'All sources'},{key:'verified',label:'Verified'},{key:'host',label:'Host'},{key:'agent',label:'Agent'},{key:'failed',label:'Failures only'}]}/></div>}
    {filter&&mixed&&<p className="side-note">Grouped by controller, broker, Pi and operator record order. No shared timestamp is inferred.</p>}
    {subset.length===0?<Empty title="No matching entries" text="Only recorded events are shown."/>:
      <div className="table-scroll"><table className="table"><thead><tr><th>Event</th><th>Engine / session</th><th>Status</th><th>Time</th></tr></thead><tbody>
      {subset.slice(start*size,(start+1)*size).map(e=><tr key={e.id} className={selected===e.id?'row-selected':''}>
        <td><button className="event-link" onClick={()=>onClick(e)}><span className={`event-mark ${e.failure?'danger':e.source}`}><CircleDot size={13}/></span><span><strong>{eventLabel(e)}</strong><small>{e.operation ? words(e.operation) : sourceName[e.source]}</small></span></button></td>
        <td><span className="mono">{e.engine ? (e.iteration==null? short(e.engine):`E${e.iteration} · ${short(e.engine)}`):e.session ? `S${e.session}`:'—'}</span></td>
        <td><Badge tone={e.failure?'bad':statusStyle(e.status??'')}>{words(e.finish==='length'?'generation_truncated':e.status??e.continuation??e.cause??e.source)}</Badge></td>
        <td className="mono">{has(e.wall_ms)?duration(e.wall_ms):has(e.at_ms)?timestamp(e.at_ms):'—'}</td>
      </tr>)}</tbody></table></div>}
    {filter && subset.length>size && <div className="pager"><span>{start*size+1}–{Math.min((start+1)*size,subset.length)} of {subset.length}</span><button className="icon-button" disabled={start===0} aria-label="Previous page" onClick={()=>setOffset(start-1)}><ChevronLeft size={17}/></button><button className="icon-button" disabled={(start+1)*size>=subset.length} aria-label="Next page" onClick={()=>setOffset(start+1)}><ChevronRight size={17}/></button></div>}
  </>
}
function PointTable({ data, selected, onClick }: {data:Snapshot;selected?:string;onClick:(id:string)=>void}) {
  if(!data.points.length)return <Empty title="No measured candidates" text="Failed or unmeasured attempts stay in the event log."/>
  return <div className="table-scroll"><table className="table"><thead><tr><th>Engine / measurement</th><th>Paired E0</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th><th>Wall</th><th>Evidence</th></tr></thead><tbody>
    {[...data.points].reverse().map(p=><tr key={p.id} className={selected===p.id?'row-selected':''}><td><button className="event-link" onClick={()=>onClick(p.id)}><span className="event-mark verified"><GitBranch size={14}/></span><span><strong>E{p.iteration} · {p.origin==='operator_replay'?'Independent replay':`Pair ${p.pair??'—'}`}</strong><small className="mono">{short(p.engine)}</small></span></button></td>
    <td className="mono ink">{has(p.aggregate)?`${num(p.aggregate,3)}×`:'—'}</td>{['W1','W2','W3','W4'].map(k=><td className="mono" key={k}>{has(p.ratios[k])?`${num(p.ratios[k],3)}×`:'—'}</td>)}
    <td className="mono">{duration(p.resources?.wall_ms)}</td><td><Badge tone={p.provisional?'warn':'good'}>{p.provisional?'Pending audit':'Dev receipt'}</Badge></td></tr>)}
  </tbody></table></div>
}
function KeyValues({ entries }: {entries:[ReactNode,ReactNode][]}) {
  return <dl className="key-values">{entries.map(([key,value],i)=><div key={i}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>
}
function Hash({value}: {value?:string|null}) {return <code className="hash" title={value??'Not recorded'}>{value??'Not recorded'}</code>}
function Inspector({ point, row, snapshot, onNavigate, dismiss }: {point?:Point;row?:EventRow;snapshot:Snapshot;onNavigate:(p:Page)=>void;dismiss?:()=>void}) {
  const [tab,setTab]=useState<'evidence'|'scenes'>('evidence')
  const quality = point && snapshot.events.filter(e=>e.engine===point.engine&&e.build===point.build&&e.operation==='develop_quality'&&e.cases).at(-1)
  return <Section className="inspector" kicker="EVIDENCE INSPECTOR" title={row ? eventLabel(row) : point ? `Engine E${point.iteration}` : 'Nothing selected'}
    action={dismiss && <button className="icon-button" aria-label="Close inspector" onClick={dismiss}><X size={17}/></button>}>
    {row ? <div className="inspector-content">
      <div className="inline-badges"><Badge tone={row.failure?'bad':'muted'}>{words(row.status??row.cause??row.kind)}</Badge><Badge tone={row.source==='verified'?'good':'muted'}>{sourceName[row.source]}</Badge></div>
      <KeyValues entries={[
        ['Engine',<Hash value={row.engine}/>],['Evidence',<Hash value={row.evidence}/>],['Session',row.session??'—'],['Operation',words(row.operation)],['Duration',duration(row.duration_ms)],
        ...(row.harness?[['Harness',<Hash value={row.harness}/>] as [string,ReactNode]]:[]),
        ...(row.previous_harness?[['Prior Harness',<Hash value={row.previous_harness}/>] as [string,ReactNode]]:[]),
        ...(row.continuation?[['Session boundary',words(row.continuation)] as [ReactNode,ReactNode]]:[]),
        ...(row.next_session?[['Successor session',row.next_session] as [ReactNode,ReactNode]]:[]),
        ...(row.cause?[['Stop / recovery cause',words(row.cause)] as [ReactNode,ReactNode]]:[]),
        ...(row.kind==='model_request'?[['Finish',words(row.finish)],['Settlement',words(row.usage_status)],['Input / Output',`${compact(row.input_tokens)} / ${compact(row.output_tokens)}`],['Reasoning',compact(row.reasoning_tokens)],['Cache hit / miss',`${compact(row.cache_hit_tokens)} / ${compact(row.cache_miss_tokens)}`],['Output reservation',compact(row.output_reserved)],['First delta',duration(row.first_delta_ms)]] as [ReactNode,ReactNode][]:[]),
        ...(row.cases?Object.entries(row.cases).map(([k,v])=>[words(k),<Badge tone={statusStyle(v)}>{words(v)}</Badge>] as [string,ReactNode]):[]),
      ]}/><p className="side-note">Only allowlisted receipt metadata. Raw inputs, outputs and local paths are not available here.</p>
    </div>:point?<div className="inspector-content">
      <div className="inline-badges"><Badge tone={point.provisional?'warn':'good'}>{point.provisional?'Weight audit pending':'Development receipt'}</Badge>{point.origin==='operator_replay'&&<Badge>Independent replay</Badge>}{snapshot.delivery?.engine===point.engine&&<Badge tone="violet">Agent selected engine</Badge>}</div>
      <div className="inspector-value"><strong>{num(point.aggregate,3)}<em>{has(point.aggregate)?'×':''}</em></strong><span>vs. paired E0 · {point.aggregate_kind==='weighted_score'?'registered weighted score':'equal-scene geomean'}</span></div>
      <div className="tabs"><button className={tab==='evidence'?'selected':''} onClick={()=>setTab('evidence')}>Evidence</button><button className={tab==='scenes'?'selected':''} onClick={()=>setTab('scenes')}>W1–W4</button></div>
      {tab==='evidence'?<KeyValues entries={[
        ['Engine',<Hash value={point.engine}/>],['Build',<Hash value={point.build}/>],['Measurement',<Hash value={point.id}/>],['Paired reference',<Hash value={point.reference}/>],['Quality',quality?<Badge tone={statusStyle(quality.status??'')}>{words(quality.status)}</Badge>:point.known_quality?'Development qualified':'Not recorded'],
        ...(point.quality?[['Quality receipt',<Hash value={point.quality}/>] as [string,ReactNode]]:[]),
        ['Wall time',duration(point.resources?.wall_ms)],['Session',point.session??'—'],
      ]}/>:<div className="scenes">{['W1','W2','W3','W4'].map(s=><div key={s}><div><b>{s}</b><small>{num(point.scenes[s]?.output_toks_per_s,2)} {has(point.scenes[s]?.output_toks_per_s)?'tok/s':''}</small></div><span>{has(point.ratios[s])?`${num(point.ratios[s],3)}×`:'—'}</span></div>)}</div>}
      <div className="inspector-links"><button onClick={()=>onNavigate('trace')}><Activity size={15}/>Trace <ArrowRight size={13}/></button><button onClick={()=>onNavigate('evaluation')}><ShieldCheck size={15}/>Evaluation <ArrowRight size={13}/></button><button onClick={()=>onNavigate('engine')}><GitBranch size={15}/>Lineage <ArrowRight size={13}/></button></div>
    </div>:<Empty title="Select a performance point" text="Details and linked evidence appear here."/>}
  </Section>
}

function Tracks({ rows, data, axis, onClick }: {rows:EventRow[];data:Point[];axis:Axis;onClick:(e:EventRow)=>void}) {
  const shown = ['Failed or uncertain','Harness / session','Pause / infrastructure'].map((label,i)=>({label,rows:rows.filter(e=>i===0?e.failure:i===1?e.kind.startsWith('harness_')||e.kind==='session_start'||e.kind==='segment_started':['pause','resume','infrastructure_change','infrastructure_fault'].includes(e.kind))}))
  const xs=data.map(p=>xvalue(p,axis)).filter(has)
  return <div className="tracks">{shown.map(({label,rows:r},i)=>{
    const time=(e:EventRow)=>axis==='effective_ms'?e.effective_ms:e.wall_ms
    const aligned=(axis==='wall_ms'||axis==='effective_ms') ? r.filter(e=>{const n=time(e);return has(n)&&xs.length&&n>=Math.min(...xs)&&n<=Math.max(...xs)}) : []
    const omitted=r.length-aligned.length
    return <div className="track" key={label}><span><i className={i?'violet-dot':'pink-dot'}/>{label}</span><div className="track-rail">
      {aligned.map(e=><button title={`${eventLabel(e)} · ${duration(time(e))}`} aria-label={eventLabel(e)} className={i?'violet-dot':'pink-dot'} key={e.id} style={{left:`${100*((time(e)??0)-Math.min(...xs))/Math.max(1,Math.max(...xs)-Math.min(...xs))}%`}} onClick={()=>onClick(e)}/>)}</div>
      <button className="count-link" disabled={!r.length} title={omitted ? `${omitted} without matching position on this axis` : label} onClick={()=>r.length&&onClick(r.at(-1)!)}>{r.length} {omitted?'· '+omitted+' unplaced':''}</button></div>
  })}</div>
}

function Resources({ s }: {s:Snapshot}) {
  const r=s.resources
  return <div className="two-panels">
    <Section kicker="RUN ACCOUNTING" title="Resource settlement"><KeyValues entries={[
      ['Wall time',duration(r.wall_ms)],['Effective exploration',duration(r.effective_ms)],['Charged node time',duration(r.node_ms)],['Settled node time',duration(r.settled_node_ms)],
      ['Unsettled node reservation',duration(r.reserved_node_ms)],['Settled model requests',num(r.requests,0)],['Unresolved requests',num(r.unresolved_requests,0)],['Start without result',num(r.requests_without_result,0)],
    ]}/><p className="side-note">Wall time and GPU node accounting are independent; they are not interchangeable.</p></Section>
    <Section kicker="BROKER ACCOUNTING" title="Model and cost"><KeyValues entries={[
      ['Input tokens',compact(r.input_tokens)],['Output incl. reasoning',compact(r.output_tokens)],['Reported reasoning',compact(r.reasoning_tokens)],
      ['Cache hit / miss',`${compact(r.cache_hit_tokens)} / ${compact(r.cache_miss_tokens)}`],['Conservative local ledger',yuan(r.ledger_rmb)],
      ['Ledger limit / outstanding',`${yuan(r.ledger_limit_rmb)} / ${yuan(r.ledger_outstanding_rmb)}`],['Usage fee estimate',`${yuan(r.fee_idle)} – ${yuan(r.fee_peak)}`],
    ]}/><p className="side-note">The local reservation ledger is not the provider invoice. Fee estimates are not verified bills; missing usage remains missing.</p></Section>
    <Section kicker="REGISTERED CEILINGS" title="Run limits"><KeyValues entries={[
      ['Input / output',`${compact(s.limits.input)} / ${compact(s.limits.output)}`],['Per request output',compact(s.limits.per_turn_output)],
      ['Model turns',num(s.limits.turns,0)],['Wall clock',duration(s.limits.wall_ms)]
    ]}/></Section>
    <Section kicker="CONTEXT EXPLORER / COUNTERS" title="Usage coverage"><KeyValues entries={[
      ['Projected request records',num(r.model_records,0)],['Settled with reported reasoning',num(r.usage_coverage && typeof r.usage_coverage==='object' ? r.usage_coverage.reasoning_tokens:null,0)],
      ['Settled with cache counters',num(r.usage_coverage && typeof r.usage_coverage==='object' ? r.usage_coverage.cache_hit_tokens:null,0)],['Raw context','Not exposed'],
    ]}/><p className="side-note">Individual request reservations, actual tokens, reasoning totals and finish status are in Live trace.</p></Section>
  </div>
}
function Lineage({ s, type, select, showOverview }: {s:Snapshot;type:'engine'|'harness';select:(e:EventRow)=>void;showOverview:()=>void}) {
  if(type==='harness')return <>
    <div className="two-panels"><Section kicker="ACTIVATION DECLARATION" title="Active Harness"><KeyValues entries={[
      ['Version',<Hash value={s.active_harness}/>],['Scope','Recorded activation only'],
    ]}/></Section><Section kicker="EXECUTION EVIDENCE" title="Actually loaded"><KeyValues entries={[
      ['Version',<Hash value={s.loaded_harness}/>],['Load proof',s.loaded_harness?'Recorded':'Not independently evidenced'],
    ]}/><p className="side-note">A staged or activated version is not proof that a later Pi session loaded or invoked it.</p></Section></div>
    <Section kicker="HARNESS AND HANDOFF" title="Lifecycle history"><EventList rows={s.events.filter(e=>e.harness||e.kind.startsWith('checkpoint')||e.kind.startsWith('session_'))} onClick={select} filter/></Section></>
  return <><Section kicker="ENGINE IDENTITY" title="Candidate lineage"><div className="table-scroll"><table className="table"><thead><tr><th>Version</th><th>Engine</th><th>Registered parent</th><th>Evidence</th></tr></thead><tbody>{s.versions.map(v=>{
    const p=s.points.findLast(p=>p.engine===v.id)
    return <tr key={v.id}><td><strong>E{v.iteration}</strong></td><td><Hash value={v.id}/></td><td><Hash value={v.parent}/></td><td>{p?<button className="text-button" onClick={()=>{select({id:p.id,kind:'comparison',source:'verified',point_id:p.id});showOverview()}}>Inspect measurement <ArrowRight size={14}/></button>:<Badge>Not measured</Badge>}</td></tr>
  })}</tbody></table></div>{!s.versions.length&&<Empty title="No engines recorded" text="Candidate identities appear after registration."/>}<p className="side-note">Hashes preserve version identity. Raw source and the source diff are not served by the read-only projection.</p></Section></>
}
function Export({s}: {s:Snapshot}) {
  function download(){
    const columns=['measurement','engine','build','paired_reference','pair','metric_method','development_ratio','W1_ratio','W2_ratio','W3_ratio','W4_ratio','wall_ms','effective_ms','node_ms','settled_input_at_dispatch','settled_output_at_dispatch','ledger_rmb_at_dispatch','provisional','origin']
    const rows=s.points.map(p=>[p.id,p.engine,p.build,p.reference,p.pair,p.aggregate_kind,p.aggregate,p.ratios.W1,p.ratios.W2,p.ratios.W3,p.ratios.W4,p.resources?.wall_ms,p.resources?.effective_ms,p.resources?.node_ms,p.resources?.input_tokens,p.resources?.output_tokens,p.resources?.ledger_rmb,!!p.provisional,p.origin??'controller'])
    const url=URL.createObjectURL(new Blob([[columns,...rows].map(x=>x.join(',')).join('\r\n')+'\r\n'],{type:'text/csv'}))
    const a=document.createElement('a');a.href=url;a.download=`${s.id}-development-measurements.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
  }
  return <button className="button" onClick={download} disabled={!s.points.length}><ArrowDownToLine size={15}/> <span>Export CSV</span></button>
}
const pageText: Partial<Record<Page,{title:string;subtitle:string}>> = {
  trace:{title:'Live trace',subtitle:'Follow model turns, Worker operations and checkpoint boundaries from recorded evidence.'},
  history:{title:'Update history',subtitle:'Connect measured candidates and failed attempts to their evidence without conflating the two.'},
  experiments:{title:'Experiments',subtitle:'Switch between explicitly registered run roots and historical archives.'},
  engine:{title:'Engine lineage',subtitle:'Trace registered engines, parent identities and development measurements.'},
  harness:{title:'Harness lineage',subtitle:'Keep declared activation distinct from attached and independently verified execution.'},
  evaluation:{title:'Evaluation',subtitle:'Development Judge results, case verdicts and evidence fingerprints. No holdout results are exposed.'},
  research:{title:'Research',subtitle:'Inspect recorded research actions and provenance fingerprints; queries and returned text are private.'},
  resources:{title:'Resources',subtitle:'Observe separate time, Node and model ledgers, not an inferred budget or bill.'},
}

function App() {
  const {runs,run,error:listError,choose}=useRuns()
  const info=runs.find(v=>v.id===run)
  const {snapshot:s,connection,error}=useSnapshot(info)
  const initial=location.hash.slice(1)
  const [page,setPage]=useState<Page>(isPage(initial)?initial:'overview')
  const [mobile,setMobile]=useState(false), [selected,setSelected]=useState<string>(), [selectedRow,setSelectedRow]=useState<EventRow>()
  const [scope,setScope]=useState(false), [replay,setReplay]=useState<number>(), [axis,setAxis]=useState<Axis>()
  const [metric,setMetric]=useState<Metric>('aggregate'),[scene,setScene]=useState('W1')
  useEffect(()=>{const handler=()=>{const p=location.hash.slice(1);if(isPage(p))setPage(p)};addEventListener('hashchange',handler);return()=>removeEventListener('hashchange',handler)},[])
  useEffect(()=>{setSelected(undefined);setSelectedRow(undefined);setAxis(undefined);setReplay(undefined);setMetric('aggregate');setScope(false)},[run])
  const navigate=(p:Page)=>{location.hash=p;setPage(p);setMobile(false);if(p==='trace'||p==='evaluation')setScope(!!selected||!!selectedRow?.engine)}
  const select=(id:string)=>{setSelected(id);setSelectedRow(undefined)}
  const view=(e:EventRow)=>{if(e.point_id){select(e.point_id);return}setSelectedRow(e)}
  const visible = s?.points.filter(p=>!replay||p.index<=replay)??[]
  const aggregateKind=visible.findLast(p=>has(p.aggregate))?.aggregate_kind
  const chartData=metric==='aggregate'?visible.filter(p=>p.aggregate_kind===aggregateKind):visible
  const latest=chartData.at(-1)
  const currentAxis=axis ?? (latest?.origin==='operator_replay' ? 'index' :
    visible.some(p=>has(xvalue(p,'effective_ms')))?'effective_ms':visible.some(p=>has(xvalue(p,'wall_ms')))?'wall_ms':'index')
  const p=visible.find(p=>p.id===selected)??visible.findLast(p=>s?.delivery?.point_id===p.id)??visible.at(-1)
  const m=metricList.find(v=>v.key===metric)!
  const best=chartData.filter(p=>has(yvalue(p,metric,scene))).reduce<Point|undefined>((acc,p)=>!acc||
    (m.high?Number(yvalue(p,metric,scene))>Number(yvalue(acc,metric,scene)):Number(yvalue(p,metric,scene))<Number(yvalue(acc,metric,scene)))?p:acc,undefined)
  const referenceIn=s?.events.find(e=>e.point_id===p?.id && ['comparison','operator_measurement'].includes(e.kind))
  const events = s?.events??[]
  const sessions=new Set(events.map(e=>e.session).filter(has))
  const summaryEvents=events.filter(e=>['comparison','operator_measurement','engine_registered','harness_submit','harness_activate','delivery','run_settlement','checkpoint_saved','pause','resume','infrastructure_change'].includes(e.kind)||e.failure)
  const traceEngine=selectedRow?.engine??p?.engine
  const scoped=(rows:EventRow[])=>scope&&traceEngine?rows.filter(e=>e.engine===traceEngine):rows
  const chartAxisOptions=axisList.map(a=>({...a,disabled:a.key!=='index'&&!visible.some(p=>has(xvalue(p,a.key)))}))
  const chartMetricOptions=metricList.map(m=>({...m,disabled:m.key!=='aggregate'&&!visible.some(p=>has(yvalue(p,m.key,scene)))}))
  const text=pageText[page]??{title:'Evolution overview',subtitle:'Track engine performance, harness changes, and the evidence behind each upgrade.'}
  const trail=page==='overview'?'Overview':pages.find(([key])=>key===page)?.[1]
  const panelNav=(p:Page)=>{navigate(p);if(traceEngine)setScope(true)}
  return <div className="app">
    {mobile&&<button className="mobile-backdrop" aria-label="Close navigation" onClick={()=>setMobile(false)}/>}
    <aside className={`sidebar ${mobile?'open':''}`}>
      <a className="brand" href="#overview" onClick={()=>navigate('overview')}><span className="logo"><Waypoints size={22} strokeWidth={2.1}/></span><span><strong>EvolveEngine</strong><small>RSI OBSERVATORY</small></span></a>
      <div className="space-card"><span className="space-icon"><CircuitBoard size={18}/></span><span><strong>Inference Lab</strong><small>Observability workspace</small></span></div>
      <nav aria-label="Main navigation">{nav.map(group=><div className="nav-group" key={group.group}><p>{group.group}</p>{group.items.map(([key,label,Icon])=>
        <button key={key} onClick={()=>{navigate(key);setScope(false)}} className={page===key?'nav-active':''} aria-current={page===key?'page':undefined}><Icon size={17}/><span>{label}</span>{page===key&&<i/>}</button>)}</div>)}</nav>
      <div className="sidebar-foot"><div className="foot-indicator"><ShieldCheck size={17}/></div><span><strong>Read-only mode</strong><small>Local evidence sidecar</small></span></div>
    </aside>
    <div className="body">
      <header className="topbar"><div className="breadcrumb"><button onClick={()=>setMobile(true)} className="icon-button mobile-menu" aria-label="Open navigation"><Menu size={19}/></button><Waypoints size={16}/><span>Workspace</span><ChevronRight size={14}/><b>{trail}</b></div>
        <div className="top-actions"><span className={`connection ${connection==='Following receipts'?'following':''}`}><Radio size={14}/>{connection}</span>
          {runs.length>0&&<Drop label="Choose run" value={run} onChange={choose} options={runs.map(v=>({key:v.id,label:v.label}))}/>}</div></header>
      <main id="main-content">
        <div className="hero"><div><div className="eyebrow hero-kicker"><i/> LEVEL 3 <span>/</span> ENGINEERING EXPLORATION</div><h1>{text.title}</h1><p>{text.subtitle}</p></div>{s&&<div className="hero-actions"><Badge tone={statusStyle(s.status)}>{s.status==='no_terminal_receipt'?'No settlement yet':s.status==='paused'&&s.stop_cause==='operator_pause_after_segment'?'Paused after segment':words(s.status)}</Badge><Export s={s}/></div>}</div>
        {!s?<section className="card"><Empty title={error||listError||(!info&&runs.length===0?'Waiting for run data':'Reading run evidence')} text={info?'The UI does not request or start experiments.':'Register a controller run or load the local archive if available.'}/></section>:
        <>
        {page==='overview'&&<>
          <div className="stats">
            <Stat icon={Cpu} label="Engine versions" value={s.versions.length} hint={s.delivery ? `Agent decision: ${short(s.delivery.engine)}` : 'No explicit delivery recorded'} accent="teal"/>
            <Stat icon={ChartNoAxesCombined} label={metric==='aggregate'?'Best development ratio':`Best ${m.label.toLowerCase()}`} value={best?`${num(yvalue(best,metric,scene),m.unit==='×'?3:2)}${m.unit==='×'?'×':` ${m.unit}`}`:'—'} hint={best?`E${best.iteration} · ${best.origin==='operator_replay'?'independent replay':`Pair ${best.pair??'—'}`} · ${best.provisional?'weight audit pending':'development only'}`:'No comparable measurement'} accent="violet"/>
            <Stat icon={Clock3} label="Wall time" value={duration(s.resources.wall_ms)} hint={`Node time: ${duration(s.resources.node_ms)}`} accent="blue"/>
            <Stat icon={AudioLines} label="Model requests" value={num(s.resources.requests,0)} hint={`${compact(s.resources.input_tokens)} input · ${compact(s.resources.output_tokens)} output`} accent="amber"/>
          </div>
          <div className="overview-grid">
            <Section title="Evolution performance curve" kicker="MEASURED HISTORY" className="curve-card" action={<div className="chart-actions"><Drop label="Chart metric" value={metric} onChange={setMetric} options={chartMetricOptions}/><Drop label="Horizontal axis" value={currentAxis} onChange={setAxis} options={chartAxisOptions}/></div>}>
              <div className="chart-subline"><span>{m.label} {metric==='aggregate'?`· ${aggregateKind==='weighted_score'?'Registered weighted score':'Equal-scene geomean'}`:''}</span>{metric!=='aggregate'&&<div className="scene-toggle" aria-label="Workload">{['W1','W2','W3','W4'].map(a=><button className={scene===a?'selected':''} key={a} aria-pressed={scene===a} onClick={()=>setScene(a)}>{a}</button>)}</div>}</div>
              <Curve points={chartData} selected={selectedRow?undefined:p?.id} metric={metric} axis={currentAxis} scene={scene} onSelect={select} delivered={s.delivery?.point_id} infrastructure={events.flatMap(e=>{const x=currentAxis==='effective_ms'?e.effective_ms:currentAxis==='wall_ms'?e.wall_ms:undefined;return e.kind==='infrastructure_change'&&has(x)?[{x,label:`Infra ${short(e.evidence)}`}]:[]})}/>
              <CurveLegend/>
              {chartData.some(p=>p.origin==='operator_replay'&&!has(xvalue(p,currentAxis)))&&<p className="side-note">Independent replay is available on the measurement-order axis. <button className="text-button" onClick={()=>setAxis('index')}>Show measurement order</button></p>}
              <Tracks rows={events} data={chartData} axis={currentAxis} onClick={view}/>
              {s.points.length>1&&<div className="replay"><History size={15}/><label htmlFor="replay">Measure replay</label><input id="replay" type="range" min={1} max={s.points.length} value={replay??s.points.length} onChange={e=>{const n=Number(e.target.value);setReplay(n===s.points.length?undefined:n);setSelectedRow(undefined)}}/><span>{visible.length}/{s.points.length}</span><button className="text-button" disabled={!replay} onClick={()=>setReplay(undefined)}>Latest</button></div>}
            </Section>
            <Inspector key={selectedRow?.id??p?.id??'none'} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav} dismiss={selectedRow?()=>setSelectedRow(undefined):undefined}/>
          </div>
          <div className="overview-grid lower"><Section kicker="EVENT JOURNAL" title="Recent updates" action={<button className="text-button" onClick={()=>navigate('history')}>View history <ArrowRight size={14}/></button>}><EventList rows={summaryEvents} selected={selectedRow?.id??referenceIn?.id} onClick={view}/></Section>
            <Section kicker="TRUST BOUNDARY" title="Evidence provenance"><div className="provenance"><div><ShieldCheck/><span><b>Verified receipts</b><small>Worker results and paired development measurements</small></span></div><div><Fingerprint/><span><b>Host observation</b><small>Journal, broker counters and transport milestones</small></span></div><div><Flag/><span><b>Agent decisions</b><small>Harness activation and nominated delivery</small></span></div></div><p className="side-note">Unmeasured failures stay in the event tracks. No holdout or production claim is implied.</p></Section></div>
        </>}
        {page==='trace'&&<div className="detail-grid"><Section kicker="SANITIZED TRACE" title="Recorded events" action={traceEngine&&<button className={`button ${scope?'pressed':''}`} onClick={()=>setScope(!scope)} aria-pressed={scope}><ListFilter size={15}/>{scope?`Engine ${short(traceEngine)}`:'All engines'}</button>}>
          <EventList key={run+':trace'} rows={scoped(events)} selected={selectedRow?.id} onClick={view} filter/></Section><Inspector key={selectedRow?.id??p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav}/></div>}
        {page==='history'&&<><div className="detail-grid"><Section kicker="PAIRED DEVELOPMENT MEASUREMENTS" title="Candidate updates"><PointTable data={s} selected={p?.id} onClick={select}/></Section><Inspector key={selectedRow?.id??p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav}/></div><Section kicker="ALL RECORDED OUTCOMES" title="Attempt and update log"><EventList key={run+':history'} rows={summaryEvents} onClick={view} selected={selectedRow?.id} filter/></Section></>}
        {page==='experiments'&&<Section kicker="RUN REGISTRY" title="Available experiments"><div className="run-list">{runs.map(r=><button key={r.id} onClick={()=>choose(r.id)} className={r.id===s.id?'run-selected':''}><Layers3 size={21}/><span><strong>{r.label}</strong><small>{r.mode==='archive'?'Local archive summary':'Explicit controller evidence root'}</small></span><Badge tone={r.id===s.id?'good':'muted'}>{r.id===s.id?'Selected':r.mode}</Badge><ArrowRight size={16}/></button>)}</div></Section>}
        {(page==='engine'||page==='harness')&&<><Lineage s={s} type={page} select={view} showOverview={()=>navigate('overview')}/>{selectedRow&&<Inspector key={selectedRow.id} row={selectedRow} snapshot={s} onNavigate={panelNav} dismiss={()=>setSelectedRow(undefined)}/>}</>}
        {page==='evaluation'&&<div className="detail-grid"><Section kicker="DEVELOPMENT GATE" title="Verified observations" action={traceEngine&&<button className={`button ${scope?'pressed':''}`} onClick={()=>setScope(!scope)} aria-pressed={scope}><ListFilter size={15}/>{scope?`Engine ${short(traceEngine)}`:'All engines'}</button>}><EventList key={run+':quality'} rows={scoped(events.filter(e=>e.source==='verified'&&e.kind!=='reference'))} selected={selectedRow?.id} onClick={view} filter/></Section><Inspector key={selectedRow?.id??p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav}/></div>}
        {page==='research'&&<div className="detail-grid"><Section kicker="ALLOWLISTED REFERENCES" title="Research activity"><EventList key={run+':research'} rows={events.filter(e=>e.kind==='research')} onClick={view} selected={selectedRow?.id} filter/></Section><Inspector key={selectedRow?.id??p?.id} point={p} row={selectedRow} snapshot={s} onNavigate={panelNav}/></div>}
        {page==='resources'&&<Resources s={s}/>}
        <footer className="main-footer"><span><ShieldCheck size={15}/> Development evidence only · {s.mode==='archive'?'Historical summary':'Explicit local run'}</span><span>{sessions.size?sessions.size+' recorded sessions':'Sessions not available'} · <span className="mono">{short(s.source_ref)}</span></span></footer>
        {s.notices.length>0&&<details className="notices"><summary>Evidence coverage · {s.notices.length} {s.notices.length===1?'note':'notes'}</summary><ul>{s.notices.map((n,i)=><li key={i}>{n}</li>)}</ul></details>}
        </>}
      </main>
    </div>
  </div>
}

createRoot(document.getElementById('root')!).render(<App/>)
