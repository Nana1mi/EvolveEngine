export type Num = number | null | undefined
export type RunInfo = { id: string; label: string; mode: 'archive' | 'controller' }
export type Source = 'verified' | 'host' | 'agent'
export type EventRow = {
  id: string; kind: string; source: Source; engine?: string; parent?: string; evidence?: string; build?: string;
  harness?: string; previous_harness?: string; continuation?: string; next_session?: number; operation?: string; status?: string; cause?: string; failure?: boolean; sequence?: number;
  point_id?: string; point_index?: number; iteration?: number; session?: number; attempt?: number; pair?: Num;
  wall_ms?: Num; effective_ms?: Num; at_ms?: Num; duration_ms?: Num; finish?: string; usage_status?: string; request?: string;
  input_tokens?: Num; output_tokens?: Num; reasoning_tokens?: Num; cache_hit_tokens?: Num; cache_miss_tokens?: Num;
  input_reserved?: Num; output_reserved?: Num; ledger_rmb?: Num; first_delta_ms?: Num; last_delta_ms?: Num;
  bytes?: Num; cost_rmb?: Num; cases?: Record<string, string>;
}
export type Point = {
  id: string; engine: string; iteration: number; index: number; session?: Num; build?: string; pair?: Num;
  reference?: string; aggregate?: Num; aggregate_kind: 'weighted_score' | 'equal_scene_geomean';
  ratios: Record<string, number>; scenes: Record<string, Record<string, number>>;
  reference_scenes: Record<string, Record<string, number>>; resources?: Record<string, Num>;
  provisional?: boolean; known_quality: boolean; quality?: string; origin?: 'operator_replay';
}
export type Snapshot = RunInfo & {
  revision: string; schema_version: number; status: string; stop_cause?: string; source_ref?: string;
  points: Point[]; events: EventRow[]; versions: { id: string; iteration: number; parent?: string }[];
  active_harness?: string; loaded_harness?: string; baseline?: string;
  delivery?: { engine: string; evidence?: string; point_id?: string; disposition: string };
  resources: Record<string, Num | boolean | Record<string, number>>; limits: Record<string, number>; notices: string[];
}
export const has = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)
export const short = (id?: string | null) => id ? id.slice(0, 10) : '—'
export const words = (text?: string | null) => text?.replaceAll('_', ' ') ?? 'Not recorded'
export const num = (n: unknown, digits = 1) => has(n) ? n.toLocaleString('en-US', { maximumFractionDigits: digits }) : '—'
export function compact(n: unknown) {
  if (!has(n)) return '—';
  return n >= 1e6 ? `${num(n / 1e6, 2)}M` : n >= 1e3 ? `${num(n / 1e3, 1)}k` : num(n, 0)
}
export const duration = (n: unknown) => !has(n) ? '—' : n >= 3600000 ? `${Math.floor(n / 3600000)}h ${Math.floor(n / 60000) % 60}m` : n >= 60000 ? `${num(n / 60000, 1)}m` : `${num(n / 1000, 2)}s`
export const yuan = (n: unknown) => has(n) ? `¥${n.toFixed(2)}` : '—'
export const timestamp = (n: number) => new Intl.DateTimeFormat('en-GB', {timeZone:'Asia/Hong_Kong', hour:'2-digit', minute:'2-digit', second:'2-digit', hourCycle:'h23'}).format(n)
export const sourceName: Record<Source,string> = { verified:'Verified receipt', host:'Host observed', agent:'Agent decision' }
export const eventNames: Record<string,string> = {
  comparison:'Development measurement', operator_measurement:'Independent replay', worker_reserved:'Worker reservation', worker_settled:'Worker result',
  worker_uncertain:'Uncertain Worker settlement', run_settlement:'Run settlement', delivery:'Agent nomination',
  reference:'Development reference', engine_registered:'Engine registered', session_start:'Session boundary',
  session_end:'Session ended', segment_started:'Agent session started', segment_settled:'Agent session settled',
  model_request:'Model request', pi_tool:'Pi tool invocation', research:'Research action', harness_submit:'Harness staged',
  harness_activate:'Harness activation', session_harness_attached:'Harness attached at launch',
  checkpoint_saved:'Checkpoint saved', checkpoint_imported:'Checkpoint imported', checkpoint_import_refused:'Checkpoint import refused',
}
export const eventLabel = (e: EventRow) => eventNames[e.kind] ?? words(e.kind)
export const metricList = [
  {key:'aggregate', label:'Development ratio', unit:'×', high:true},
  {key:'ratio', label:'Paired scene ratio', unit:'×', high:true},
  {key:'output_toks_per_s', label:'Output throughput', unit:'tok/s', high:true},
  {key:'tpot_ms', label:'Time per output token', unit:'ms', high:false},
  {key:'ttft_ms', label:'Time to first token', unit:'ms', high:false},
  {key:'e2e_ms', label:'End-to-end latency', unit:'ms', high:false},
] as const
export type Metric = typeof metricList[number]['key']
export const axisList = [
  {key:'effective_ms', label:'Effective exploration time'}, {key:'wall_ms', label:'Wall time'}, {key:'index', label:'Measurement order'}, {key:'iteration', label:'Engine iteration'},
  {key:'session', label:'Session'}, {key:'input_tokens', label:'Input tokens'}, {key:'output_tokens', label:'Output tokens'},
  {key:'ledger_rmb', label:'API ledger'}, {key:'node_ms', label:'Node time'},
] as const
export type Axis = typeof axisList[number]['key']
export const xvalue = (p: Point, a: Axis): Num => a === 'index' || a === 'iteration' || a === 'session' ? p[a] : p.resources?.[a]
export const yvalue = (p:Point, m:Metric, scene:string): Num => m === 'aggregate' ? p.aggregate : m === 'ratio' ? p.ratios[scene] : p.scenes[scene]?.[m]
export const reference = (p:Point, m:Metric, scene:string): Num => m === 'aggregate' || m === 'ratio' ? 1 : p.reference_scenes[scene]?.[m]
export const xformat = (n:unknown, a:Axis) => a.endsWith('_ms') ? duration(n) : a.endsWith('_tokens') ? compact(n) : a === 'ledger_rmb' ? yuan(n) : num(n, 0)
export const yformat = (n:unknown, unit:string) => has(n) ? `${num(n, unit === '×' ? 3 : 2)}${unit === '×' ? '' : ' '}${unit}` : '—'
