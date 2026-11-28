export type Num = number | null | undefined
export type RunInfo = { id: string; label: string; mode: 'archive' | 'controller' }
export type Source = 'verified' | 'host' | 'agent'

export type QualityDetail = {
  trajectory_screen?: { status?: string; maxima?: Record<string, number>; failed?: string[]; limits?: Record<string, number> }
  sampling_screen?: { status?: string; max_tv?: number; threshold?: number; p_value?: number }
  verdict_only_cases?: string[]
}

export type EventRow = {
  id: string; kind: string; source: Source; engine?: string; parent?: string; evidence?: string; build?: string;
  harness?: string; previous_harness?: string; continuation?: string; next_session?: number; operation?: string; status?: string; cause?: string; failure?: boolean; sequence?: number;
  point_id?: string; point_index?: number; iteration?: number; session?: number; attempt?: number; pair?: Num;
  wall_ms?: Num; effective_ms?: Num; at_ms?: Num; duration_ms?: Num; finish?: string; usage_status?: string; request?: string;
  input_tokens?: Num; output_tokens?: Num; reasoning_tokens?: Num; cache_hit_tokens?: Num; cache_miss_tokens?: Num;
  input_reserved?: Num; output_reserved?: Num; ledger_rmb?: Num; first_delta_ms?: Num; last_delta_ms?: Num;
  bytes?: Num; cost_rmb?: Num; cases?: Record<string, string>; quality_detail?: QualityDetail;
}

export type ScoringPolicy = {
  version: string
  weight_percent: Record<string, number>
  scene_floors: Record<string, number>
  guardrail_floors: Record<string, number>
  scene_precision_targets: Record<string, number>
  minimum_pairs: number
  sla_targets_tps: number[]
}

export type BuildSummary = {
  engine: string
  iteration?: number | null
  build?: string | null
  pairs: number
  minimum_pairs: number
  ci_available: boolean
  status: string
  weighted_score?: Num
  scene_ratios: Record<string, number>
  intervals: Record<string, [number, number]>
  guardrail_ratios?: Record<string, number> | null
  guardrail_intervals?: Record<string, [number, number]>
  below_floor: string[]
  uncertain_scenes: string[]
}

export type Point = {
  id: string; engine: string; iteration: number; index: number; session?: Num; build?: string; pair?: Num;
  reference?: string; aggregate?: Num; aggregate_kind: 'weighted_score' | 'equal_scene_geomean';
  ratios: Record<string, number>; scenes: Record<string, Record<string, number>>;
  reference_scenes: Record<string, Record<string, number>>; resources?: Record<string, Num>;
  guardrail_ratios?: Record<string, number>;
  guardrails?: Record<string, Record<string, number>>;
  reference_guardrails?: Record<string, Record<string, number>>;
  unfinished_count?: Num;
  provisional?: boolean; known_quality: boolean; quality?: string; origin?: 'operator_replay';
}

export type Snapshot = RunInfo & {
  revision: string; schema_version: number; status: string; stop_cause?: string; source_ref?: string;
  scoring?: ScoringPolicy | null;
  has_trace_content?: boolean;
  build_summaries?: BuildSummary[];
  points: Point[]; events: EventRow[]; versions: { id: string; iteration: number; parent?: string }[];
  active_harness?: string; loaded_harness?: string; baseline?: string;
  delivery?: { engine: string; evidence?: string; point_id?: string; disposition: string };
  resources: Record<string, Num | boolean | Record<string, number>>; limits: Record<string, number>; notices: string[];
}

export type TraceToolCall = {
  tool?: string; rpc_operation?: string | null; tool_call_id?: string | null; call_sha256?: string | null;
  path?: string | null; paths?: string[] | null; engine?: string | null; parent?: string | null;
  build?: string | null; evidence?: string | null;
  start_utc_ns?: Num; end_utc_ns?: Num; elapsed_ms?: Num; duration_ms?: Num;
  is_error?: boolean; args_preview?: string; args_blob?: string | null; args_inline?: any;
  result_preview?: string; result_blob?: string | null; result_inline?: any;
  rpc_status?: Num; op_status?: string | null;
}

export type TraceTurn = {
  id: string; segment: number; attempt?: number | null; utc_ns?: Num; elapsed_ms?: Num;
  correlation?: string;
  model?: {
    started_utc?: string; ended_utc?: string; wall_ms?: Num; finish?: string; result?: string; usage_status?: string;
    input_tokens?: Num; output_tokens?: Num; reasoning_tokens?: Num; cache_hit_tokens?: Num; cache_miss_tokens?: Num;
    first_delta_ms?: Num; last_delta_ms?: Num; output_span_ms?: Num; request_message_count?: Num;
    reasoning_chars?: number; content_chars?: number; content_preview?: string; rejection?: any;
  } | null;
  response_blob?: string | null;
  response_inline?: { reasoning?: string; content?: string; tool_calls?: any[]; finish?: string } | null;
  tools: TraceToolCall[];
  worker_events: { kind: string; operation?: string; engine?: string; status?: string; evidence?: string; build?: string; node_ms?: Num; pair?: Num; elapsed_ms?: Num }[];
  engines: string[];
  has_error: boolean;
  is_candidate_turn: boolean;
}

export type TraceResponse = {
  has_trace_content: boolean; total_turns: number; matched_turns: number; offset: number; limit: number;
  segments: number[]; tools: string[]; turns: TraceTurn[];
}

export type FileDiff = { path: string; status: 'added' | 'modified' | 'deleted'; added_lines: number; removed_lines: number; truncated: boolean; diff: string }
export type KeyChange = { key: string; before: unknown; after: unknown }
export type DiffBundle = { files: FileDiff[]; runtime_diff?: KeyChange[] | null; build_args_diff: KeyChange[] }

export type CandidateDetail = {
  engine: string; iteration?: number | null; parent?: string | null; baseline?: string | null;
  scoring?: ScoringPolicy | null; has_trace_content: boolean;
  diff_vs_parent: DiffBundle; diff_vs_e0: DiffBundle;
  builds: EventRow[]; qualities: EventRow[]; points: Point[]; build_summaries: BuildSummary[];
  conversation: TraceTurn[];
}

export type WaterfallSpan = {
  lane: string; label: string; start_ms: number; end_ms: number; duration_ms: number;
  segment?: number; status?: string; engine?: string; evidence?: string;
}
export type WaterfallData = {
  total_wall_ms: number;
  breakdown: Record<string, number>;
  lanes: string[];
  spans: WaterfallSpan[];
}

export type ResearchData = {
  available: boolean;
  searches: { call: number | string; utc?: string; query?: string; count?: number; status?: number; elapsed_ms?: number; results: { title?: string; url?: string; snippet?: string; date?: string; site_name?: string }[] }[];
  fetches: { call: number | string; utc?: string; url?: string; title?: string; status?: number; elapsed_ms?: number; content_bytes?: number }[];
  pages: { id: string; url?: string; title?: string; bytes: number; preview: string }[];
  reuses: { receipt: string; kind?: string; paths: string[]; url?: string; revision?: string; license?: string; identity?: string; matched_engines: string[] }[];
}

export type HarnessData = {
  active_harness?: string | null; loaded_harness?: string | null;
  versions: {
    id: string; files: string[]; load: Record<string, string[]>;
    evidence: { submitted: boolean; activated: boolean; attached_sessions: number[]; prompt_loaded_turns: number; invoked_tools: number };
  }[];
  diff_left?: string | null; diff_right?: string | null; diff: FileDiff[];
}

export type ContextPoint = {
  index: number; segment?: number | null; attempt?: number | null; at_ms?: Num;
  input_tokens?: Num; cache_hit_tokens?: Num; cache_miss_tokens?: Num; cache_hit_rate?: Num;
  output_tokens?: Num; reasoning_tokens?: Num; wall_ms?: Num; first_delta_ms?: Num; finish?: string | null;
}

export const has = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)
export const short = (id?: string | null) => id ? id.slice(0, 10) : '—'
export const words = (text?: string | null) => text?.replaceAll('_', ' ') ?? 'Not recorded'
export const num = (n: unknown, digits = 1) => has(n) ? n.toLocaleString('en-US', { maximumFractionDigits: digits }) : '—'
export const pct = (n: unknown, digits = 1) => has(n) ? `${(n * 100).toFixed(digits)}%` : '—'
export function compact(n: unknown) {
  if (!has(n)) return '—'
  return n >= 1e6 ? `${num(n / 1e6, 2)}M` : n >= 1e3 ? `${num(n / 1e3, 1)}k` : num(n, 0)
}
export const duration = (n: unknown) => !has(n) ? '—' : n >= 3600000 ? `${Math.floor(n / 3600000)}h ${Math.floor(n / 60000) % 60}m` : n >= 60000 ? `${num(n / 60000, 1)}m` : n >= 1000 ? `${num(n / 1000, 2)}s` : `${num(n, 0)}ms`
export const yuan = (n: unknown) => has(n) ? `¥${n.toFixed(2)}` : '—'
export const timestamp = (n: number) => new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Hong_Kong', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }).format(n)
export const sourceName: Record<Source, string> = { verified: 'Verified receipt', host: 'Host observed', agent: 'Agent decision' }

export const eventNames: Record<string, string> = {
  comparison: 'Development measurement', operator_measurement: 'Independent replay', worker_reserved: 'Worker reservation', worker_settled: 'Worker result',
  worker_uncertain: 'Uncertain Worker settlement', run_settlement: 'Run settlement', delivery: 'Agent nomination',
  reference: 'Development reference', engine_registered: 'Engine registered', session_start: 'Session boundary',
  session_end: 'Session ended', segment_started: 'Agent session started', segment_settled: 'Agent session settled',
  model_request: 'Model request', pi_tool: 'Pi tool invocation', research: 'Research action', harness_submit: 'Harness staged',
  harness_activate: 'Harness activation', session_harness_attached: 'Harness attached at launch',
  checkpoint_saved: 'Checkpoint saved', checkpoint_imported: 'Checkpoint imported', checkpoint_import_refused: 'Checkpoint import refused',
}
export const eventLabel = (e: EventRow) => eventNames[e.kind] ?? words(e.kind)

export const isWorkBuddyV2 = (s?: Snapshot | null): boolean =>
  s?.scoring?.version === 'workbuddy-agent-serving-v2' ||
  Boolean(s?.points.some(p => 'W5' in p.ratios || 'W4_hot' in p.ratios || 'W5' in p.scenes))

export const ratioKeysFor = (s?: Snapshot | null): string[] => {
  if (s?.scoring?.weight_percent && Object.keys(s.scoring.weight_percent).length > 0) {
    return Object.keys(s.scoring.weight_percent)
  }
  return isWorkBuddyV2(s) ? ['W1', 'W2', 'W3', 'W4_hot', 'W4_cold', 'W5'] : ['W1', 'W2', 'W3', 'W4']
}

export const sceneKeysFor = (s?: Snapshot | null): string[] =>
  isWorkBuddyV2(s) ? ['W1', 'W2', 'W3', 'W4', 'W5'] : ['W1', 'W2', 'W3', 'W4']

export const guardrailLabels: Record<string, string> = {
  w4_cold_e2e: 'W4 冷请求完成时间',
  g64k_ttft: '64K 首 token 延迟',
  g64k_decode: '64K 解码速度',
  p95_itl: 'P95 ITL 流畅度',
}

export const metricList = [
  { key: 'aggregate', label: '综合比值 (Score)', unit: '×', high: true },
  { key: 'ratio', label: '分场景比值 (vs E0)', unit: '×', high: true },
  { key: 'dev_score', label: '场景主分 (dev_score)', unit: '', high: true },
  { key: 'output_toks_per_s', label: '输出吞吐', unit: 'tok/s', high: true },
  { key: 'tpot_ms', label: '单 token 解码耗时 (TPOT)', unit: 'ms', high: false },
  { key: 'ttft_ms', label: '首 token 延迟 (TTFT)', unit: 'ms', high: false },
  { key: 'p95_itl_ms', label: 'P95 ITL', unit: 'ms', high: false },
  { key: 'share_ge_40_tps', label: '≥40 tok/s 达标率', unit: '%', high: true },
  { key: 'share_ge_50_tps', label: '≥50 tok/s 达标率', unit: '%', high: true },
  { key: 'e2e_ms', label: '端到端延迟', unit: 'ms', high: false },
] as const
export type Metric = typeof metricList[number]['key']

export const axisList = [
  { key: 'effective_ms', label: 'Effective exploration time' }, { key: 'wall_ms', label: 'Wall time' },
  { key: 'index', label: 'Measurement order' }, { key: 'iteration', label: 'Engine iteration' },
  { key: 'session', label: 'Session' }, { key: 'input_tokens', label: 'Input tokens' },
  { key: 'output_tokens', label: 'Output tokens' }, { key: 'ledger_rmb', label: 'API ledger' },
  { key: 'node_ms', label: 'Node time' },
] as const
export type Axis = typeof axisList[number]['key']

export const xvalue = (p: Point, a: Axis): Num => a === 'index' || a === 'iteration' || a === 'session' ? p[a] : p.resources?.[a]
export const yvalue = (p: Point, m: Metric, scene: string): Num => {
  if (m === 'aggregate') return p.aggregate
  if (m === 'ratio') return p.ratios[scene]
  const raw = p.scenes[scene]?.[m]
  return (m === 'share_ge_40_tps' || m === 'share_ge_50_tps') && has(raw) ? raw * 100 : raw
}
export const reference = (p: Point, m: Metric, scene: string): Num => {
  if (m === 'aggregate' || m === 'ratio') return 1
  const raw = p.reference_scenes[scene]?.[m]
  return (m === 'share_ge_40_tps' || m === 'share_ge_50_tps') && has(raw) ? raw * 100 : raw
}
export const xformat = (n: unknown, a: Axis) => a.endsWith('_ms') ? duration(n) : a.endsWith('_tokens') ? compact(n) : a === 'ledger_rmb' ? yuan(n) : num(n, 0)
export const yformat = (n: unknown, unit: string) => has(n) ? `${num(n, unit === '×' ? 3 : unit === '%' ? 1 : 2)}${unit === '×' || unit === '%' ? unit : unit ? ` ${unit}` : ''}` : '—'
