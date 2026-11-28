import { useEffect, useRef } from 'react'
import { init, use, type EChartsCoreOption } from 'echarts/core'
import { BarChart, LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import { Activity, ChartNoAxesCombined } from 'lucide-react'
import {
  axisList, compact, has, metricList, reference, short, xformat, xvalue, yformat, yvalue,
  type Axis, type ContextPoint, type Metric, type Point,
} from './model'

use([BarChart, LineChart, ScatterChart, GridComponent, LegendComponent, TooltipComponent, MarkLineComponent, SVGRenderer])

export function Curve({ points, selected, metric, axis, scene, onSelect, delivered, infrastructure }: {
  points: Point[]; selected?: string; metric: Metric; axis: Axis; scene: string;
  onSelect: (id: string) => void; delivered?: string; infrastructure: { x: number; label: string }[];
}) {
  const container = useRef<HTMLDivElement>(null)
  const selection = useRef(onSelect)
  selection.current = onSelect
  const config = metricList.find(m => m.key === metric)!
  const data = points.flatMap(p => {
    const x = xvalue(p, axis), y = yvalue(p, metric, scene)
    return has(x) && has(y) ? [{ x, y, p }] : []
  }).sort((a, b) => a.x - b.x || a.p.index - b.p.index)
  const count = data.length
  const boundedInfra = count ? infrastructure.filter(m => m.x >= data[0].x && m.x <= data.at(-1)!.x) : []
  const infraRevision = JSON.stringify(boundedInfra)
  const revision = JSON.stringify(data.map(({ x, y, p }) => [x, y, p.id, p.iteration, p.index, p.pair, p.provisional, reference(p, metric, scene)]))

  useEffect(() => {
    if (!container.current || count === 0) return
    const chart = init(container.current, undefined, { renderer: 'svg' })
    const marks = data.map(({ x, y, p }) => ({
      name: p.id,
      value: [x, y],
      symbolSize: p.id === selected ? 16 : p.id === delivered ? 13 : 9,
      itemStyle: {
        color: p.id === selected ? '#58ecd2' : p.id === delivered ? '#cfb9ff' : '#5ba8c7',
        borderColor: p.id === selected ? '#d1fff8' : '#101c2b',
        borderWidth: p.id === selected ? 2 : 1,
      },
      label: p.id === selected ? {
        show: true, position: 'top', formatter: `E${p.iteration}`, color: '#defdf7', fontSize: 11,
        padding: [4, 8], backgroundColor: '#193b42', borderRadius: 5, distance: 9,
      } : undefined,
    }))
    let best = config.high ? -Infinity : Infinity
    const bestLine = data.map(({ x, y }) => [x, best = config.high ? Math.max(best, y) : Math.min(best, y)])
    const base = data.flatMap(({ x, p }) => { const r = reference(p, metric, scene); return has(r) ? [[x, r]] : [] })
    const same = count === 1 || data.at(-1)?.x === data[0].x
    const opt: EChartsCoreOption = {
      animation: false,
      grid: { left: 66, right: 24, top: 27, bottom: 54 },
      xAxis: {
        type: 'value', name: axisList.find(a => a.key === axis)!.label, nameLocation: 'center', nameGap: 36,
        nameTextStyle: { color: '#92a2b8', fontSize: 11 },
        min: ({ min }: { min: number }) => same ? Math.max(0, min - Math.max(1, min * .1)) : Math.max(0, min - (data.at(-1)!.x - min) * .085),
        max: ({ max }: { max: number }) => same ? max + Math.max(1, max * .1) : max + (max - data[0].x) * .085,
        minInterval: ['index', 'iteration', 'session'].includes(axis) ? 1 : undefined,
        axisLabel: { color: '#8d9ab0', formatter: (v: number) => xformat(v, axis), hideOverlap: true },
        axisLine: { lineStyle: { color: '#28364a' } },
        axisTick: { show: false },
        splitLine: { show: true, lineStyle: { color: '#192438' } },
      },
      yAxis: {
        type: 'value', scale: true, splitNumber: 5,
        axisLabel: { color: '#8d9ab0', formatter: (v: number) => yformat(v, config.unit) },
        splitLine: { lineStyle: { color: '#233045', type: 'dashed' } },
        axisTick: { show: false }, axisLine: { show: false },
      },
      tooltip: {
        trigger: 'item', confine: true, appendTo: container.current,
        backgroundColor: '#0e1928', borderColor: '#314962', textStyle: { color: '#ecf4ff', fontSize: 12 },
        formatter: (params: { name: string; value: number[] }) => {
          const p = data.find(d => d.p.id === params.name)?.p
          if (!p) return ''
          return `<b>Engine E${p.iteration}</b> · ${short(p.engine)}<br>${yformat(params.value[1], config.unit)} · ${xformat(params.value[0], axis)}<br>Paired measurement #${p.pair ?? '—'} · Click to inspect diff & trace`
        },
      },
      series: [
        {
          type: 'line', name: 'Paired E0',
          data: same && base.length ? [[Math.max(0, base[0][0] - Math.max(1, base[0][0] * .1)), base[0][1]], [base[0][0] + Math.max(1, base[0][0] * .1), base[0][1]]] : base,
          symbol: 'none', silent: true, z: 1, lineStyle: { color: '#b1a1e3', width: 1, type: 'dashed' },
        },
        {
          type: 'line', name: 'Observed best', data: bestLine, silent: true, step: 'end', symbol: 'none',
          lineStyle: { color: '#55dcca', width: 2 },
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(64,216,197,.16)' }, { offset: 1, color: 'rgba(64,216,197,0)' }] } },
        },
        {
          type: 'scatter', name: 'Measured', data: marks, z: 5, emphasis: { scale: 1.28 },
          markLine: {
            silent: true, symbol: 'none', lineStyle: { color: '#c6a6ff', type: 'dashed' },
            label: { show: true, position: 'insideEndTop', color: '#dccbff', formatter: '{b}' },
            data: boundedInfra.map(m => ({ name: m.label, xAxis: m.x })),
          },
        },
      ],
    }
    chart.setOption(opt)
    chart.on('click', 'series.scatter', (e) => { if (e.name) selection.current(e.name) })
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(container.current)
    return () => { resize.disconnect(); chart.dispose() }
  }, [revision, count, metric, axis, scene, selected, delivered, infraRevision])

  if (!count) return <div className="curve-empty"><ChartNoAxesCombined size={34} /><strong>No comparable measurements</strong><span>Unavailable values are not plotted as zero.</span></div>
  return <div ref={container} className="curve" role="img" aria-label={`Evolution performance graph: ${config.label}. ${count} plotted measurements.`} />
}

export function ContextGrowthChart({ points }: { points: ContextPoint[] }) {
  const container = useRef<HTMLDivElement>(null)
  const revision = JSON.stringify(points.map(p => [p.index, p.input_tokens, p.cache_hit_tokens, p.cache_miss_tokens, p.reasoning_tokens]))

  useEffect(() => {
    if (!container.current || !points.length) return
    const chart = init(container.current, undefined, { renderer: 'svg' })
    const xs = points.map(p => `#${p.index}`)
    const hits = points.map(p => p.cache_hit_tokens ?? 0)
    const misses = points.map(p => p.cache_miss_tokens ?? (has(p.input_tokens) && !has(p.cache_hit_tokens) ? p.input_tokens : 0))
    const reasoning = points.map(p => p.reasoning_tokens ?? 0)
    const hitRate = points.map(p => has(p.cache_hit_rate) ? Number((p.cache_hit_rate * 100).toFixed(1)) : null)

    const opt: EChartsCoreOption = {
      animation: false,
      grid: { left: 60, right: 52, top: 38, bottom: 42 },
      legend: { top: 4, textStyle: { color: '#a5b7cb', fontSize: 11 } },
      tooltip: {
        trigger: 'axis', confine: true, backgroundColor: '#0e1928', borderColor: '#314962',
        textStyle: { color: '#ecf4ff', fontSize: 11 },
      },
      xAxis: {
        type: 'category', data: xs,
        axisLabel: { color: '#8d9ab0', fontSize: 10 },
        axisLine: { lineStyle: { color: '#28364a' } },
      },
      yAxis: [
        {
          type: 'value', name: 'Tokens', nameTextStyle: { color: '#8d9ab0', fontSize: 10 },
          axisLabel: { color: '#8d9ab0', formatter: (v: number) => compact(v) },
          splitLine: { lineStyle: { color: '#1e2c40', type: 'dashed' } },
        },
        {
          type: 'value', name: 'Cache %', min: 0, max: 100,
          nameTextStyle: { color: '#8d9ab0', fontSize: 10 },
          axisLabel: { color: '#8d9ab0', formatter: (v: number) => `${v}%` },
          splitLine: { show: false },
        },
      ],
      series: [
        { name: 'Cache Hit', type: 'bar', stack: 'ctx', data: hits, itemStyle: { color: '#3bc9b4' } },
        { name: 'Cache Miss', type: 'bar', stack: 'ctx', data: misses, itemStyle: { color: '#e8b36b' } },
        { name: 'Reasoning Out', type: 'line', data: reasoning, smooth: true, symbol: 'none', lineStyle: { color: '#b79cff', width: 1.5 } },
        { name: 'Cache Hit Rate', type: 'line', yAxisIndex: 1, data: hitRate, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#70b7ff', width: 1.5, type: 'dashed' }, itemStyle: { color: '#70b7ff' } },
      ],
    }
    chart.setOption(opt)
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(container.current)
    return () => { resize.disconnect(); chart.dispose() }
  }, [revision, points.length])

  if (!points.length) {
    return <div className="curve-empty"><ChartNoAxesCombined size={30} /><strong>No model requests recorded</strong><span>Context growth appears as DeepSeek turns settle.</span></div>
  }
  return <div ref={container} className="curve context-chart" role="img" aria-label="Context growth and prefix cache hit chart" />
}

export function CurveLegend() {
  return <div className="legend"><span><i className="solid-line" />Observed best</span><span><i className="dot measured" />Measured (click to inspect)</span><span><i className="dash-line" />Paired E0</span><span><Activity size={14} />Development only</span></div>
}
