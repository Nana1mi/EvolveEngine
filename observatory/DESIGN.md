# RSI Observatory — source review and implementation decisions

Reviewed on 2026-11-26. This is a **local, read-only sidecar**, not a replacement
for the Runner, Journal, Judge, budget ledger, or resource arbiter.

| Upstream | Reviewed revision / license | Decision |
| --- | --- | --- |
| [`comet-ml/opik`](https://github.com/comet-ml/opik/tree/cfdb0665b498bc958845cd5568394fd5ca7222b6/apps/opik-frontend/src/v2/pages-shared/experiments/OptimizationProgressChart) | `cfdb0665b498bc958845cd5568394fd5ca7222b6`, Apache-2.0 | Chart content uses Recharts plus internal chart wrappers, optimization types, status/legend helpers, custom hit testing, and a shared Radix popover. Preserve the chart → trial card → trace interaction, not the implementation. |
| [Opik TraceDetailsPanel](https://github.com/comet-ml/opik/tree/cfdb0665b498bc958845cd5568394fd5ca7222b6/apps/opik-frontend/src/v2/pages-shared/traces/TraceDetailsPanel) | same revision/license | React Query, trace/span API hooks, URL state, workspace permissions, internal tree store, graph and analytics wiring. No direct transplant. Read-only detail and trace views use our own event identities. |
| [`satnaing/shadcn-admin`](https://github.com/satnaing/shadcn-admin/tree/a156d9b90519348c4d71f0f0e6fcea179c49a43f/src/components/layout) | `a156d9b90519348c4d71f0f0e6fcea179c49a43f`, MIT | Reviewed AppSidebar, AuthenticatedLayout and NavGroup. Use the grouped sidebar, workspace picker, content/inspector pattern, without bringing in its router/auth/team/sponsor scaffolding. |
| `jhochenbaum/pi-autoresearch-studio`, `EricJahns/pi-autoresearch-vkf` | `96920ee8726170f5380b7f04fe6e150c87a9d428` and `5fc304ec374cd5a29a1d921996a950bca3733d2e`, MIT | Comparison references only: compact trial progress and run history; do not bind to their input or write protocols. |
| `aimhubio/aim`, `asvskartheek/autoresearch-scaffold` | `6e098e38065364c76b2bb7c028f266e53b647642` Apache-2.0; `06bd1259a613d539f4e6c9a7e0b443d6cca7dee2` MIT | Comparison references only: resource axes and evaluation stages. |

No upstream application source is copied or vendored; there are no patches to
those projects. We reuse Apache ECharts modules for plotting and Lucide for icons
and keep exact npm versions/licenses in the lockfile/package installs. Direct
component portability was assessed from the pinned sources; we do not claim to
have built a modified Opik or Shadcn Admin checkout. Context7 was unreachable;
the pinned upstream sources, installed types and a real bundle/type check were
used to verify the choice. Standalone React/TypeScript and a Python standard-library
loopback server avoid a second app scaffold, service database, or external CDN.

## First vertical slice

1. Project only explicitly registered controller run roots and an existing local Level 3
   archive when present. Follow existing SQLite (read-only) and selected sanitized receipt metadata.
2. One versioned snapshot for initial load, refresh, SSE reconnect, and closed runs.
3. Chart with separate units, paired E0 reference, measured points and rolling observed
   best. Journal-only failures never get a zero y coordinate. History/trace/evaluation
   drill down with the same identities. Show an actual delivery marker only when recorded;
   it is not a measured or audited promotion chain.
4. Versions distinguish declared activation from evidenced actual load. Resources
   separate the conservative ledger, token totals, and estimated fees from provider bills.
5. Replay through recorded performance points and expose evidence fingerprints; no
   hidden Q/T, arbitrary files, raw context, candidate output, credentials, or Run controls.

Deferred: per-tool span instrumentation where not already persisted, sanitized source
and harness diffs via an explicit audited export path, independently attested promotion
chains, multi-worker lanes beyond recorded attribution, comparison of multiple run
series, production telemetry, and paper-grade uncertainty analysis/exports. An absent
record must remain absent; adding UI does not manufacture trace data or approvals.
