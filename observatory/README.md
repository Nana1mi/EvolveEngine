# EvolveEngine RSI Observatory

A **local, read-only** dashboard for Level 3 engineering exploration. It follows
existing Controller receipts; it does not launch Pi, contact a model API,
reserve a node, alter the Judge, or enter a holdout run.

## Build and open

Run from the EvolveEngine repository root with the project's Python and npm:

```powershell
npm --prefix observatory ci
npm --prefix observatory run build
python -m evolveengine.observatory --port 8848
```

Open `http://127.0.0.1:8848/`. With no `--run` the dashboard shows the repository's
local Level 3 archive summary when that evidence file is present. For full detail, repeat `--run` once per local,
trusted Controller evidence directory (the directory containing
`registration.json`, not its `run/` child):

```powershell
python -m evolveengine.observatory --port 8848 --run "engineering=C:\path\to\controller\exploration"
```

A future directory can be registered before the Controller creates it. Its
status stays `awaiting receipts` until a qualifying registration is available.
`Following receipts` only describes the browser's SSE connection; it does not
prove that DeepSeek, the Controller or the Worker is alive. Use `Ctrl+C` in
this terminal to stop only the dashboard. Use `--no-archive` to omit the local
archive summary. Do not expose the server to the LAN or use untrusted roots.

## How to read the dashboard

- **Overview:** choose wall time, measurement order, engine iteration, session,
  settled tokens, the local API ledger or node time. Choose a paired development
  ratio or a W1–W4 throughput/latency metric. Select a point for its evidence
  IDs and paired E0 measurement, scrub measured points, or export them as CSV.
  Failed attempts without a measurement are events, not zero-valued points.
- **Update history / Trace / Evaluation:** inspect recorded attempts, Worker
  returns, gateway/settlement data where recorded, and Pi tool spans where the
  terminal report actually includes both boundaries. Controller order, broker
  request order and Pi events are **not** promoted to a shared timeline.
- **Engine / Harness:** inspect registered ancestry, staged and activated
  Harness versions, session attachment, checkpoints and recovery. Activation
  and attachment are not independent proof of loading or invocation. An Agent
  nomination does not identify an exact timing point unless the evidence does.
- **Research / Resources:** show safe research-event metadata, wall/node time,
  token/cache accounting and estimates. Point token/ledger x-values are host
  snapshots **at Worker dispatch**; point wall time is the bound Worker return
  clock. The local ledger and rate-based estimates are **not** provider invoices.

The observed best is in the selected development view; it is not a promotion,
independent weight-audit result, or result on production/holdout. No missing
measurement is substituted with a zero. The archive alone lacks per-point
clocks and trace detail. This first version does not add new Pi span
instrumentation, diff viewing, independent load attestation, cross-run series
comparison, multi-node aggregation, or paper-grade analytics.

## API and boundaries

- `GET /api/runs`: explicit run labels and modes, not filesystem paths.
- `GET /api/runs/{id}/snapshot`: versioned allowlisted snapshot.
- `GET /api/runs/{id}/events`: SSE `snapshot` events; reconnect using
  `Last-Event-ID` (content revision). Only the latest snapshot is served; it is
  not a guaranteed-delivery event queue.

The sidecar reads the Journal in SQLite read-only mode and verifies stored
Worker object addresses before projecting comparisons. It publishes only
allowlisted metadata, not raw prompts, completions, secrets, local paths,
unregistered files, or holdout records. The run producer and its receipts
remain the sources of truth. See [DESIGN.md](DESIGN.md) for the source review.

```powershell
python -m unittest discover -s tests -p test_observatory.py -v
```

## Server deployment and private access

The viewer can run on a separate server without copying the Controller. Bind
the viewer to `127.0.0.1` on that server and start it with
`--published LIVE-ID=/absolute/snapshots/LIVE-ID.json`. Register the **new** ID
first to make the new curve the default; register historical IDs separately.
A missing file means waiting for receipts, not a zero-valued run. The publisher
uses the same projection as the local viewer and sends **only its sanitized
snapshots** over SSH/SFTP, with an atomic rename on the server:

```powershell
uv run --offline --with paramiko==5.1.0 python -m evolveengine.observatory_publish `
  --host observatory.example.internal --user publisher --remote-directory /home/publisher/ee-observatory/snapshots `
  --run 'new-live-run=C:\absolute\path\to\the\new\run' --watch-seconds 10 --password-prompt
```

SSH host keys must already be trusted. No model key or raw output is copied.
A publisher stopped or disconnected only makes the viewer stale; it cannot stop
or modify the experiment. A fresh publisher can republish the last known state.
The server retains the last published snapshots when the Controller goes offline.

For one viewer with authorized SSH access, use
`ssh -N -L 8848:127.0.0.1:8848 viewer@observatory.example.internal` and then open
`http://127.0.0.1:8848/`; the local port is a tunnel, not a local dashboard server.
For a normal browser URL that works for other company Wi-Fi users, use an
approved intranet hostname and authenticated HTTPS reverse proxy. Preserve the
external Host, authenticate **both** the pages and `/api/*` (including SSE), and
disable buffering on the event-stream route. Configure the backend with
`--viewer-origin https://<approved-host>`; this adds a Host/Origin match, **not**
authentication or a publicly bound listener. Do not bind it directly to Wi-Fi.

## Effective exploration time

For a run with a trusted timing ledger, the chart defaults to effective time.
Wall time is always retained. The ledger is a sibling of the Controller output
directory, `<run-directory-name>.observatory-timing.json`, at the same clock
origin as that run's worker receipt `elapsed_ms`:

```json
{
  "schema": "ee-observatory-timing-v1",
  "approval_sha256": "<run approval hash>",
  "events": [
    {"kind":"pause","cause":"infrastructure_repair","wall_ms":120000},
    {"kind":"resume","wall_ms":150000},
    {"kind":"infrastructure_change","wall_ms":150000,"infrastructure_sha256":"<new version hash>"}
  ]
}
```

Update the ledger atomically, preserving previous records. `infrastructure_fault`
can mark a fault without treating it as a pause. Unpaired or inferred handoffs
are not deducted. A recorded pause without a resume is treated as still open.
An ordinary handoff does not imply pause, and this viewer does not stitch
independent Controller launches together or manufacture missing timestamps.
