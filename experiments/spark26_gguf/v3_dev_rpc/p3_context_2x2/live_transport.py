"""One authorized paid P3 diagnostic. No agent, GPU, retries, or provider preflight."""
import argparse
from datetime import datetime, timezone
from http.client import HTTPSConnection, HTTPException
import os
from pathlib import Path
import re
import sys
import time

import diagnostic as d

FROZEN_SHA = '4259814224d61a4f13323b878736ce93483b023c4d33af0256f0103984d648f2'
RUN = 'p3-context-2x2-live-v1-20261123'
TERMINALS = ('stop', 'length', 'tool_calls', 'content_filter', 'insufficient_system_resource', 'aborted', None)


class TransportFailure(Exception):
    def __init__(self, reason, status=None):
        self.reason, self.status = reason, status


def submit(key, wire):
    """Exactly one POST; no redirects, implicit retries, follow-up ping, or external tools."""
    deadline = time.monotonic() + 180
    conn = HTTPSConnection('api.deepseek.com', 443, timeout=60)
    try:
        conn.request('POST', '/chat/completions', body=wire, headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
            'Accept': 'text/event-stream', 'Accept-Encoding': 'identity', 'Connection': 'close',
        })
        with conn.getresponse() as res:
            if res.status != 200:
                raise TransportFailure('upstream_http', res.status)
            mime = res.getheader('Content-Type', '').split(';', 1)[0].strip().lower()
            if mime != 'text/event-stream':
                raise TransportFailure('unexpected_content_type', res.status)
            raw = bytearray()
            while True:
                if time.monotonic() >= deadline:
                    raise TransportFailure('response_deadline')
                take = min(65536, d.MAX_STREAM_BYTES + 1 - len(raw))
                piece = res.read1(take)
                if not piece:
                    return bytes(raw)
                raw.extend(piece)
                if len(raw) > d.MAX_STREAM_BYTES:
                    raise TransportFailure('response_too_large')
    except (OSError, HTTPException):
        raise TransportFailure('transport_failure') from None
    finally:
        conn.close()


def execute(bundle, counter, send, record):
    """Reuse the frozen parser, scorer and budget rules. Persist each attempt before the wire."""
    cells = bundle.manifest['cells']
    files = bundle.manifest['frozen_files']
    d.require(bundle.manifest_sha256 == FROZEN_SHA and tuple(c['cell'] for c in cells) == d.ORDER,
              'unapproved_manifest_or_order')
    wires = tuple(bundle.files[c['request']] for c in cells)
    d.require(all(d.digest(wire) == files[c['request']]['sha256'] for c,wire in zip(cells,wires)),
              'unapproved_request_bytes')
    rows = []
    allocated_input = allocated_output = charged_input = charged_output = attempts = 0
    stop = None
    for cell, wire in zip(cells, wires):
        row = dict(cell=cell['cell'], request_sha256=d.digest(wire), request_bytes=len(wire),
                   tool_result_bytes=cell['read_bytes'], dispatched=False)
        rows.append(row)
        try:
            request = d.decode(wire)
            parts = counter.reservation(request)
            reserve = d.validate_budget(parts, allocated_input, allocated_output)
            d.require(parts == cell['reservation'] and request['max_tokens'] == d.PER_OUTPUT, 'reservation_drift')
        except (ValueError, KeyError, TypeError, AssertionError, RuntimeError, OSError, IndexError):
            row.update(status='reservation_failed', charged_input=0, charged_output=0)
            record('blocked_before_dispatch', row)
            stop = row['status']
            break
        allocated_input += reserve
        allocated_output += d.PER_OUTPUT
        attempts += 1
        row.update(dispatched=True, input_reserved=reserve, output_reserved=d.PER_OUTPUT)
        record('attempt_committed_before_dispatch', row)
        started = time.monotonic()
        try:
            raw = send(wire)
        except TransportFailure as exc:
            raw = None
            row.update(transport_issue=exc.reason, http_status=exc.status)
        row['wall_ms'] = round((time.monotonic() - started) * 1000)
        stream = d.Stream(raw)
        actual = stream.usage
        over = actual is not None and (actual['input'] > reserve or actual['output'] > d.PER_OUTPUT)
        if actual is None or over:
            debit_in = max(reserve, actual['input']) if actual else reserve
            debit_out = max(d.PER_OUTPUT, actual['output']) if actual else d.PER_OUTPUT
            row.update(status='usage_uncertain', usage_issue='over_reservation' if over else stream.usage_issue)
        else:
            debit_in, debit_out = actual['input'], actual['output']
            row['status'] = 'fingerprint_mismatch' if not stream.fingerprint_ok else (
                'protocol_invalid' if stream.issue else 'settled')
        charged_input += debit_in
        charged_output += debit_out
        row.update(reported_usage=actual, charged_input=debit_in, charged_output=debit_out,
                   input_overrun=max(0, actual['input'] - reserve) if actual else None,
                   output_overrun=max(0, actual['output'] - d.PER_OUTPUT) if actual else None,
                   fingerprint_matches=stream.fingerprint_ok if type(raw) is bytes and raw else None,
                   finish_reason=stream.finish if stream.finish in TERMINALS else 'unexpected',
                   protocol_issue=stream.issue, response_sha256=d.digest(raw) if type(raw) is bytes else None)
        if row['status'] == 'settled':
            try:
                row.update(d.classify(stream, counter))
            except (ValueError, TypeError, RuntimeError):
                row['status'] = 'classification_unavailable'
        record('settled_once', row)
        if row['status'] != 'settled':
            stop = row['status']
            break
    return dict(run=RUN, protocol=d.ID, manifest_sha256=bundle.manifest_sha256, rows=rows,
                attempted_paid_calls=attempts, gpu_calls=0, allocated_input=allocated_input,
                allocated_output=allocated_output, charged_input=charged_input, charged_output=charged_output,
                stop=stop, decision=d.decision(rows, stop))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--expected-driver-sha256', required=True)
    args = parser.parse_args()
    driver_sha = d.digest(Path(__file__).read_bytes())
    d.require(driver_sha == args.expected_driver_sha256, 'unapproved_driver_bytes')
    bundle, counter = d.verify(args.assets, FROZEN_SHA)
    key = sys.stdin.readline(514).rstrip('\r\n')
    d.require(bool(re.fullmatch(r'[A-Za-z0-9_-]{10,512}', key)), 'invalid_credential_input')
    folder = Path(os.environ['LOCALAPPDATA']) / 'EvolveEngine' / RUN
    # Permanent one-use lock. Any interrupted run remains terminal until separate approval.
    folder.mkdir(exist_ok=False)
    with (folder / 'journal.jsonl').open('xb') as log:
        def record(event, data):
            raw = d.canonical(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), data=data)) + b'\n'
            log.write(raw)
            log.flush()
            os.fsync(log.fileno())
        record('execution_started', dict(manifest_sha256=FROZEN_SHA, driver_sha256=driver_sha,
                                         order=list(d.ORDER), scope='authorized_single_2x2'))
        result = execute(bundle, counter, lambda wire: submit(key, wire), record)
        result['driver_sha256'] = driver_sha
        record('execution_finished', dict(stop=result['stop'], decision=result['decision'],
                                         attempted_paid_calls=result['attempted_paid_calls']))
        with (folder / 'receipt.json').open('xb') as out:
            out.write(d.manifest_bytes(result))
            out.flush()
            os.fsync(out.fileno())
    print(d.canonical(dict(attempted_paid_calls=result['attempted_paid_calls'], stop=result['stop'],
                           decision=result['decision'], gpu_calls=0)).decode())


if __name__ == '__main__':
    main()
