"""Use a silent loopback HTTP server to exercise the real Pi, Relay and Broker wait."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
from unittest.mock import patch

import p3_explore as explore


def run(output, silence, override_idle=None, override_socket_idle=None):
    config, limits = explore.load_config(explore.CONFIG)
    transport = config['transport']
    if override_idle is not None:
        transport['upstream_idle_seconds'] = override_idle
    if override_socket_idle is not None:
        transport['relay_socket_idle_seconds'] = override_socket_idle
    idle = transport['upstream_idle_seconds']
    archive = Path(os.environ['LOCALAPPDATA']) / 'EvolveEngine'
    source = (archive/'v3-pinned-sglang-source/gguf.py').read_bytes()
    counter = explore.narrow.PublicPromptCounter(archive/'deepseek-flash-1017-08c8657f-offline')
    scenes = explore.p3.record(explore.p3.HERE/'g1a-03.json')['common_runner']['worker'][1]['result']['scenes']
    requests = []
    released_after = []

    class MockUpstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request['max_tokens'])
            started = monotonic()
            print(f'Mock upstream received request; withholding response headers and body for {silence}s', flush=True)
            sleep(silence)
            released_after.append(monotonic()-started)
            chunk = dict(model='deepseek-flash', system_fingerprint=explore.narrow.FINGERPRINT,
                         choices=[dict(index=0, delta={'content':'Done.'}, finish_reason='stop')],
                         usage=dict(prompt_tokens=500, completion_tokens=2, total_tokens=502))
            body = b'data: '+explore.p3.canonical(chunk)+b'\n\ndata: [DONE]\n\n'
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def baseline(op, edits, engine, _base, _lease, _job):
        assert (op, edits) == ('develop_measure', {})
        return explore.p3.DevReceipt(engine, op, 'observed', 100, 'offline-e0', dict(scenes=scenes))

    with ThreadingHTTPServer(('127.0.0.1', 0), MockUpstream) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = explore.p3.Provider(f'http://127.0.0.1:{server.server_port}', 'deepseek-flash',
                                       'EE_OFFLINE_MOCK_KEY', 'DeepSeek-V4.1-Flash', limits.per_turn_output)
        try:
            with patch.dict(os.environ, {'EE_OFFLINE_MOCK_KEY':'local-fixture', 'no_proxy':'127.0.0.1,localhost'}):
                report = explore.run(source, output, config, limits, counter,
                                     explore.provider_transport(provider, idle), baseline)
        finally:
            server.shutdown()
            thread.join()
    runner, row = report['runner'], report['provider'][0]
    assert requests == [limits.per_turn_output], requests
    assert runner['relay_model_socket_idle_seconds'] == transport['relay_socket_idle_seconds'], runner
    assert runner['transport_fault'] is None, runner
    assert runner['isolated_state_removed'] and runner['socket_removed'], runner
    if silence < idle:
        assert released_after and released_after[0] >= silence, released_after
        assert report['status'] == 'session_completed' and row['finish'] == 'stop', report['status']
        assert row['wall_ms'] >= silence*1000, row
    else:
        assert report['stop_cause'] == 'upstream_transport', report['stop_cause']
        assert row['wall_ms'] >= idle*1000-150, row
        assert row['usage'] == 'uncertain_charged_reservation', row
    return dict(silence_seconds=silence, actual_silence_seconds=released_after[0] if released_after else None,
                relay_socket_idle_seconds=transport['relay_socket_idle_seconds'], upstream_idle_seconds=idle,
                status=report['status'], stop_cause=report['stop_cause'], provider_wall_ms=row['wall_ms'],
                relay_transport_fault=runner['transport_fault'], requested_max_tokens=requests[0],
                provider_usage=row['usage'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--silence-seconds', type=float, default=1205)
    parser.add_argument('--upstream-idle-seconds', type=int)
    parser.add_argument('--relay-socket-idle-seconds', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        with TemporaryDirectory() as tmp:
            result = run(Path(tmp)/'silent', args.silence_seconds,
                         args.upstream_idle_seconds, args.relay_socket_idle_seconds)
    else:
        result = run(args.output, args.silence_seconds,
                     args.upstream_idle_seconds, args.relay_socket_idle_seconds)
        explore.p3.write_once(args.output/'silent-upstream-check.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
