"""Transparent isolated Unix socket to trusted stdio bridge. Contains no keys or policy."""
import base64
from http.server import BaseHTTPRequestHandler
import json
import os
import socketserver
import sys
from threading import Lock

IN_LIMIT=512*1024
OUT_LIMIT=6*1024*1024
SOCKET_IDLE_SECONDS=390


def main():
    binary_replies = len(sys.argv) > 1
    model_in, model_out, model_socket_idle = (map(int, sys.argv[1:]) if binary_replies
                                               else (IN_LIMIT, OUT_LIMIT, SOCKET_IDLE_SECONDS))
    lock=Lock(); pipes=(sys.stdin.buffer,sys.stdout.buffer); counter=0
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_POST(self):
            nonlocal counter
            try:
                model_route = self.path == '/v1/chat/completions'
                # The timeout covers socket I/O. The trusted host reply arrives over stdio.
                self.connection.settimeout(model_socket_idle if model_route else SOCKET_IDLE_SECONDS)
                n=int(self.headers.get('Content-Length','0'))
                in_limit, out_limit = (model_in, model_out) if model_route else (IN_LIMIT, OUT_LIMIT)
                if not 0<n<=in_limit:raise ValueError('bad length')
                raw=self.rfile.read(n)
                if len(raw)!=n:raise ValueError('truncated request')
                if not self.path.startswith('/') or len(self.path)>150:raise ValueError('bad path')
                with lock:
                    counter+=1; i=counter
                    frame=dict(id=i,path=self.path,body=base64.b64encode(raw).decode('ascii'),auth=self.headers.get('Authorization'))
                    pipes[1].write(json.dumps(frame,separators=(',',':')).encode()+b'\n');pipes[1].flush()
                    header_limit = 4096 if binary_replies else out_limit
                    line=pipes[0].readline(header_limit+1)
                    if not line.endswith(b'\n') or len(line)>header_limit:raise OSError('relay closed')
                    reply=json.loads(line)
                    if reply['id']!=i:raise OSError('sequence mismatch')
                    typ, code = reply['type'], reply['status']
                    if typ not in ('application/json','text/event-stream') or type(code)is not int or not 200<=code<=599:
                        raise ValueError('bad response')
                    if binary_replies:
                        remaining = reply['length']
                        if type(remaining) is not int or not 0 <= remaining <= out_limit:
                            raise ValueError('bad response length')
                        try:
                            self.send_response(code)
                            self.send_header('Content-Length', str(remaining))
                            self.send_header('Content-Type', typ)
                            self.end_headers()
                            while remaining:
                                part = pipes[0].read(min(64*1024, remaining))
                                if not part:
                                    raise OSError('incomplete relay body')
                                remaining -= len(part)
                                self.wfile.write(part)
                        except OSError:
                            # Keep the stdio framing intact if the guest disconnected.
                            while remaining:
                                part = pipes[0].read(min(64*1024, remaining))
                                if not part:
                                    break
                                remaining -= len(part)
                            raise
                        return
                out=base64.b64decode(reply['body'],validate=True)
                self.send_response(code);self.send_header('Content-Length',str(len(out)));self.send_header('Content-Type',typ);self.end_headers()
                self.wfile.write(out)
            except (ValueError,KeyError,OSError,TypeError):
                self.close_connection=True
    class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
        daemon_threads=True
    server=Server('/gate/gateway.sock',Handler);os.chmod('/gate/gateway.sock',0o666)
    sys.stdout.buffer.write(b'{"ready":true}\n');sys.stdout.buffer.flush();server.serve_forever()

if __name__=='__main__':main()
