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


def main():
    lock=Lock(); pipes=(sys.stdin.buffer,sys.stdout.buffer); counter=0
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_POST(self):
            nonlocal counter
            try:
                self.connection.settimeout(390)
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=IN_LIMIT:raise ValueError('bad length')
                raw=self.rfile.read(n)
                if len(raw)!=n:raise ValueError('truncated request')
                if not self.path.startswith('/') or len(self.path)>150:raise ValueError('bad path')
                with lock:
                    counter+=1; i=counter
                    frame=dict(id=i,path=self.path,body=base64.b64encode(raw).decode('ascii'),auth=self.headers.get('Authorization'))
                    pipes[1].write(json.dumps(frame,separators=(',',':')).encode()+b'\n');pipes[1].flush()
                    line=pipes[0].readline(OUT_LIMIT+1)
                    if not line.endswith(b'\n') or len(line)>OUT_LIMIT:raise OSError('relay closed')
                    reply=json.loads(line)
                    if reply['id']!=i:raise OSError('sequence mismatch')
                out=base64.b64decode(reply['body'],validate=True)
                typ=reply['type'];code=reply['status']
                if typ not in ('application/json','text/event-stream') or type(code)is not int or not 200<=code<=599:raise ValueError('bad response')
                self.send_response(code);self.send_header('Content-Length',str(len(out)));self.send_header('Content-Type',typ);self.end_headers()
                self.wfile.write(out)
            except (ValueError,KeyError,OSError,TypeError):
                self.close_connection=True
    class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
        daemon_threads=True
    server=Server('/gate/gateway.sock',Handler);os.chmod('/gate/gateway.sock',0o666)
    sys.stdout.buffer.write(b'{"ready":true}\n');sys.stdout.buffer.flush();server.serve_forever()

if __name__=='__main__':main()
