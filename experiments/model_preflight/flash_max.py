"""Real Pi + DeepSeek Flash Max smoke; only a synthetic one-file fixture is shared.
Run: python flash_max.py <new-receipt.json>. Env: DEEPSEEK_API_KEY. No GPU.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evolveengine.deepseek import Provider, ProviderBudget
from evolveengine.pi import pi_events, json_answer
from evolveengine.schema import Cost, canonical

CLI = REPO/'.evolveengine/pi-smoke/node_modules/@earendil-works/pi-coding-agent/dist/cli.js'
GUARD = Path(__file__).with_name('fixture_guard.mjs')
BASE = 'def decode_rate(n, first_ms, last_ms):\n    return n / (last_ms - first_ms)\n'
MODEL = 'deepseek-flash'
LIMIT = Cost(wall_ms=150000,input_tokens=32000,output_tokens=6000)
TURN_CAP, TURN_MAX = 5, 2000
PROMPT = ('Repair decode_rate in metric.py. Its inputs are the count of emitted tokens and monotonic client timestamps in milliseconds of the first and last emitted token. Return the subsequent decode rate in tokens per second, excluding the first token. Return None when fewer than two tokens were emitted or elapsed time is not positive. Keep the signature, use plain comparisons/arithmetic without helpers or imports, and do not add files. The read tool arguments must be exactly {"path":"metric.py","offset":1,"limit":8}, with numeric offset/limit. Use the same literal relative path metric.py for edit (not an absolute path or ./metric.py). Report only {"edited":true} after the edit.')
CASES = [(1,5,100,None),(0,1,2,None),(2,100,100,None),(6,110,108,None),(2,10,510,2),(5,210,410,20),(256,100,2850,255/2.75),(3,0.125,0.375,8000)]


def demand(x,msg):
    if not x: raise ValueError(msg)


def oracle(path):
    source=path.read_bytes(); demand(source!=BASE.encode() and len(source)<4096,'empty/oversized patch'); tree=ast.parse(source)
    demand(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef),'expected one function'); fn=tree.body[0]
    demand(fn.name=='decode_rate' and [v.arg for v in fn.args.args]==['n','first_ms','last_ms'] and not (fn.args.defaults or fn.decorator_list or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg),'signature mismatch')
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.If,ast.IfExp,ast.Assign,ast.Name,ast.Load,ast.Store,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Compare,ast.Lt,ast.LtE,ast.Eq,ast.NotEq,ast.Gt,ast.GtE,ast.BoolOp,ast.And,ast.Or,ast.UnaryOp,ast.USub,ast.Not,ast.Constant,ast.Expr)
    demand(all(isinstance(v,allowed) for v in ast.walk(tree)),'outside side-effect-free fixture grammar'); ns={'__builtins__':{}}; exec(compile(tree,'<fixture>','exec'),ns)
    results=[]
    for n,f,l,want in CASES:
        value=ns['decode_rate'](n,f,l); results.append(value is None if want is None else type(value) in (int,float) and math.isclose(value,want))
    return results, hashlib.sha256(source).hexdigest()


class MaxWire(ProviderBudget):
    """Isolated preflight: no production/default broker overrides or silent mode conversion."""
    def handle(self,h):
        if h.path!='/chat/completions' or not secrets.compare_digest(h.headers.get('Authorization',''),'Bearer '+self.token):
            h.send_error(403); return
        started=time.monotonic(); row={}; sent=False
        try:
            demand(not self.failure and len(self.rows)<TURN_CAP and self.ready.is_file(),'lease/guard not available'); size=int(h.headers.get('Content-Length','0')); demand(0<size<90000,'body out of range'); req=json.loads(h.rfile.read(size)); demand(req.get('model')==MODEL and req.get('stream') is True,'unexpected model or stream'); demand(req.get('thinking')=={'type':'enabled'} and req.get('reasoning_effort')=='max','Pi did not request enabled + max');
            ceiling=min(TURN_MAX,self.limit.output_tokens-self.spent.output_tokens); input_bound=len(canonical([req['messages'],req.get('tools',[])]).encode())+256; demand(input_bound<=self.limit.input_tokens-self.spent.input_tokens and ceiling>0,'token cap'); left=self.deadline-time.monotonic(); demand(left>0,'deadline'); req.pop('max_completion_tokens',None); req['max_tokens']=ceiling; req['stream_options']={'include_usage':True}; wire=canonical(req).encode()
            row.update(turn=len(self.rows)+1,model_requested=MODEL,thinking='enabled',reasoning_effort='max',input_reservation=input_bound,max_tokens=ceiling,prior_tool_thinking_replayed=any(m.get('role')=='assistant' and m.get('tool_calls') and m.get('reasoning_content') for m in req['messages']),upstream_body_sha256=hashlib.sha256(wire).hexdigest()); self.rows.append(row); sent=True
            with self.config.request('/chat/completions',payload=wire,timeout=left) as response:
                body,usage,fps,models=self.collect_stream(response,ceiling)
            seen=False; finishes=[]
            for line in bytes(body).splitlines():
                if not line.startswith(b'data:') or line[5:].strip()==b'[DONE]': continue
                chunk=json.loads(line[5:]); choices=chunk.get('choices') or []
                seen |= any(bool(c.get('delta',{}).get('reasoning_content')) for c in choices); finishes.extend(c['finish_reason'] for c in choices if c.get('finish_reason'))
            u,v=usage.get('prompt_tokens'),usage.get('completion_tokens'); demand(type(u) is int and type(v) is int and u>0 and v>0,'missing provider usage'); self.spent+=Cost(input_tokens=u,output_tokens=v); row.update(models=sorted(models),fingerprints=sorted(fps),reasoning_delta_seen=seen,finish_reasons=finishes,usage=usage,provider_ms=round((time.monotonic()-started)*1000)); demand(models=={MODEL} and usage.get('total_tokens')==u+v and u<=input_bound and v<=ceiling,'identity/usage mismatch'); demand(b'[DONE]' in body and finishes,'incomplete SSE'); h.send_response(200); h.send_header('Content-Type','text/event-stream'); h.send_header('Content-Length',str(len(body))); h.end_headers(); h.wfile.write(body)
        except Exception as e:
            self.failure=f'HTTP {e.code}' if isinstance(e,HTTPError) else str(e)[:160]
            if sent and 'usage' not in row: row['settlement']='unknown, do not automatically retry'
            h.send_error(502,'preflight failed'); return


def main(out):
    demand(not out.exists(),'receipt already exists'); start=time.monotonic(); result={'status':'failed','scope':'isolated Pi model/tool/metering preflight; synthetic fixture; no engine, GPU or RSI','ceiling':dict(input=LIMIT.input_tokens,output=LIMIT.output_tokens,turns=TURN_CAP,seconds=150)}
    try:
        ver=subprocess.check_output(['node',str(CLI),'--version'],timeout=15,text=True).strip(); demand(ver=='1.5.2','Pi version drift'); provider=Provider('https://api.deepseek.com',MODEL,'DEEPSEEK_API_KEY','DeepSeek-V4.1-Flash',TURN_MAX); result.update(pi=ver,identity=provider.preflight())
        with tempfile.TemporaryDirectory(prefix='ee-max-probe-') as tmp:
            base=Path(tmp).resolve(); root=base/'toy'; root.mkdir(); ag=base/'pi'; ag.mkdir(); ready=ag/'ready'; tools=base/'tools.jsonl'; target=root/'metric.py'; target.write_bytes(BASE.encode());
            with MaxWire(provider,LIMIT,ready) as broker:
                profile={'id':MODEL,'name':'DeepSeek V4.1 Flash','reasoning':True,'thinkingLevelMap':{'minimal':None,'low':'low','medium':None,'high':'high','max':'max'},'contextWindow':1000000,'maxTokens':TURN_MAX,'compat':{'supportsStore':False,'supportsDeveloperRole':False,'maxTokensField':'max_tokens','thinkingFormat':'deepseek','requiresReasoningContentOnAssistantMessages':True,'supportsReasoningEffort':True}}
                (ag/'models.json').write_text(canonical({'providers':{'wire':{'baseUrl':broker.local_url,'api':'openai-completions','apiKey':broker.token,'models':[profile]}}})); (ag/'settings.json').write_text(canonical({'compaction':{'enabled':False},'retry':{'enabled':False,'provider':{'maxRetries':0}}})); env={k:v for k,v in os.environ.items() if k.upper()!='DEEPSEEK_API_KEY'}; env.update(PI_CODING_AGENT_DIR=str(ag),EVOLVE_PI_GUARD=canonical(dict(root=str(root),edit=['metric.py'],read=['metric.py'],ready=str(ready),compile=None,receipt=str(tools),max_lines=16,max_tools=5)));
                args=['node',str(CLI),'--mode','json','--provider','wire','--model',MODEL,'--thinking','max','--no-session','--no-extensions','--no-mcp','--no-skills','--no-prompt-templates','--no-context-files','--system-prompt','You repair only the synthetic fixture specified by the user. End with the requested JSON object.','-e',str(GUARD),'--tools','read,edit']; t=time.monotonic()
                try:
                    proc=subprocess.run(args,input=PROMPT.encode(),capture_output=True,cwd=root,env=env,timeout=150); demand(not broker.failure,broker.failure or 'broker failed'); demand(proc.returncode==0,'Pi nonzero exit'); events,ans=pi_events(proc.stdout); result['tool_completions']=[{'name':e.get('toolName'),'error':e.get('isError')} for e in events if e['type']=='tool_execution_end']; demand(json_answer(ans)=={'edited':True},'incorrect completion'); demand(ready.is_file() and tools.is_file(),'missing tool evidence'); checks,sha=oracle(target); result.update(oracle_passed=sum(checks),oracle_total=len(checks),patch_sha256=sha); demand(all(checks),'fixture hidden checks failed'); demand(len(list(root.iterdir()))==1,'unexpected file');
                finally:
                    result.update(session_ms=round((time.monotonic()-t)*1000),turns=broker.rows,usage=broker.spent.record(),tools=[json.loads(l) for l in tools.read_text().splitlines()] if tools.exists() else [])
                demand(all(r['accepted'] for r in result['tools']) and {r['tool'] for r in result['tools']}=={'read','edit'},'wrong tools'); demand(len(result['tool_completions'])==len(result['tools']) and all(e['error'] is False for e in result['tool_completions']),'tool completion not confirmed'); demand(any(r.get('reasoning_delta_seen') for r in broker.rows) and any(r['prior_tool_thinking_replayed'] for r in broker.rows),'no evidence of thinking/tool round-trip'); result['status']='pass'
    except (Exception,) as e: result['failure']=str(e)[:200]
    result['total_ms']=round((time.monotonic()-start)*1000); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); print(json.dumps(result,ensure_ascii=True,indent=2)); return result['status']!='pass'

if __name__=='__main__': sys.exit(main(Path(sys.argv[1])))
