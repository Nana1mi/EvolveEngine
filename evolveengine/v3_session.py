"""Trusted session operations and accounting. Backend callbacks are host-only."""
import ast
from dataclasses import dataclass
import io
import tokenize
from threading import RLock
import time

from .v3_harness import canonical, digest, relative


DEVELOP={'develop_measure','develop_profile','develop_quality'}


def need(ok,why):
    if not ok:raise ValueError(why)


def text_cost(src):
    tree=ast.parse(src); comment=sum(len(t.string) for t in tokenize.tokenize(io.BytesIO(src).readline) if t.type==tokenize.COMMENT)
    string_literals=sum(len(n.value) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) is str)
    largest=max((len(n.value) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) is str),default=0)
    return comment+string_literals,largest


@dataclass(frozen=True)
class Engine:
    digest: str
    edits: tuple


class SourceContract:
    def __init__(self, base, editable, *, max_files, max_bytes, max_text_growth, max_new_literal):
        self.base=dict(base);self.editable=frozenset(editable)
        self.limits=max_files,max_bytes,max_text_growth,max_new_literal
        need(all(type(n) is int and n>0 for n in self.limits),'edit contract caps required')
        need(self.editable and self.editable<=self.base.keys(),'invalid engine allowlist')
        for p in self.base:relative(p)
        for p in self.editable:need(p.endswith('.py'),'pilot accepts Python source, including inline Triton, only')
        need(all(type(x) is bytes for x in self.base.values()),'engine base must be exact bytes')
        self.baseline=digest(canonical({n:digest(x) for n,x in sorted(self.base.items())}))

    def from_args(self,args):
        need(set(args)=={'files'} and type(args['files']) is dict,'candidate requires exact changed source map')
        max_files,max_bytes,max_text,new_literal=self.limits
        files=args['files']; need(len(files)<=max_files and files.keys()<=self.editable,'candidate outside Python/Triton allowlist')
        checked=[];total=0;text=0
        for name,code in sorted(files.items()):
            need(type(code) is str,'source must be UTF-8 text')
            raw=code.encode();old=self.base[name]
            if raw==old:continue
            total+=len(raw);need(total<=max_bytes,'candidate too large')
            try:a,x=text_cost(raw);b,y=text_cost(old)
            except (SyntaxError,tokenize.TokenError,UnicodeError) as e:raise ValueError('invalid Python source') from e
            text+=max(0,a-b);need(text<=max_text and x<=max(y,new_literal),'new text exceeds declared budget')
            checked.append((name,raw))
        h=digest(canonical(dict(base=self.baseline,edits={name:digest(raw) for name,raw in checked})))
        return Engine(h,tuple(checked))


@dataclass(frozen=True)
class DevReceipt:
    engine: str
    operation: str
    status: str
    node_ms: int
    receipt_id: str
    data: dict


class Session:
    """One bounded operation lease. Dev workers are injected trusted callbacks.

    They receive an immutable byte artifact and a wall/node lease. They must run only
    the frozen development workload, never a task/tool name or path supplied by Pi.
    """
    def __init__(self,mode,lineage,contract,facts,worker,*,wall_ms,node_ms,dev_calls,reservations,clock=time.monotonic):
        need(mode in {'fixed_train','evolving_train','evaluation'},'unknown mode')
        need(type(wall_ms)is int and wall_ms>0 and type(node_ms)is int and node_ms>=0 and type(dev_calls)is int and dev_calls>=0,'invalid session limits')
        need(reservations and reservations.keys()<=DEVELOP and all(type(x)is int and x>0 for x in reservations.values()),'each operation requires a worst-case reservation')
        self.mode=mode;self.lineage=lineage;self.contract=contract;self.facts=facts;self.worker=worker
        self.clock=clock;self.deadline=clock()+wall_ms/1000;self.node_limit=node_ms;self.calls_left=dev_calls;self.reservations=reservations
        self.rows=[];self.qualified={};self.delivered=None;self.spent_node=0;self.serial=RLock();self.worker_invalid=False

    def event(self,operation,status,**info):
        row=dict(event_id=len(self.rows)+1,operation=operation,status=status,**info);self.rows.append(row);return row

    def __call__(self,op,args):
        with self.serial:
            if self.clock()>=self.deadline:
                self.event(op,'wall_expired');return 403,dict(status='wall_expired')
            try:
                status,data=self._handle(op,args)
                return status,data
            except (ValueError,PermissionError,KeyError,TypeError):
                self.event(op,'denied');return 403,dict(status='denied')

    def _handle(self,op,args):
        if op=='facts':
            need(not args,'facts require empty args');self.event(op,'ok',fact_sha=digest(canonical(self.facts)));return 200,dict(facts=self.facts)
        if op=='harness_status':
            self.lineage.assert_train(self.mode)
            need(not args,'Harness status requires empty args');r=dict(active=self.lineage.active().identity,versions=self.lineage.catalogue())
            self.event(op,'ok',active=r['active']);return 200,r
        if op=='harness_submit':
            self.lineage.assert_train(self.mode)
            need(set(args)=={'files','load'} and type(args['files']) is dict,'invalid Harness submission')
            files={k:v.encode() for k,v in args['files'].items() if type(v) is str};need(len(files)==len(args['files']),'invalid Harness bytes')
            version=self.lineage.submit(self.mode,files,args['load']);self.event(op,'stored',version=version);return 200,dict(version=version)
        if op=='harness_activate':
            self.lineage.assert_train(self.mode)
            need(set(args)=={'version','expected'},'activation must name exact versions')
            result=self.lineage.activate(self.mode,args['version'],args['expected'],actor='agent_rpc',request_id=f'rpc-{len(self.rows)+1}')
            self.event(op,'active',**result);return 200,result
        if op=='deliver_engine':
            c=self.contract.from_args(args);r=self.qualified.get(c.digest)
            need(r is not None and r.status=='valid','this exact artifact has no accepted Quality receipt')
            self.delivered=c;self.event(op,'selected_by_agent',engine=c.digest,receipt_id=r.receipt_id);return 200,dict(result='signed',engine=c.digest)
        if op not in self.reservations:
            self.event('unregistered','denied');return 403,dict(status='denied')
        return self._develop(op,self.contract.from_args(args))

    def _develop(self,op,candidate):
        reserve=self.reservations[op];remaining_ms=max(0,int(1000*(self.deadline-self.clock())))
        need(not self.worker_invalid and self.calls_left>0 and reserve<=self.node_limit-self.spent_node and reserve<=remaining_ms,'development operation budget unavailable')
        self.calls_left-=1
        try:
            r=self.worker(op,candidate,min(reserve,remaining_ms))
            need(isinstance(r,DevReceipt) and r.engine==candidate.digest and r.operation==op and type(r.node_ms)is int and 0<=r.node_ms<=reserve
                 and isinstance(r.receipt_id,str) and r.receipt_id and r.status in {'valid','violation','insufficient_evidence','observed'} and type(r.data)is dict,'unsettled development receipt')
            need(self.clock()<=self.deadline,'development expired')
        except (ValueError,OSError,TimeoutError,RuntimeError):
            self.worker_invalid=True;self.spent_node+=reserve;self.event(op,'insufficient_evidence',engine=candidate.digest,node_ms_charged=reserve)
            return 502,dict(status='insufficient_evidence')
        self.spent_node+=r.node_ms;self.event(op,r.status,engine=candidate.digest,node_ms_charged=r.node_ms,receipt_id=r.receipt_id)
        if op=='develop_quality':self.qualified[candidate.digest]=r
        return 200,dict(engine=candidate.digest,status=r.status,receipt_id=r.receipt_id,data=r.data)
