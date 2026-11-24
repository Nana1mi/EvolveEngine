"""Six fixed, single-session reproductions; no retry or censored outcomes. Mock provider only."""
import argparse, json, tempfile
from pathlib import Path
from check import session


def diagnose():
    attempts=[]
    for i in range(6):
        trace={};row={'i':i,'trace':trace};attempts.append(row)
        with tempfile.TemporaryDirectory(prefix='ee-s1-diagnostic-') as tmp:
            try:
                session(Path(tmp),'s1-B2',trace)
                row['status']='completed'
            except (ValueError, OSError, RuntimeError) as ex:
                row.update(status='not_completed',reason=str(ex))
    return dict(scope='S1-only fixed six diagnostic replays; not the five-session P2 trajectory',attempts=attempts)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);p=parser.parse_args().path
    if p.exists():raise SystemExit('receipt exists')
    doc=diagnose()
    with p.open('x',encoding='utf8') as out:json.dump(doc,out,indent=2,ensure_ascii=False)
    print([(r['status'],r['trace'].get('assistant_errors',[])) for r in doc['attempts']])
