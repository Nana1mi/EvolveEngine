"""Dispatch ONE released P1 trial. No scheduler, independent scoring, or automatic retry."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

from evolveengine.deepseek import Provider
from evolveengine.pi import PiAgent
from evolveengine.schema import digest
from evolveengine.workspace import Repository
from .release import balanced_schedule, read_registered, require, verified_cards
from .trial import run_trial


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol_hash():
    import evolveengine.pi as pi
    import evolveengine.deepseek as ds
    modules=[Path(__file__).with_name(n) for n in ('trial.py','release.py','dispatch.py')]
    modules += [Path(pi.__file__),Path(ds.__file__),Path(pi.__file__).with_name('pi_guard.ts')]
    return digest({p.name:sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in modules})


def dispatch(expected, manifest, reviewer_sha, evidence, config, cards, output, ordinal):
    require(expected.get('executor_protocol_sha256')==protocol_hash(),'Executor source changed')
    ids=verified_cards(manifest,reviewer_sha,evidence,expected)
    seq=balanced_schedule(ids)
    require(type(ordinal) is int and 1<=ordinal<=len(seq),'Invalid registered slot')
    key,arm,repeat=seq[ordinal-1];spec=expected['cards'][key]
    require(digest(config)==expected['controller_digest'] and config['engine_revision']==expected['engine_sha'],
            'Wrong controller or engine')
    require(config['pi_version']==expected['pi_version'] and config['provider']['model']==expected['model'],'Model changed')
    card=read_registered(cards,spec['public_file'],spec['card_sha256'])
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    release=sha256(manifest).hexdigest();marker=root/'release.sha256'
    if not marker.exists():
        try:
            with marker.open('x') as f:f.write(release+'\n')
        except FileExistsError:pass
    require(marker.read_text().strip()==release,'Different study uses this output directory')
    # Reserving (not just completing) a slot is durable. A failed slot is never silently retried.
    dest=root/f'{ordinal:02}-{key}-{arm}-{repeat}';dest.mkdir(exist_ok=False)
    row={'slot':ordinal,'card_id':key,'arm':arm,'repeat':repeat,'release_sha256':release,'state':'reserved'}
    (dest/'reservation.json').write_text(json.dumps(row,indent=2)+'\n')
    repo=Repository(config['engine_checkout'],config['engine_revision'],config['editable'])
    agent=PiAgent(config['pi_command'],config['pi_version'],Provider(**config['provider']))
    # Pin the local CLI without making an extra paid chat request.
    import subprocess
    require(subprocess.check_output([*agent.command,'--version'],text=True,timeout=15).strip()==agent.version,'Pi version mismatch')
    trial=run_trial(agent,repo,{'base':config['engine_revision'],'edits':{}},card_bytes=card,
                    card_sha256=spec['card_sha256'],arm=arm,scope=tuple(spec['public_scope']))
    (dest/'submission.json').write_text(json.dumps({**row,'trial':trial,'state':'collected_unscored'},
                                        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('expected','manifest','reviewer-sha256','evidence','config','cards','output'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--slot',type=int,required=True);a=p.parse_args()
    print(dispatch(read(a.expected),Path(a.manifest).read_bytes(),a.reviewer_sha256,a.evidence,read(a.config),a.cards,a.output,a.slot))
