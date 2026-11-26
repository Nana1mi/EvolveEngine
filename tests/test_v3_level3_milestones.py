from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from tempfile import TemporaryDirectory

from evolveengine import v3_level3_sampling as sampling, v3_level3_scoring as scoring, v3_level3_trajectory as trajectory
from evolveengine.v3_harness import canonical,digest
from evolveengine.v3_level3_chat_timing import bounds,load_chat,SCHEMA,ChatSchedule
from evolveengine.v3_level3_source import SourceContract
from evolveengine.v3_level3_startup import PATH, options
from evolveengine.v3_level3_deploy import launch_args
from evolveengine.v3_level3_run import Run


class MilestoneTests(TestCase):
    def test_runtime_declaration_inherits_changes_identity_and_preserves_locked_args(self):
        c=SourceContract({'python/foo.py':b'# data\n'}); a=c.from_args({'files':{PATH:'{"radix-cache":true,"cuda-graph-bs-decode":[1,4],"mem-fraction-static":0.6}'}})
        b=c.from_parent(a,{'files':{'python/foo.py':'# local\n'}}); raw=dict(b.edits)[PATH];self.assertEqual(dict(a.edits)[PATH],raw)
        self.assertEqual(c.from_args(c.to_args(b)).digest,b.digest)
        base=SimpleNamespace(launch_profile='graph_defaults'); args=launch_args(base,SimpleNamespace(runtime_config=raw))
        self.assertNotIn('--disable-radix-cache',args);self.assertIn('--disable-radix-cache',launch_args(base))
        for flag,value in (('--stream-interval','1'),('--context-length','8192'),('--mem-fraction-static','0.6')):
            self.assertEqual(args.count(flag),1);self.assertEqual(args[args.index(flag)+1],value)
        for raw in [b'{"radix-cache":1}',b'{"radix-cache":true,"radix-cache":false}',b'{"port":4000}',b'{"stream-interval":3}',b'{"context-length":4096}',b'{"attention-backend":"--other"}']:
            with self.subTest(raw=raw),self.assertRaises(ValueError): options(raw)

    def test_replay_advisory_floor_not_perf_gate(self):
        tokens={s:{'input':[1,2]} for s in scoring.WEIGHTS}; changed=deepcopy(tokens);changed['W1']['input']=[1,3,4]
        agreement=scoring.replay_agreement(tokens,changed);self.assertEqual(agreement['W1']['rate'],1/3)
        arm=dict(paired_replay_agreement=agreement,timed_request_outputs=tokens);pairs=[(arm,arm)]*scoring.MIN_PAIRS
        floor=scoring.replay_audit_floors(pairs);self.assertAlmostEqual(floor['W1'],1/3-.10);self.assertEqual(floor['W3'],.9)
        with self.assertRaises(ValueError): scoring.replay_audit_floors(pairs[1:])
        policy=scoring.natural_policy('1'*64,'2'*64,{k:.02 for k in scoring.WEIGHTS},chat_corpus='3'*64,replay_floors=floor)
        self.assertEqual(scoring.resolve_policy(policy),policy)

    def test_milestone_claims_at_most_two_looks_and_final_draws_are_fresh(self):
        run=Run.__new__(Run);run.tiered_sampling=True;run.engines={'e':'engine'};run.sample_screens={};seq=[]
        decisions=iter(['inconclusive','pass','different']);run._event=lambda *a,**kw:None
        def draw(op,engine,sample,sample_build):
            seq.append((sample['milestone'],sample['stage'],sample['role'],sample_build));i=len(seq)
            d={'decision':'reference'} if i%2 else {'decision':next(decisions),'reference':str(i-1)}
            return dict(evidence=str(i),data=d,status='sampled')
        run.develop=draw; chosen=SimpleNamespace(engine='e',build='b')
        best=run.milestone_screen(chosen,'historical_best');self.assertEqual(best['decision'],'pass');self.assertEqual(len(seq),4)
        self.assertEqual(run.milestone_screen(chosen,'historical_best'),best);self.assertEqual(len(seq),4)
        self.assertEqual(run.milestone_screen(chosen,'final_nomination')['decision'],'different');self.assertEqual(len(seq),6)
        self.assertTrue(run.sampling_rejected('e'));self.assertFalse(run.sampling_rejected('other'))
        self.assertEqual(len(set(seq)),len(seq));self.assertEqual(run.milestone_screen(SimpleNamespace(engine='e',build='new'),'final_nomination')['decision'],'already_attempted_other_build')

    def test_sampling_distinguishes_positive_violation_from_no_equivalence(self):
        ref={str(c):[[1,2,3]]*sampling.SAMPLES for c in sampling.CASES};bad={k:[[9]]*len(v) for k,v in ref.items()}
        self.assertEqual(sampling.decision(sampling.compare(ref,ref,alpha=.025)),'pass')
        self.assertEqual(sampling.decision(sampling.compare(ref,bad,alpha=.025)),'different')
        borderline={k:[[9]]*180+v[180:] for k,v in ref.items()}
        self.assertEqual(sampling.decision(sampling.compare(ref,borderline,alpha=.025)),'inconclusive')

    def test_e0_margin_clips_not_observation(self):
        r=dict(protocol=trajectory.VERSION,maxima=dict.fromkeys(trajectory.METRICS,.03),reference_quantum=dict.fromkeys(trajectory.METRICS,.001))
        self.assertEqual(trajectory.envelope([r,r])['decode_tv'],.05);r['maxima']['decode_tv']=.051
        with self.assertRaisesRegex(ValueError,'observed E0'):trajectory.envelope([r,r])

    def test_generalization_timed_waves_have_genuinely_different_lengths(self):
        shapes={'W1':[(128,256)],'W3':[(4096,128)],'W4':[(128,96),(512,96)]};rows={};uid=0
        for scene,shape in shapes.items():
            rows[scene]=[]
            for i in range(4):
                w=[]
                for nominal,_ in shape:
                    uid+=1; n=bounds(scene,nominal,'generalization',i)[0]
                    w.append(dict(ids=[uid]*n,source='synthetic',source_id=str(uid),license='test',task='fixture'))
                rows[scene].append(w)
        doc=dict(requests=rows,split='generalization',schema=SCHEMA,screening=None,enable_thinking=True,tokenizer_sha256='1'*64,template_sha256='2'*64)
        with TemporaryDirectory() as tmp:
            path=(Path(tmp)/'synthetic.json').resolve()
            def load():
                raw=canonical(doc)+b'\n';path.write_bytes(raw)
                return load_chat(dict(path=str(path),sha256=digest(raw)),'1'*64,'2'*64,shapes,1,2,split='generalization')
            waves=load();sched=ChatSchedule(waves,1,2,'a'*64);a,_=sched.get(1);b,_=sched.get(1,replay=True)
            self.assertEqual(a['W1'][1:],list(reversed(b['W1'][1:])))
            self.assertNotEqual(len(a['W1'][1][0]),len(a['W1'][2][0]));rows['W1'][2][0]['ids']=[9]*128
            with self.assertRaises(ValueError):load()

    def test_export_rejects_final_draw_from_wrong_service_or_wrong_build(self):
        import sqlite3,json
        from evolveengine.schema import digest as obj
        from evolveengine.v3_level3_delivery import _sampling_proof
        from evolveengine.journal import Journal
        chosen=dict(engine='candidate',build='build');samples={str(c):[[1,2,3]]*sampling.SAMPLES for c in sampling.CASES}
        report=dict(protocol=sampling.VERSION,decision='pass',receipts=['ref','trial']);jobs={}
        with TemporaryDirectory() as tmp,sqlite3.connect(':memory:') as db:
            db.execute('create table objects(id text,kind text,content text)')
            for i,pin in enumerate(report['receipts']):
                data=dict(protocol=sampling.VERSION,challenge=dict(milestone='final_nomination',role='trial' if i else 'reference',stage=1),
                          service=str(i),observed_engine='candidate' if i else 'E0',decision='pass' if i else 'reference')
                if i:data.update(test=sampling.compare(samples,samples,alpha=.025),reference='ref')
                row=dict(engine='candidate',build='build',operation='develop_quality',status='sampled',node_ms=5,evidence=pin,data=data);jobs[pin]=row
                db.execute('insert into objects values(?,?,?)',(obj(dict(kind='level3_worker_result',content=row)),'level3_worker_result',json.dumps(row)))
            self.assertEqual(_sampling_proof(db,jobs,chosen,'E0','final_nomination',report),report)
            for owner in ('other-build','other-engine'):
                with self.assertRaises(ValueError): _sampling_proof(db,jobs,dict(chosen,build=owner),'E0','final_nomination',report)
            jobs['trial']['data']['service']='0'
            with self.assertRaises(ValueError):_sampling_proof(db,jobs,chosen,'E0','final_nomination',report)

    def test_admission_draft_cannot_release_on_partial_or_wrong_control_receipt(self):
        import subprocess, sys
        driver=Path(__file__).resolve().parents[1]/'experiments/spark26_gguf/level3/draft_development_admission.py'
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'controls').mkdir();target=root/'draft'
            call=[sys.executable,str(driver),str(root),str(target),'--expected-acceptance','0'*64,
                  '--remote','/var/tmp/ee-dev','--judge-deps','/var/tmp/dev-deps',
                  '--split-root',str(root),'--split-controls-sha256','1'*64,'--worker-root','/var/tmp/ee-split']
            for malformed in (False,True):
                if malformed: (root/'controls/acceptance.json').write_text('{}')
                proc=subprocess.run(call,capture_output=True,timeout=30)
                self.assertNotEqual(proc.returncode,0);self.assertFalse(target.exists())
                self.assertIn(b'unexpected engineering acceptance' if malformed else b'acceptance.json',proc.stderr)

    def test_controls_do_not_promote_a_partial_or_unstable_baseline(self):
        from experiments.spark26_gguf.level3.accept_development import checked_noise,verify
        with self.assertRaises((KeyError,ValueError)): checked_noise({'receipts':[]},{},'a','b')
        with TemporaryDirectory() as tmp,self.assertRaises((KeyError,ValueError)):
            verify(Path(tmp),{'schema':'older'},reg='a',trajectory_pin='b',noise_pin='c',graph_pin='d')
