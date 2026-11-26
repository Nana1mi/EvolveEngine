import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_source import (
    E0_OVERLAY_LF_SHA256, E0_OVERLAY_SHA256, HELPER, LOADER, MODEL, SGLANG_ARCHIVE_SHA256,
    SGLANG_REVISION, Level3Engine, SourceContract, read_source_archive,
)
from evolveengine.v3_session import DevReceipt, Engine, Session

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = os.environ.get('EVOLVEENGINE_SGLANG_ARCHIVE')
PYTHON = 'python/sglang/srt/scheduler.py'
TRITON = 'python/sglang/kernels/jit/attention.triton'
CUDA = 'python/sglang/kernels/aot/csrc/gguf.cu'
CPP = 'python/sglang/kernels/aot/csrc/gguf.cpp'
HEADER = 'python/sglang/kernels/aot/include/shared.cuh'
ALIAS = 'python/sglang/kernels/jit/include/shared.cuh'
CMAKE = 'python/sglang/kernels/aot/CMakeLists.txt'
OLD = 'python/sglang/srt/old.py'
ASSET = 'assets/example.bin'
PROTECTED = 'benchmarks/golden.py'
BASE = {PYTHON: b'x = 1\r\n', CUDA: b'__global__ void k() {}\n', CMAKE: b'project(sglang)\n',
        HEADER: b'#pragma once\n', OLD: b'x = 2\n', PROTECTED: b'measure()\n', ASSET: b'\x00\xff'}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = SourceContract(BASE, links={ALIAS: HEADER}, modes={PYTHON: 0o755})

    def test_complete_multifile_candidate_round_trips_with_same_worker_identity(self):
        edits = {PYTHON: 'import triton\n', TRITON: 'def attn(): pass\n',
                 CUDA: '__global__ void optimized() {}\n', CPP: 'int value() { return 1; }\n',
                 HEADER: '#pragma once\n#define SHARED 1\n', CMAKE: 'project(optimized)\n'}
        args = {'files': edits, 'deleted': [OLD], 'build_args': {'jobs': 8,
                'cuda_architectures': ['100a'], 'cuda_fast_math': False, 'build_type': 'Release'}}
        engine = self.contract.from_args(args)
        self.assertIsInstance(engine, Engine)
        wire = self.contract.to_args(engine)
        worker = SourceContract(dict(reversed(list(BASE.items()))), links={ALIAS: HEADER}, modes={PYTHON: 0o755})
        self.assertEqual(engine, worker.from_args(wire))
        self.assertEqual(engine, worker.from_args({**wire, 'files': dict(reversed(list(edits.items())))}))
        self.assertNotEqual(engine, worker.from_args({**wire, 'build_args': {'jobs': 16}}))
        self.assertNotEqual(engine, worker.from_args({**wire, 'deleted': []}))
        effective, modes, dirs = worker.effective_tree(engine)
        self.assertEqual(modes[PYTHON], 0o755)
        self.assertEqual(modes[CPP], 0o644)
        self.assertIn('python/sglang/kernels/aot/csrc', dirs)
        self.assertEqual(effective, {**{p: b for p, b in BASE.items() if p != OLD},
                                    **{p: s.encode() for p, s in edits.items()}, ALIAS: edits[HEADER].encode()})
        with tempfile.TemporaryDirectory() as tmp:
            dest = worker.materialize(engine, Path(tmp) / 'candidate')
            on_disk = {p.relative_to(dest).as_posix(): p.read_bytes() for p in dest.rglob('*') if p.is_file()}
            self.assertEqual(effective, on_disk)
            self.assertFalse((dest / ALIAS).is_symlink())
            if os.name != 'nt':
                self.assertEqual(stat.S_IMODE((dest / PYTHON).stat().st_mode), 0o755)
            with self.assertRaises(FileExistsError):
                worker.materialize(engine, dest)

    def test_two_parent_branches_restore_independently_without_workspace_state(self):
        parent = self.contract.from_args({'files': {CPP: 'int f();', PYTHON: 'x=2'}, 'deleted': [OLD],
                                          'build_args': {'jobs': 5}})
        first = self.contract.from_parent(parent, {'files': {OLD: 'x = 2\n', CUDA: '// branch A'},
                                                    'deleted': [CPP]})
        second = self.contract.from_parent(parent, {'files': {PYTHON: 'x = 1\r\n', TRITON: '# branch B'},
                                                     'build_args': {'jobs': 2}})
        with tempfile.TemporaryDirectory() as tmp:
            worker = SourceContract(BASE, links={ALIAS: HEADER}, modes={PYTHON: 0o755})
            for n, expected, absent in (('a', first, (CPP, TRITON)), ('b', second, (OLD,))):
                lock = Path(tmp) / (n + '.json')
                lock.write_bytes(canonical(self.contract.to_args(expected)))
                restored = worker.from_args(json.loads(lock.read_bytes()))
                self.assertEqual(restored, expected)
                dest = worker.materialize(restored, Path(tmp) / n)
                self.assertFalse(any((dest / p).exists() for p in absent))
                (dest / 'unregistered.py').write_bytes(b'not in the checkpoint')
                clean = worker.materialize(restored, Path(tmp) / (n + '-restore'))
                self.assertFalse((clean / 'unregistered.py').exists())
            self.assertEqual(first.build_args, {'jobs': 5})
            self.assertEqual(second.build_args, {'jobs': 2})
            self.assertNotEqual(first.digest, second.digest)
            self.assertEqual(worker.effective(first)[OLD], BASE[OLD])
            self.assertEqual(worker.effective(second)[PYTHON], BASE[PYTHON])

    def test_identity_rejects_changed_baseline_or_forged_artifact(self):
        e0 = self.contract.from_args({'files': {}})
        self.assertEqual(e0, self.contract.from_args({'files': {PYTHON: 'x = 1\r\n'}}))
        different = SourceContract({**BASE, ASSET: b'alt'}, links={ALIAS: HEADER}, modes={PYTHON: 0o755})
        with self.assertRaisesRegex(ValueError, 'digest'):
            different.effective(e0)
        for forged in (Level3Engine('0' * 64, ()),
                       Level3Engine(e0.digest, ((PYTHON, b'\xff'),)),
                       Level3Engine(e0.digest, ((PYTHON, b'x=1'), (PYTHON, b'x=1'))),
                       Level3Engine(e0.digest, (), _build_json=b'{ }')):
            with self.subTest(forged=forged), self.assertRaises(ValueError):
                self.contract.effective(forged)
        with self.assertRaises(TypeError):
            self.contract.base[PYTHON] = b'invalid'

    def test_rejects_protected_binary_nonportable_and_conflicting_paths(self):
        files = ['model/Tokenizer.JSON', 'model/weights.gguf', 'trusted/tune.json', ASSET, PROTECTED, '.git/hooks/pre_commit.py', '.GITHUB/workflows/new.py',
                 'evolveengine/v3_provider.py', 'python/judge/gate.py', ALIAS,
                 '../outside.py', 'python/./entry.py', '/entry.py', 'C:/entry.py',
                 'python\\escape.py', 'a//b.py', 'a/aux.py', 'a/com¹.py', 'a/b.py:ads',
                 'a./b.py', 'a /b.py', 'PYTHON/sglang/new.py', PYTHON.upper(), PYTHON + '/x.py']
        for path in files:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.contract.from_args({'files': {path: 'x=1'}})
        for bad in ({'files': {'One/x.py': '', 'one/y.py': ''}},
                    {'files': {'n.py': '', 'n.py/x.py': ''}},
                    {'files': {PYTHON: 'bad\0data'}}, {'files': {PYTHON: '\ud800'}},
                    {'files': {PYTHON: 'x=0'}, 'deleted': [PYTHON]},
                    {'files': {}, 'deleted': [OLD, OLD]}, {'files': {}, 'deleted': [HEADER]},
                    {'files': {}, 'deleted': ['nonexistent.py']}, {'files': {}, 'deleted': [ASSET]},
                    {'files': {}, 'command': 'sh'},
                    {'files': {}, 'build_args': {'cmake_args': ['-P a.cmake']}},
                    {'files': {}, 'build_args': {'env': {'PATH': 'elsewhere'}}},
                    {'files': {}, 'build_args': {'cuda_architectures': ['100;message()']}},
                    {'files': {}, 'build_args': {'jobs': True}}):
            with self.subTest(args=bad), self.assertRaises(ValueError):
                self.contract.from_args(bad)
        bounded = SourceContract(BASE, max_files=1, max_bytes=4)
        for files in ({PYTHON: '01234'}, {PYTHON: '', CPP: ''}, {CPP: '你你'}):
            with self.assertRaises(ValueError):
                bounded.from_args({'files': files})

    def test_unknown_text_formats_round_trip_without_a_suffix_allowlist(self):
        edits = {'kernels/tuning.gb10': 'tiles: [64, 128]', 'rust/CONFIG': 'warp=4'}
        engine = self.contract.from_args({'files': edits})
        self.assertEqual(self.contract.from_args(self.contract.to_args(engine)), engine)
        for path, text in edits.items():
            self.assertEqual(self.contract.effective(engine)[path], text.encode())

    def test_new_or_modified_config_cannot_project_into_protected_directory(self):
        c = SourceContract({'kernels/cfg/a.json':b'{}'}, links={'judge/data':'kernels/cfg'})
        for path in ('kernels/cfg/a.json','kernels/cfg/new.yaml'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'protected projected'):
                c.from_args({'files':{path:'different'}})

    def test_existing_session_quality_and_delivery_use_the_same_candidate(self):
        def worker(op, engine, _lease):
            self.assertIsInstance(engine, Level3Engine)
            self.assertEqual(self.contract.effective(engine)[CPP], b'int f();\n')
            return DevReceipt(engine.digest, op, 'valid', 3, 'external-evidence', {})

        session = Session('evaluation', None, self.contract, {}, worker, wall_ms=10000,
                          node_ms=100, dev_calls=1, reservations={'develop_quality': 20})
        args = {'files': {CPP: 'int f();\n'}, 'build_args': {'jobs': 5}}
        self.assertEqual(session('develop_quality', args)[1]['status'], 'valid')
        self.assertEqual(session('deliver_engine', {**args, 'build_args': {'jobs': 6}})[0], 403)
        self.assertEqual(session('deliver_engine', args)[1]['result'], 'signed')
        self.assertEqual(session.delivered, self.contract.from_args(args))
        self.assertEqual(session.spent_node, 3)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, entries):
        file = self.root / 'source.tar.gz'
        with tarfile.open(file, 'w:gz') as t:
            for name, kind, value in [('root', tarfile.DIRTYPE, None), *entries]:
                e = tarfile.TarInfo(name)
                e.type = kind
                e.mode = 0o100755
                if kind == tarfile.REGTYPE:
                    e.size = len(value)
                    t.addfile(e, io.BytesIO(value))
                else:
                    e.linkname = value or ''
                    t.addfile(e)
        return file, digest(file.read_bytes())

    def test_portable_file_and_directory_aliases_and_empty_directories(self):
        file, sha = self.archive([
            ('root/empty', tarfile.DIRTYPE, None), ('root/pkg/a.py', tarfile.REGTYPE, b'x=1\r\n'),
            ('root/pkg/include/h.h', tarfile.REGTYPE, b'\xff'),
            ('root/alias', tarfile.SYMTYPE, 'pkg'), ('root/pkg/b.py', tarfile.SYMTYPE, './a.py'),
        ])
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            read_source_archive(file, expected_sha256='a' * 64, expected_root='root')
        i = read_source_archive(file, expected_sha256=sha, expected_root='root')
        c = SourceContract(i.files, modes=i.modes, links=i.links, directories=i.directories)
        self.assertEqual(c.base['alias/b.py'], b'x=1\r\n')
        self.assertEqual(c.modes['alias/b.py'], 0o755)
        e = c.from_args({'files': {'pkg/a.py': 'x=2\n', 'pkg/new.py': 'x=3\n'}})
        dest = c.materialize(e, self.root / 'candidate')
        self.assertTrue((dest / 'empty').is_dir())
        self.assertEqual((dest / 'alias/b.py').read_bytes(), b'x=2\n')
        self.assertEqual((dest / 'alias/new.py').read_bytes(), b'x=3\n')
        self.assertEqual((dest / 'alias/include/h.h').read_bytes(), b'\xff')

    def test_invalid_archive_members_and_links_fail_closed(self):
        regular = ('root/pkg/a.py', tarfile.REGTYPE, b'x=0')
        cases = [
            [('other/file.py', tarfile.REGTYPE, b'x')], [('root/../outside.py', tarfile.REGTYPE, b'x')],
            [('root/pkg/b.py', tarfile.SYMTYPE, '../../outside.py')],
            [('root/pkg/b.py', tarfile.SYMTYPE, '/outside.py')],
            [('root/pkg/b.py', tarfile.SYMTYPE, 'missing.py')],
            [('root/pkg/b.py', tarfile.LNKTYPE, 'pkg/a.py')],
            [('root/pkg/dev', tarfile.CHRTYPE, None)],
            [('root/pkg/b.py', tarfile.SYMTYPE, 'c.py'), ('root/pkg/c.py', tarfile.SYMTYPE, 'b.py')],
            [('root/pkg/self', tarfile.SYMTYPE, '.')], [('root/Pkg/b.py', tarfile.REGTYPE, b'x')],
            [('root/pkg/a.py/b.py', tarfile.REGTYPE, b'x')], [regular],
        ]
        for entries in cases:
            with self.subTest(entries=entries):
                file, sha = self.archive([regular, *entries])
                with self.assertRaises(ValueError):
                    read_source_archive(file, expected_sha256=sha, expected_root='root')

    def test_git_checkout_line_endings_do_not_change_the_overlay_lock(self):
        target = self.root / 'experiments' / 'spark26_gguf'
        target.mkdir(parents=True)
        for ending in (b'\n', b'\r\n'):
            for name in E0_OVERLAY_SHA256:
                data = (ROOT / 'experiments/spark26_gguf' / name).read_bytes().replace(b'\r\n', b'\n')
                (target / name).write_bytes(data.replace(b'\n', ending))
            with self.subTest(ending=ending), self.assertRaises(FileNotFoundError):
                SourceContract.acquire_spark26_e0(self.root / 'absent', self.root)

    def test_tampered_overlay_rejected_before_archive_or_script_execution(self):
        target = self.root / 'experiments' / 'spark26_gguf'
        target.mkdir(parents=True)
        for name in E0_OVERLAY_SHA256:
            (target / name).write_bytes((ROOT / 'experiments/spark26_gguf' / name).read_bytes())
        (target / 'apply_overlay.py').write_bytes(b'raise RuntimeError("must not run")')
        with self.assertRaisesRegex(ValueError, 'overlay hash mismatch'):
            SourceContract.acquire_spark26_e0(self.root / 'absent', self.root)


@unittest.skipUnless(ARCHIVE, 'set EVOLVEENGINE_SGLANG_ARCHIVE to run exact local E0 acquisition')
class PinnedArchiveTest(unittest.TestCase):
    def test_real_offline_acquisition_and_cli_materialization(self):
        contract = SourceContract.acquire_spark26_e0(ARCHIVE, ROOT)
        self.assertEqual(contract.provenance['revision'], SGLANG_REVISION)
        self.assertEqual(contract.provenance['archive_sha256'], SGLANG_ARCHIVE_SHA256)
        self.assertEqual(len(contract.inventory_manifest()['physical_files']), 12_812)
        self.assertEqual(len(contract.base), 12_941)
        self.assertEqual(len(contract.links), 8)
        self.assertIn(b'strict_gguf=True', contract.base[LOADER])
        self.assertIn(b'Unrecognized Spark GGUF parameter', contract.base[MODEL])
        self.assertEqual(digest(contract.base[HELPER]), E0_OVERLAY_LF_SHA256['spark2_5_gguf.py'])
        alias, target = next((a, p) for a, p in contract.links.items() if a.endswith('vecdotq.cuh'))
        new = contract.base[target].decode() + '\n// test checked alias\n'
        e = contract.from_args({'files': {target: new}})
        self.assertEqual(contract.effective(e)[alias], new.encode())
        with tempfile.TemporaryDirectory(prefix='evol3-') as tmp:
            dest, manifest = Path(tmp) / 'src', Path(tmp) / 'inventory.json'
            result = subprocess.run([sys.executable, str(ROOT / 'experiments/spark26_gguf/level3/prepare_source.py'),
                                     ARCHIVE, str(dest), '--manifest', str(manifest)],
                                    check=True, capture_output=True, text=True, timeout=90)
            self.assertIn(contract.from_args({'files': {}}).digest, result.stdout)
            self.assertEqual(manifest.read_bytes(), canonical(contract.inventory_manifest()) + b'\n')
            actual = {p.relative_to(dest).as_posix(): digest(p.read_bytes()) for p in dest.rglob('*') if p.is_file()}
            self.assertEqual(actual, {p: digest(b) for p, b in contract.base.items()})


if __name__ == '__main__':
    unittest.main()
