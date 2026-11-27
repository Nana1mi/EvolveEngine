"""Pinned, offline Level 3 source inventory and candidate identity.

Materialization is data-only. Candidate code and build scripts remain untrusted and
must only run with the Native worker's isolation; this contract is not that boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import posixpath
import re
import tarfile
import tempfile
from types import MappingProxyType
import unicodedata

from .v3_harness import canonical, digest
from .v3_session import Engine as SessionEngine

SGLANG_REVISION = 'b2ce7f8811bc40fd0d67000030c22edbbe0f2a37'
SGLANG_ARCHIVE_SHA256 = 'e3a71cd04d9513f47e25ffa91644b130b4b20adf0b21554a0b42a78b5433c4ae'
ARCHIVE_ROOT = f'sglang-{SGLANG_REVISION}'
E0_OVERLAY_SHA256 = MappingProxyType({
    'apply_overlay.py': 'd29c7bac1f0f98f5557f0aa8a53020b61c4184fd2fac228b8ccb0c4c99edeb5b',
    'spark2_5_gguf.py': '61e02ae9838a79ee0165c3ff05eafb340f7148dcc3ba3f5b27907b52099957a3',
})
# SHA256SUMS records the historical byte exports (mixed EOL). This second
# lock admits the same Python source from LF/CRLF Git checkouts; the E0
# overlay itself reads and writes these sources using universal newlines.
E0_OVERLAY_LF_SHA256 = MappingProxyType({
    'apply_overlay.py': '4295db08cb397d1a1fd42cbb272392193ac5e4d63ae523051c74fd2322e23425',
    'spark2_5_gguf.py': '4dc1c3ada1d915702631a58fd10e073a4d7ecefc14ddf34e011d3101c14b413d',
})
LOADER = 'python/sglang/srt/model_loader/loader.py'
MODEL = 'python/sglang/srt/models/spark2_5.py'
HELPER = 'python/sglang/srt/model_loader/spark2_5_gguf.py'
MAX_ARCHIVE_BYTES = 128 << 20
MAX_TREE_BYTES = 512 << 20
MAX_TREE_ENTRIES = 50_000
MAX_FILE_BYTES = 32 << 20
MAX_EDITS = 2_000
MAX_EDIT_BYTES = 32 << 20
_SCHEMA = 'sglang-level3-source-v1'
_WINDOWS_RESERVED = re.compile(r'(?:con|prn|aux|nul|conin\$|conout\$|(?:com|lpt)[0-9¹²³])\Z', re.I)
# Inputs are UTF-8 regular files inside the isolated source tree. Model assets
# are not engine configuration; the service mounts the registered ones read-only.
_MODEL_ASSETS = frozenset({
    'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
    'added_tokens.json', 'chat_template.jinja', 'merges.txt', 'vocab.json', 'vocab.txt',
    'model.safetensors.index.json', 'pytorch_model.bin.index.json',
})
_PROTECTED = frozenset({
    '.git', '.gitmodules', '.github', '.claude', '.agents', '.codex', '.evolveengine',
    'evolveengine', 'trusted', 'protected', 'judge', 'judges', 'evaluator', 'evaluators',
    'benchmark', 'benchmarks', 'ledger', 'accounting', 'sealed',
})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _path(path):
    _need(type(path) is str and 0 < len(path) <= 512 and path == unicodedata.normalize('NFC', path),
          'invalid or noncanonical source path')
    parts = path.split('/')
    _need(len(parts) <= 48, 'source path is too deep')
    for part in parts:
        _need(part and part not in {'.', '..'} and not part.endswith(('.', ' '))
              and not _WINDOWS_RESERVED.fullmatch(part.split('.', 1)[0].rstrip(' '))
              and not any(ord(c) < 32 or ord(c) == 127 or c in '\\:<>"|?*' for c in part),
              f'unsafe or nonportable source path: {path!r}')
        try:
            _need(len(part.encode('utf-8')) <= 255, 'source path component is too long')
        except UnicodeError as e:
            raise ValueError('source path is not valid UTF-8') from e
    return path


def _editable(path):
    parts = path.casefold().split('/')
    return (not path.casefold().startswith('experiments/spark26_gguf/')
            and not _PROTECTED.intersection(parts) and parts[-1] not in _MODEL_ASSETS
            and not parts[-1].endswith(('.gguf', '.safetensors')))


def _children(files, links, directories):
    """Validate the complete namespace, including implicit parent directories."""
    types = {}
    folded = {}
    children = {}

    def add(name, kind):
        other = folded.setdefault(name.casefold(), name)
        _need(other == name, f'case-colliding source paths: {other!r}, {name!r}')
        _need(types.setdefault(name, kind) == kind, f'file/directory/link collision: {name!r}')
        parent = name.rpartition('/')[0]
        if parent and parent not in types:
            add(parent, 'directory')
        _need(not parent or types[parent] == 'directory', f'non-directory source parent: {parent!r}')
        children.setdefault(parent, set()).add(name)

    for paths, kind in ((directories, 'directory'), (files, 'file'), (links, 'link')):
        for path in paths:
            add(_path(path), kind)
    return types, children


def _project(files, modes, links, directories):
    """Expand internal aliases, retaining link semantics without OS symlink privileges."""
    types, children = _children(files, links, directories)
    projected, permissions, dirs = {}, {}, set()
    total = 0

    def emit(source, target, stack):
        nonlocal total
        _need(source not in stack, f'cyclic source link at {source!r}')
        _need(len(stack) < 96, 'source link chain too deep')
        kind = types.get(source)
        _need(kind is not None, f'dangling source link: {source!r}')
        if kind == 'link':
            emit(links[source], target, (*stack, source))
        elif kind == 'file':
            projected[target] = files[source]
            permissions[target] = modes[source]
            total += len(files[source])
        else:
            dirs.add(target)
            for child in children.get(source, ()):
                emit(child, target + '/' + child.rsplit('/', 1)[-1], (*stack, source))
        _need(len(projected) + len(dirs) <= MAX_TREE_ENTRIES and total <= MAX_TREE_BYTES,
              'portable source projection exceeds inventory limits')

    for name in children.get('', ()):
        emit(name, name, ())
    _children(projected, {}, dirs)  # also validate paths expanded through directory aliases
    return projected, permissions, dirs


@dataclass(frozen=True)
class SourceInventory:
    """Physical files, relative link targets and normalized executable metadata."""
    files: Mapping[str, bytes]
    modes: Mapping[str, int]
    links: Mapping[str, str]
    directories: frozenset[str]
    archive_sha256: str


def _mode(mode):
    return 0o755 if mode & 0o111 else 0o644


def read_source_archive(archive_path, *, expected_sha256: str, expected_root: str) -> SourceInventory:
    """Verify the actual bytes before parsing; never use tar extraction or the network."""
    _need(type(expected_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', expected_sha256),
          'expected archive SHA-256 must be lowercase hex')
    _need('/' not in _path(expected_root), 'archive root must be one path component')
    with Path(archive_path).open('rb') as f:
        raw = f.read(MAX_ARCHIVE_BYTES + 1)
    _need(len(raw) <= MAX_ARCHIVE_BYTES, 'compressed source archive too large')
    _need(digest(raw) == expected_sha256, 'source archive SHA-256 mismatch')
    files, modes, links, dirs, seen = {}, {}, {}, set(), set()
    byte_count = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r|gz') as tar:
        for i, member in enumerate(tar, 1):
            _need(i <= MAX_TREE_ENTRIES, 'too many archive members')
            name = _path(member.name)
            _need(name not in seen, f'duplicate archive member: {name!r}')
            seen.add(name)
            if name == expected_root:
                _need(member.isdir(), 'archive root is not a directory')
                continue
            _need(name.startswith(expected_root + '/'), 'unexpected source archive root')
            path = name[len(expected_root) + 1:]
            if member.isdir():
                dirs.add(path)
            elif member.isfile() and not member.issparse():
                _need(0 <= member.size <= MAX_FILE_BYTES, 'invalid or oversized archive member')
                byte_count += member.size
                _need(byte_count <= MAX_TREE_BYTES, 'source archive expands beyond inventory limit')
                with tar.extractfile(member) as entry:
                    data = entry.read(member.size)
                _need(len(data) == member.size, 'incomplete archive member')
                files[path], modes[path] = data, _mode(member.mode)
            elif member.issym():
                target = member.linkname
                _need(target and not target.startswith('/') and '\\' not in target and ':' not in target
                      and not any(ord(c) < 32 for c in target), 'unsafe source archive link')
                links[path] = _path(posixpath.normpath(posixpath.join(posixpath.dirname(path), target)))
            else:
                raise ValueError('unsupported source archive entry (hard link, device, or sparse file)')
    _need(files and expected_root in seen, 'source archive is empty or root entry is missing')
    _project(files, modes, links, dirs)
    return SourceInventory(MappingProxyType(files), MappingProxyType(modes), MappingProxyType(links),
                           frozenset(dirs), expected_sha256)


def _build_args(value):
    """The worker maps these values into its fixed build recipe, never a shell command."""
    _need(type(value) is dict, 'build_args must be an object')
    choices = {
        'build_type': {'Release', 'RelWithDebInfo', 'MinSizeRel', 'Debug'},
        'cxx_optimization': {'0', '1', '2', '3', 's'},
        'cuda_optimization': {'0', '1', '2', '3'},
    }
    out = {}
    for key, v in value.items():
        if key in choices:
            ok = type(v) is str and v in choices[key]
        elif key == 'cuda_fast_math':
            ok = type(v) is bool
        elif key == 'jobs':
            ok = type(v) is int and 1 <= v <= 128
        elif key == 'cuda_architectures':
            ok = (type(v) is list and 1 <= len(v) <= 16
                  and all(type(a) is str and re.fullmatch(r'[1-9][0-9]{1,2}a?', a) for a in v)
                  and len(set(v)) == len(v))
            if ok:
                v = v.copy()
        else:
            raise ValueError(f'unsupported build argument: {key!r}')
        _need(ok, f'invalid build argument: {key!r}')
        out[key] = v
    return canonical(out)


def _manifest(files, modes):
    return {p: {'sha256': digest(data), 'size': len(data), 'mode': modes[p]} for p, data in files.items()}


def _is_utf8(data):
    try:
        data.decode('utf-8')
    except UnicodeError:
        return False
    return b'\0' not in data


@dataclass(frozen=True)
class Level3Engine(SessionEngine):
    """Immutable edits, deletes and build parameters, compatible with v3 Session."""
    deleted: tuple[str, ...] = ()
    _build_json: bytes = b'{}'

    @property
    def build_args(self) -> dict:
        """A copy of the normalized structured arguments."""
        return json.loads(self._build_json)


class SourceContract:
    """Immutable baseline plus a host/worker-replayable candidate protocol.

    ``base`` maps portable relative paths to exact bytes, including read-only assets
    and projected archive links. Submit edits to the canonical target of a link, not
    to its alias. Both views are regenerated on materialization. ``destination``
    must be fresh and its parent must be trusted (not writable by an Agent).
    """

    def __init__(self, base: Mapping[str, bytes], *, modes=None, links=None, directories=(), provenance=None,
                 max_files=MAX_EDITS, max_bytes=MAX_EDIT_BYTES):
        _need(isinstance(base, Mapping) and base and all(type(b) is bytes for b in base.values()),
              'baseline must be a non-empty mapping of exact source bytes')
        _need(type(max_files) is int and max_files > 0 and type(max_bytes) is int and max_bytes > 0,
              'candidate limits must be positive integers')
        self.max_files, self.max_bytes = max_files, max_bytes
        self._files = MappingProxyType(dict(base))
        modes = dict(modes or {})
        _need(set(modes) <= self._files.keys() and all(type(m) is int and 0 <= m <= 0o7777 for m in modes.values()),
              'invalid baseline file modes')
        self._modes = MappingProxyType({p: _mode(modes.get(p, 0o644)) for p in self._files})
        self.links = MappingProxyType(dict(links or {}))
        self._dirs = frozenset(directories)
        for target in self.links.values():
            _path(target)
        files, mode_map, material_dirs = _project(self._files, self._modes, self.links, self._dirs)
        self.base = MappingProxyType(files)
        self.modes = MappingProxyType(mode_map)
        self.directories = frozenset(material_dirs)
        self.editable = frozenset(p for p, data in self._files.items() if _editable(p) and _is_utf8(data))
        self._namespace = _children(self._files, self.links, self._dirs)[0]
        self._folded = {p.casefold(): p for p in self._namespace}
        self._provenance = canonical(dict(provenance or {}))
        self._inventory_identity = {
            'schema': _SCHEMA,
            'provenance': json.loads(self._provenance),
            'physical_files': _manifest(self._files, self._modes),
            'links': dict(self.links),
            'directories': sorted(self._dirs),
        }
        self.baseline = digest(canonical(self._inventory_identity))

    @property
    def provenance(self):
        return json.loads(self._provenance)

    def inventory_manifest(self) -> dict:
        """Portable lock covering provenance, physical entries and expanded file identities."""
        return {**json.loads(canonical(self._inventory_identity)), 'baseline': self.baseline,
                'portable_files': _manifest(self.base, self.modes),
                'portable_directories': sorted(self.directories)}

    @classmethod
    def acquire_spark26_e0(cls, archive_path, repository_root, **limits):
        """Build E0 from the exact archive and verified overlay bytes in this repository."""
        overlay_dir = Path(repository_root) / 'experiments' / 'spark26_gguf'
        overlay = {}
        for name, expected in E0_OVERLAY_SHA256.items():
            data = (overlay_dir / name).read_bytes()
            normalized = data.replace(b'\r\n', b'\n')
            _need(digest(data) == expected or digest(normalized) == E0_OVERLAY_LF_SHA256[name],
                  f'trusted E0 overlay hash mismatch: {name}')
            overlay[name] = normalized
        inventory = read_source_archive(archive_path, expected_sha256=SGLANG_ARCHIVE_SHA256,
                                        expected_root=ARCHIVE_ROOT)
        files = dict(inventory.files)
        _need(HELPER not in files and {LOADER, MODEL} <= files.keys(), 'unexpected E0 overlay inputs')
        # Run only the verified trusted overlay. Write just its expected inputs into
        # an isolated scratch directory; archive files are never executed or imported.
        with tempfile.TemporaryDirectory(prefix='evolve-level3-overlay-') as tmp:
            scratch = Path(tmp)
            root = scratch / 'src'
            script = scratch / 'apply_overlay.py'
            (scratch / 'spark2_5_gguf.py').write_bytes(overlay['spark2_5_gguf.py'])
            for path in (LOADER, MODEL):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(files[path])
            ns = {'__file__': str(script), '__name__': '_verified_level3_e0_overlay'}
            exec(compile(overlay[script.name], str(script), 'exec'), ns)
            _need(ns['PIN'] == SGLANG_REVISION, 'unexpected E0 overlay revision')
            ns['apply'](root)
            for path in (LOADER, MODEL, HELPER):
                files[path] = (root / path).read_bytes()
        return cls(files, modes=inventory.modes, links=inventory.links, directories=inventory.directories,
                   provenance={'revision': SGLANG_REVISION, 'archive_sha256': inventory.archive_sha256,
                               'overlay_sha256': dict(E0_OVERLAY_SHA256),
                               'overlay_execution_lf_sha256': dict(E0_OVERLAY_LF_SHA256)}, **limits)

    @classmethod
    def acquire_spark26_bf16_e0(cls, archive_path, **limits):
        """Build unmodified BF16 E0 from the exact pinned SGLang archive (no GGUF overlay)."""
        inventory = read_source_archive(archive_path, expected_sha256=SGLANG_ARCHIVE_SHA256,
                                        expected_root=ARCHIVE_ROOT)
        files = dict(inventory.files)
        _need(HELPER not in files and {LOADER, MODEL} <= files.keys(), 'unexpected BF16 E0 archive inputs')
        return cls(files, modes=inventory.modes, links=inventory.links, directories=inventory.directories,
                   provenance={'revision': SGLANG_REVISION, 'archive_sha256': inventory.archive_sha256,
                               'target_precision': 'bf16', 'overlay': None}, **limits)

    def _check_edit_path(self, path):
        _path(path)
        _need(_editable(path) and (path not in self._files or path in self.editable),
              f'protected or unsupported source path: {path!r}')
        parent = path
        while parent:
            same = self._folded.get(parent.casefold(), parent)
            _need(same == parent, f'case-colliding source path: {path!r}')
            kind = self._namespace.get(parent)
            expected = 'file' if parent == path else 'directory'
            _need(kind is None or kind == expected, f'file/directory collision or archive alias edit: {path!r}')
            parent = parent.rpartition('/')[0]

    def _changes(self, args, existing):
        _need(type(args) is dict and 'files' in args and args.keys() <= {'files', 'deleted', 'build_args'},
              'candidate requires files with optional deleted and build_args')
        files, deleted = args['files'], args.get('deleted', [])
        _need(type(files) is dict and type(deleted) is list, 'expected file map and deletion list')
        _need(len(files) + len(deleted) <= self.max_files, 'candidate changes too many files')
        for p in deleted:
            self._check_edit_path(p)
            _need(p in existing and p not in files, 'deleted path missing or also submitted as an edit')
        deletions = tuple(sorted(set(deleted)))
        _need(len(deletions) == len(deleted), 'duplicate deletion')
        edits = {}
        total = 0
        for path, text in files.items():
            self._check_edit_path(path)
            _need(type(text) is str, 'source replacements must be UTF-8 text')
            try:
                data = text.encode('utf-8')
            except UnicodeError as e:
                raise ValueError('source replacement is not valid UTF-8') from e
            _need(b'\0' not in data and len(data) <= MAX_FILE_BYTES, 'binary or oversized source replacement')
            total += len(data)
            _need(total <= self.max_bytes, 'candidate source bytes exceed limit')
            if existing.get(path) != data:
                edits[path] = data
        return edits, deletions

    def _make(self, args, *, project=False):
        edits, deletions = self._changes(args, self._files)
        from . import v3_level3_startup as runtime
        if runtime.PATH in edits: runtime.options(edits[runtime.PATH])
        build = _build_args(args.get('build_args', {}))
        # Validate namespace/links even before publishing a candidate identity.
        tree = self._project_candidate(edits, deletions) if edits or deletions else None
        identity = digest(canonical({'schema': _SCHEMA, 'baseline': self.baseline,
                                     'edits': {p: digest(b) for p, b in edits.items()},
                                     'deleted': deletions, 'build_args': json.loads(build)}))
        engine = Level3Engine(identity, tuple(sorted(edits.items())), deletions, build)
        return (engine, tree) if project else engine

    def from_args(self, args: dict) -> Level3Engine:
        """Create an E0-relative candidate identity from the RPC payload."""
        return self._make(args)

    def from_parent(self, parent: SessionEngine, args: dict) -> Level3Engine:
        """Apply a relative change to a verified parent; export is still E0-relative.

        The caller records the parent edge separately from the content identity.
        Build arguments are inherited when omitted; an explicit object replaces them.
        """
        parent_args = self.to_args(parent)
        files = dict(self._files)
        for path in parent.deleted:
            del files[path]
        files.update(parent.edits)
        edits, deletions = self._changes(args, files)
        replacements, removed = dict(parent_args['files']), set(parent.deleted)
        for path in deletions:
            replacements.pop(path, None)
            if path in self._files:
                removed.add(path)
        for path, data in edits.items():
            removed.discard(path)
            if self._files.get(path) == data:
                replacements.pop(path, None)
            else:
                replacements[path] = data.decode('utf-8')
        return self.from_args({'files': replacements, 'deleted': sorted(removed),
                               'build_args': args.get('build_args', parent_args['build_args'])})

    def _project_candidate(self, edits, deleted):
        files = dict(self._files)
        for path in deleted:
            del files[path]
        files.update(edits)
        modes = {p: self._modes.get(p, 0o644) for p in files}
        tree = _project(files, modes, self.links, self._dirs)
        if self.links:
            # A pinned directory alias may project a writable target into a protected subtree.
            for path in self.base.keys() | tree[0].keys():
                if self.base.get(path) != tree[0].get(path):
                    _need(_editable(path), f'protected projected source path: {path!r}')
        return tree

    def _replay(self, engine):
        _need(type(engine) is Level3Engine and type(engine.edits) is tuple and type(engine.deleted) is tuple,
              'expected a Level 3 Engine')
        try:
            args = {'files': {path: raw.decode('utf-8') for path, raw in engine.edits},
                    'deleted': list(engine.deleted), 'build_args': engine.build_args}
        except (TypeError, AttributeError, ValueError) as e:
            raise ValueError('invalid Level 3 Engine payload') from e
        verified, tree = self._make(args, project=True)
        _need(verified == engine, 'candidate digest or payload does not match this baseline')
        return args, tree

    def to_args(self, engine: SessionEngine) -> dict:
        """Canonical JSON-compatible candidate deltas; verifies identity before export."""
        return self._replay(engine)[0]

    def effective_tree(self, engine: SessionEngine) -> tuple[dict[str, bytes], dict[str, int], frozenset[str]]:
        """Verified complete files, modes and directories for the trusted build worker."""
        _, tree = self._replay(engine)
        if tree is None:
            return dict(self.base), dict(self.modes), self.directories
        files, modes, dirs = tree
        return files, modes, frozenset(dirs)

    # Keep the build adapter compatible during the transition to the public method.
    _effective = effective_tree

    def effective(self, engine: SessionEngine) -> dict[str, bytes]:
        """Complete byte map of a verified candidate; no commands are executed."""
        return self.effective_tree(engine)[0]

    def materialize(self, engine: SessionEngine, destination) -> Path:
        """Write a verified tree, without following links or replacing an existing tree."""
        files, modes, directories = self.effective_tree(engine)
        dest = Path(destination).absolute()
        # Exclusive root creation above all candidate paths. Callers must only use
        # this pathname after materialization returns; I/O failures may leave staging.
        dest.mkdir(mode=0o700, parents=False, exist_ok=False)
        for path in sorted(directories):
            (dest / path).mkdir(mode=0o755)
        for path, data in sorted(files.items()):
            target = dest / path
            with target.open('xb') as f:
                f.write(data)
            os.chmod(target, modes[path])
        return dest


Level3SourceContract = SourceContract
