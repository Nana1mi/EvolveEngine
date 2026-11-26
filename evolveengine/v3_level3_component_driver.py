"""Pinned non-root container entrypoint for incremental development builds."""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from time import monotonic


def main():
    phases = json.loads(Path('/ee-input/plan.json').read_bytes())
    tree = json.loads(Path('/ee-input/tree-plan.json').read_bytes())
    limits = json.loads(sys.argv[2])
    root = Path('/work')
    out = sys.stdout.buffer
    costs = {}

    def emit(lane, name, path, *, keep_timestamp=False):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('expected regular build output')
        row = [lane, name, info.st_size]
        if keep_timestamp:
            row.append(info.st_mtime_ns)
        out.write(json.dumps(row).encode() + b'\n')
        with path.open('rb') as src:
            shutil.copyfileobj(src, out, 1 << 20)

    start = monotonic()
    for name in ('tmp', 'build', 'cache', 'lib'):
        (root / name).mkdir()
    shutil.copytree('/ee-input/source', root / 'src')
    if 'configure' in phases:
        cached_build = '_ee_cmake_aot'
        shutil.copytree('/ee-input/cache', root / 'ccache', ignore=lambda _,names: {cached_build} & set(names))
        if tree['reuse']:
            shutil.copytree(Path('/ee-input/cache') / cached_build / 'tree', root/'build/aot')
            # The normal source staging time is fixed. Mark changed inputs newer than the restored objects.
            for name in tree['changed']:
                path = root/'src'/name
                if path.is_file():
                    os.utime(path, None)
        os.environ['CCACHE_MAXSIZE'] = str(max(1, limits['cache_bytes'] >> 20)) + 'M'
    if 'rust' in phases:
        shutil.copytree('/opt/ee-native/cargo', root / 'cargo')
    costs['prepare_source_and_dependencies'] = round(1000 * (monotonic() - start))
    for phase, command in phases.items():
        start = monotonic()
        if phase != 'configure' or not tree['reuse']:
            subprocess.run(command, cwd=root / 'src/python', stdin=subprocess.DEVNULL, stdout=sys.stderr, check=True)
        costs[phase] = round(1000 * (monotonic() - start))
    start = monotonic()
    out.write(b'level3-component-v1\n')
    native = root / 'build/aot'
    if 'native' in phases:
        for variant in ('sm90', 'sm100'):
            path = native / variant / 'common_ops.abi3.so'
            emit('module', f'sgl_kernel/{variant}/{path.name}', path)
        if 'infllm_ops' in phases['native']:
            libs = list(native.glob('infllm_ops.*.so'))
            if len(libs) != 1:
                raise ValueError('expected one infllm extension')
            emit('module', 'sgl_kernel/' + libs[0].name, libs[0])
        cache_files = [path for path in sorted((root/'ccache').rglob('*')) if path.is_file() and not path.is_symlink()]
        build_files = [path for path in sorted(native.rglob('*')) if path.is_file() and not path.is_symlink()]
        max_cache = limits['cache_bytes']
        tree_bytes = sum(path.stat().st_size for path in build_files)
        keep_tree = ('rust' not in phases and (native/'build.ninja').is_file() and
                     tree_bytes+64 <= max_cache and len(build_files)+1 <= limits['files'])
        if keep_tree:
            marker = root/'build/recipe.sha256'
            marker.write_text(tree['key'], encoding='ascii')
            emit('cache', f'{cached_build}/recipe.sha256', marker)
            for path in build_files:
                emit('cache', f'{cached_build}/tree/{path.relative_to(native).as_posix()}', path, keep_timestamp=True)
            max_cache -= tree_bytes+64
        count = len(build_files)+1 if keep_tree else 0
        for path in cache_files:
            size = path.stat().st_size
            if size <= max_cache and count < limits['files']:
                emit('cache', path.relative_to(root/'ccache').as_posix(), path)
                max_cache -= size
                count += 1
    if 'rust' in phases:
        for path in sorted((root / 'lib').rglob('*.so')):
            emit('module', path.relative_to(root / 'lib').as_posix(), path)
    costs['export_modules_and_cache'] = round(1000 * (monotonic() - start))
    out.write(json.dumps(['durations', costs]).encode() + b'\n')
    out.flush()


if __name__ == '__main__':
    main()
