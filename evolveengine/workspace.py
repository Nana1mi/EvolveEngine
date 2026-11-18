"""Materialize a pinned Git tree and apply only tracked text-file edits."""

from contextlib import contextmanager
import io
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

from .schema import in_scope, safe_path


class Repository:
    def __init__(self, root, revision, editable):
        self.root = Path(root).resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Engine revision must be a full Git commit SHA")
        self.revision = revision
        self.editable = tuple(editable)
        if not self.editable or not all(safe_path(name) for name in self.editable):
            raise ValueError("Engine editable paths must be configured")
        if self.git("rev-parse", revision + "^{commit}").decode().strip() != revision:
            raise ValueError("Engine base is not the configured commit")

    def git(self, *args):
        return subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(self.root), *args], check=True,
                              capture_output=True, timeout=45).stdout

    def source(self, engine, path):
        if engine["base"] != self.revision or not safe_path(path):
            raise ValueError("Engine source request is outside the pinned tree")
        if path in engine["edits"]:
            return engine["edits"][path]
        try:
            return self.git("show", f"{self.revision}:{path}").decode("utf-8")
        except (subprocess.CalledProcessError, UnicodeError):
            return None

    @contextmanager
    def checkout(self, engine):
        if engine["base"] != self.revision:
            raise ValueError("Engine snapshot has a different base")
        with tempfile.TemporaryDirectory(prefix="evolveengine-") as folder:
            directory = Path(folder)
            archive = self.git("archive", "--format=zip", self.revision)
            links = []
            with zipfile.ZipFile(io.BytesIO(archive)) as tree:
                for item in tree.infolist():
                    name = item.filename.rstrip("/")
                    if not safe_path(name):
                        raise ValueError("Engine source contains an unsupported path")
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        target = tree.read(item).decode("utf-8")
                        if not safe_path(target) or in_scope(name, self.editable):
                            raise ValueError(f"Unsafe or editable engine symlink: {name}")
                        links.append((directory / name, target))
                    else:
                        tree.extract(item, directory)
            # Upstream includes a read-only .dockerignore link. Copy its in-tree
            # target on Windows and Linux so neither engine workspace has symlinks.
            for link, target in links:
                resolved = link.parent / target
                if not resolved.is_file() or any(resolved == name for name, _ in links):
                    raise ValueError(f"Engine symlink lacks a regular source file: {link.name}")
                shutil.copyfile(resolved, link)
            for name, content in engine["edits"].items():
                file = directory / name
                if (not safe_path(name) or not in_scope(name, self.editable) or not file.is_file()
                        or not isinstance(content, str)):
                    raise ValueError(f"Unregistered engine edit: {name}")
                file.write_bytes(content.encode("utf-8"))
            yield directory

    @staticmethod
    def editable_snapshot(directory, scope):
        paths = {}
        for file in directory.rglob("*"):
            name = file.relative_to(directory).as_posix()
            if file.is_file() and not file.is_symlink() and in_scope(name, scope) and file.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}:
                paths[name] = file.read_bytes().decode("utf-8")
        return paths
