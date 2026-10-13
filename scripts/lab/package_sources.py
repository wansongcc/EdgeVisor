#!/usr/bin/env python3
"""Package build sources only; never copy models, credentials or Git internals."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--source-root', default=str(pathlib.Path(__file__).resolve().parents[2]))
args = parser.parse_args()
root = pathlib.Path(args.source_root).resolve()
output = pathlib.Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
allowed = {'.cpp', '.hpp', '.h', '.cu', '.comp', '.inc', '.inl', '.c'}
paths = [p for p in (root / 'EdgeVisor/src').rglob('*') if p.is_file() and p.suffix in allowed]
paths += [root / 'Makefile', root / 'EdgeVisor/Makefile']
manifest = {}
archive = output / 'source.tar.gz'
with tarfile.open(archive, 'w:gz', dereference=True) as tar:
    for path in sorted(paths):
        # A source-tree symlink must not pull a private file from elsewhere.
        path.resolve().relative_to(root)
        name = str(path.relative_to(root))
        data = path.read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
        tar.add(path, arcname=name, recursive=False)
(output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                        text=True, capture_output=True, check=True).stdout.strip()
dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--',
                        'Makefile', 'EdgeVisor/Makefile', 'EdgeVisor/src'],
                       text=True, capture_output=True, check=True).stdout.splitlines()
info = {'commit': commit, 'dirty_build_sources': dirty,
        'source_files': len(manifest),
        'source_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
(output / 'build_source.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
