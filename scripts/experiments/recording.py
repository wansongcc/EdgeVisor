"""Client-side evidence. Engine and sensor clocks remain separate domains."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def measurement(value=None, status='not_observed', unit=None):
    if status != 'measured' and value is not None:
        raise ValueError('An unavailable measurement cannot have a numeric value')
    if status == 'measured' and value is None:
        raise ValueError('A measured field must have a value')
    if status not in {'measured', 'not_observed', 'unsupported', 'error'}:
        raise ValueError('Unknown measurement status')
    return {'value': value, 'status': status, 'unit': unit}


class RunRecorder:
    """Exclusive run directory, append-only journal, sealed artifact digests.

    Do not pass a process environment or credentials as metadata. Sensor and
    perturbation measurements must explicitly distinguish observed/unavailable.
    A failed trial remains a sealed failed trial; it is never removed here.
    """
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.closed = False
        self.sequence = 0
        self.run_id = metadata.get('run_id') or uuid.uuid4().hex
        self.clock_id = self.run_id + ':client:' + uuid.uuid4().hex
        self.manifest = {'schema_version': 1, 'run_id': self.run_id,
                         'status': 'running', 'started_at_unix_ns': time.time_ns(),
                         'metadata': metadata, 'artifacts': {}}
        self.stream = (self.directory / 'client_events.jsonl').open('x')
        self._write_manifest()
        self.event('run_start')

    def _write_manifest(self):
        temp = self.directory / 'manifest.tmp'
        with temp.open('w') as output:
            json.dump(self.manifest, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temp.replace(self.directory / 'manifest.json')

    def event(self, phase, attributes=None):
        if self.closed:
            raise RuntimeError('Run already sealed')
        event = {'schema_version': 1, 'run_id': self.run_id,
                 'clock_id': self.clock_id, 'clock': 'monotonic', 'role': 'client',
                 'seq': self.sequence, 'ts_ns': time.monotonic_ns(),
                 'type': 'point', 'phase': phase, 'span_id': 0, 'parent_id': 0,
                 'attributes': attributes or {}}
        self.stream.write(json.dumps(event, allow_nan=False) + '\n')
        self.stream.flush()
        self.sequence += 1
        return event

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if not self.closed:
            self.seal('failed' if error else 'rejected',
                      {'type': kind.__name__, 'message': str(error)} if error else
                      {'message': 'No explicit acceptance result'})
        return False

    def seal(self, status, error=None):
        if self.closed:
            raise RuntimeError('Run already sealed')
        if status not in {'passed', 'failed', 'rejected', 'cancelled'}:
            raise ValueError('Run requires an explicit terminal status')
        self.event('run_end', {'status': status})
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.closed = True
        try:
            inventory = {}
            for path in sorted(self.directory.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Run evidence cannot be a symlink')
                name = str(path.relative_to(self.directory))
                if path.is_file() and name not in {'manifest.json', 'manifest.tmp'}:
                    inventory[name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            self.manifest['artifacts'] = inventory
        except BaseException as failure:
            self.manifest.update(status='rejected', error={'type': type(failure).__name__,
                'message': str(failure)}, artifact_integrity='error', ended_at_unix_ns=time.time_ns())
            self._write_manifest()
            raise
        self.manifest.update(status=status, error=error, artifact_integrity='sealed',
                             ended_at_unix_ns=time.time_ns())
        self._write_manifest()


def verify_artifacts(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('artifact_integrity') != 'sealed' or manifest['status'] not in {'passed', 'failed', 'rejected', 'cancelled'}:
        raise ValueError('Run is incomplete, not a successful trial')
    listed = set(manifest['artifacts'])
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and str(p.relative_to(directory)) not in {'manifest.json', 'manifest.tmp'}}
    if listed != actual:
        raise ValueError('Artifact inventory changed after sealing')
    for name, expected in manifest['artifacts'].items():
        path = directory / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Artifact name escapes run directory')
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Artifact digest mismatch: ' + name)
    return manifest
