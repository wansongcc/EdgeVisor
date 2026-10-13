"""Link observed executables to a sealed build of the deployed source snapshot."""
import hashlib
import json
from pathlib import Path
from scripts.experiments.recording import verify_artifacts


def validate_build_identity(directory, source_manifest, cli_sha256, api_sha256):
    directory = Path(directory)
    build = verify_artifacts(directory)
    metadata = build['metadata']
    if build['status'] != 'passed' or metadata.get('check_only', False):
        raise ValueError('Model gates require a passed deployment, not an isolated/failed build')
    source_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    if source_hash != metadata['source']['source_manifest_sha256']:
        raise ValueError('Deployed source differs from accepted build')
    compiled = {line.split()[1]: line.split()[0]
                for line in (directory/'built_binaries.sha256').read_text().splitlines()}
    for label, observed in [('dllama', cli_sha256), ('dllama-api', api_sha256)]:
        name = 'deployment/' + ('binary.sha256' if label == 'dllama' else 'api_binary.sha256')
        declared = (directory/name).read_text().split()[0]
        if observed != declared or observed != compiled['EdgeVisor/' + label]:
            raise ValueError('Observed ' + label + ' differs from fresh accepted build')
    return build
