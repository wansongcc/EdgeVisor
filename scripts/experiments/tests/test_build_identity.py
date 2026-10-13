import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.experiments.build_identity import validate_build_identity
from scripts.experiments.recording import RunRecorder


class BuildIdentityTests(unittest.TestCase):
    def make_build(self, path, status='passed', isolated=False, compiled_api='api-new'):
        source = {'EdgeVisor/src/app.cpp': 'source-hash'}
        fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        run = RunRecorder(path, {'source': {'source_manifest_sha256': fingerprint}, 'check_only': isolated})
        (path/'deployment').mkdir()
        (path/'built_binaries.sha256').write_text('cli-new EdgeVisor/dllama\n'+compiled_api+' EdgeVisor/dllama-api\n')
        (path/'deployment/binary.sha256').write_text('cli-new /source/dllama\n')
        (path/'deployment/api_binary.sha256').write_text('api-new /source/dllama-api\n')
        run.seal(status)
        return source

    def test_old_api_cannot_hide_behind_matching_source_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run'; source = self.make_build(path)
            self.assertEqual(validate_build_identity(path, source, 'cli-new', 'api-new')['status'], 'passed')
            with self.assertRaises(ValueError): validate_build_identity(path, source, 'cli-new', 'api-old')

    def test_copied_binary_must_match_fresh_compile_not_just_deployment_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run'; source = self.make_build(path, compiled_api='api-old')
            with self.assertRaises(ValueError): validate_build_identity(path, source, 'cli-new', 'api-new')

    def test_failed_and_isolated_builds_cannot_be_model_gate_inputs(self):
        for status, isolated in [('failed', False), ('passed', True)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'run'; source = self.make_build(path, status, isolated)
                with self.assertRaises(ValueError): validate_build_identity(path, source, 'cli-new', 'api-new')

    def test_changed_source_cannot_reuse_an_accepted_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run'; source = self.make_build(path)
            source['EdgeVisor/src/app.cpp'] = 'changed'
            with self.assertRaises(ValueError): validate_build_identity(path, source, 'cli-new', 'api-new')


if __name__ == '__main__': unittest.main()
