import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.refinetrack_v01.transfer_artifacts import ROOT, file_sha256, verify


class TransferTests(unittest.TestCase):
    def test_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'data.bin'
            file.write_bytes(b'first')
            records = [{'path': 'data.bin', 'bytes': 5, 'sha256': file_sha256(file)}]
            self.assertEqual(verify(root, records), [file])
            file.write_bytes(b'other')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verify(root, records)

    def test_parent_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            verify(Path('/tmp'), [{'path': '../outside'}])

    def test_pack_has_only_selected_relative_files_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data.bin').write_bytes(b'example')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'files': [
                {'path': 'data.bin', 'bytes': 7, 'sha256': file_sha256(root / 'data.bin'), 'required': True},
                {'path': 'absent-optional.bin', 'required': False},
            ]}))
            archive = root / 'transfer.tar.gz'
            cmd = [sys.executable, str(ROOT / 'scripts/refinetrack_v01/transfer_artifacts.py'),
                   'pack', '--root', str(root), '--manifest', str(manifest), '--output', str(archive)]
            subprocess.run(cmd, check=True, capture_output=True)
            with tarfile.open(archive) as stream:
                self.assertEqual(stream.getnames(), ['data.bin'])
                self.assertEqual(stream.extractfile('data.bin').read(), b'example')
            before = file_sha256(archive)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, file_sha256(archive))


if __name__ == '__main__':
    unittest.main()
