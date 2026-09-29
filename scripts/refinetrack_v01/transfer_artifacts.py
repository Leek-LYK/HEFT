"""Verify or pack the separately transferred RefineTrack artifacts (stdlib only)."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / '科研/v0.1/evidence/migration_artifacts.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root, records):
    paths = []
    for row in records:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe manifest path: {relative}')
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing file or unexpected symlink: {path}')
        if path.stat().st_size != row['bytes'] or file_sha256(path) != row['sha256']:
            raise ValueError(f'Artifact mismatch: {path}')
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'pack'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--include-optional', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [row for row in manifest['files'] if row['required'] or args.include_optional]
    paths = verify(args.root.resolve(), records)
    if args.action == 'pack':
        if args.output is None:
            parser.error('pack requires --output (a new archive path)')
        # Exclusive creation prevents overwriting an existing transfer archive.
        with args.output.open('xb') as stream:
            with tarfile.open(fileobj=stream, mode='w:gz') as archive:
                for row, path in zip(records, paths):
                    archive.add(path, arcname=row['path'], recursive=False)
        print(f'Archive: {args.output}; SHA256: {file_sha256(args.output)}')
    print(f'PASS: {len(records)} files, {sum(row["bytes"] for row in records)} bytes')


if __name__ == '__main__':
    main()
