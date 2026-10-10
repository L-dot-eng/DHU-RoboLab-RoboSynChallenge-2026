"""Verify all release files without importing machine-learning packages."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'checksums.json').read_text(encoding='utf-8'))
    errors=[]
    for name,expected in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):
            errors.append(name+': invalid path')
        elif not path.is_file():
            errors.append(name+': missing')
        elif path.stat().st_size!=expected['bytes'] or sha256(path)!=expected['sha256']:
            errors.append(name+': checksum mismatch')
    return errors


if __name__=='__main__':
    problems=verify(Path(__file__).resolve().parent)
    for problem in problems:print(problem)
    print('FAILED' if problems else 'All release file checksums match')
    raise SystemExit(bool(problems))
