import csv
import hashlib
import json
import os
import re
from pathlib import Path


def redact(text):
    text = str(text)
    for key, value in os.environ.items():
        if value and ('PROXY' in key.upper() or key.upper() in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}):
            text = text.replace(value, '<redacted>')
    return re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1<redacted>@', text)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def guarded(main):
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'{type(exc).__name__}: {redact(exc)}') from None
