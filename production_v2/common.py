"""Artifact integrity and explicit, local-only model access."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/terraclip_v2'
PARQUET = ROOT / 'data/external/nwpu_resisc45/train-00000-of-00001-d0e01c925a6227a8.parquet'
MODEL = ROOT / 'models/clip-vit-base-patch16'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def identity(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def log(message): print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)

def write(path, obj, update=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not update:
        if read(path) != obj: raise FileExistsError(f'Artifact differs: {path}')
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8'); temp.replace(path)

def save_npz(path, **arrays):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez(f, **arrays)
    temp.replace(path)

def backbone():
    files = sorted(p for p in MODEL.iterdir() if p.suffix in ('.json', '.txt', '.safetensors', '.bin'))
    if not any(p.suffix in ('.safetensors', '.bin') for p in files): raise ValueError('Local CLIP weights missing')
    return {p.name: sha(p) for p in files}

def load_clip(device):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).to(device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    if model.config.projection_dim != 512: raise ValueError('Expected CLIP ViT-B/16 dimension512')
    return model, processor

def artifacts():
    manifest = read(OUT/'manifests/manifest.json')
    taxonomy = read(OUT/'taxonomy/taxonomy.json')
    if manifest['taxonomy_id'] != identity(taxonomy): raise ValueError('Taxonomy mismatch')
    from .taxonomy import build
    from .data import validate_manifest
    expected, euro, nwpu = build()
    if taxonomy != expected: raise ValueError('Taxonomy differs from pinned V2 definition')
    validate_manifest(manifest['rows'], len(taxonomy['classes']))
    for row in manifest['rows']:
        mapping = euro if row['source']=='eurosat' else nwpu if row['source']=='nwpu' else None
        if mapping is None or mapping[str(row['source_label'])] != {'original_name':row['source_name'],'unified_id':row['unified_id']}:
            raise ValueError('Source/unified label mismatch')
    return manifest, taxonomy

def vectors(array, rows, dimension=512):
    if array.shape != (rows, dimension) or array.dtype != np.float32 or not np.isfinite(array).all(): raise ValueError('Invalid embedding array')
    if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4): raise ValueError('Embeddings must be normalized')
