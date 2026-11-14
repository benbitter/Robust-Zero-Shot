"""Pure image geometry and score aggregation; no model imports or initialization."""
import json
import hashlib
from zipfile import BadZipFile
from pathlib import Path
import numpy as np
from PIL import Image

CONFIGURATIONS = {
    'rotations': ('r0','r90','r180','r270'),
    'rotations_flips': ('r0','r90','r180','r270','r0h','r90h','r180h','r270h'),
    'flips': ('r0','r0h','r180h','r180'),
}
VIEW_VERSION = 'pillow-transpose-ccw-then-horizontal-v1'


def select_tta(name):
    if name == 'all': return list(CONFIGURATIONS)
    if name not in CONFIGURATIONS: raise ValueError('Unknown TTA configuration')
    return [name]


def make_view(image, view):
    """Lossless right-angle transpose, followed by horizontal flip if requested.

    r180h equals a vertical flip; r180 equals combined H+V flips.
    """
    if view not in CONFIGURATIONS['rotations_flips']: raise ValueError('Unknown view')
    image = image.convert('RGB')
    angle = int(view[1:].removesuffix('h'))
    if angle:
        image = image.transpose({90:Image.Transpose.ROTATE_90,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_270}[angle])
    else: image = image.copy()
    if view.endswith('h'): image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def average_scores(features, prototypes, available_views, configuration):
    """Mean of per-view cosine class scores; never renormalize the mean image vector."""
    features, prototypes = np.asarray(features), np.asarray(prototypes)
    if features.ndim != 3 or prototypes.ndim != 2 or features.shape[2] != prototypes.shape[1]:
        raise ValueError('Expected image x view x dimension and class x dimension')
    if features.shape[1] != len(available_views) or len(set(available_views)) != len(available_views):
        raise ValueError('View ordering mismatch')
    if not np.isfinite(features).all() or not np.isfinite(prototypes).all(): raise ValueError('Nonfinite features')
    indices = [available_views.index(v) for v in CONFIGURATIONS[configuration]]
    f = features[:,indices,:]; f = f / np.maximum(np.linalg.norm(f,axis=-1,keepdims=True),1e-12)
    t = prototypes / np.maximum(np.linalg.norm(prototypes,axis=-1,keepdims=True),1e-12)
    return (f @ t.T).mean(axis=1)


def identity(metadata):
    return hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()


def read_cache(path, metadata, ids, views):
    try:
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z['metadata'])) != metadata or z['ids'].tolist()!=ids or z['views'].tolist()!=views: return None
            f=z['features']
            if f.dtype!=np.float32 or f.shape!=(len(ids),len(views),metadata['embedding_dimension']): return None
            if not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=-1),1,atol=1e-4): return None
            if hashlib.sha256(f.tobytes()).hexdigest()!=str(z['sha256']): return None
            return f.copy()
    except (OSError,ValueError,KeyError,EOFError,BadZipFile): return None


def save_cache(path, metadata, ids, views, features):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    f=np.asarray(features,dtype=np.float32); temp=path.with_suffix('.tmp')
    with temp.open('wb') as out:
        np.savez_compressed(out,metadata=json.dumps(metadata,sort_keys=True),ids=np.array(ids),views=np.array(views),
                            features=f,sha256=hashlib.sha256(f.tobytes()).hexdigest())
    temp.replace(path)


def safe_output(path, allowed):
    path,allowed=Path(path).resolve(),Path(allowed).resolve()
    if not path.is_relative_to(allowed): raise ValueError(f'E3 writes must stay under {allowed}')
    return path


def require_empty(path):
    path=Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output exists: {path}. Choose a fresh --results-dir.')
