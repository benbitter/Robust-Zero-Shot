"""Deterministic pixel-space degradation before official CLIP preprocessing."""
import hashlib
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
VERSION='rgb-bicubic-round-noise-jpeg-subsampling0-v1'
TRAIN=['clean','resolution:48','resolution:32','resolution:16','blur:1']
QUICK=['clean','resolution:32','resolution:16','blur:2','noise:15','jpeg:40']
FULL=['clean']+[f'resolution:{x}' for x in (48,32,24,16,8)]+[f'blur:{x}' for x in (.5,1,2)]+[f'noise:{x}' for x in (5,15,25)]+[f'jpeg:{x}' for x in (95,70,40,15)]
def noise_seed(image_id,condition,seed):
    return int.from_bytes(hashlib.sha256(f'{seed}|{image_id}|{condition}'.encode()).digest()[:8],'little')
def corrupt(image,condition,image_id,seed=42):
    x=image.convert('RGB').copy()
    if x.size!=(64,64): raise ValueError('Expected native 64x64 image')
    if condition=='clean': return x
    kind,value=condition.split(':'); value=float(value)
    if kind=='resolution':
        if value not in (48,32,24,16,8): raise ValueError('Invalid resolution')
        return x.resize((int(value),int(value)),Image.Resampling.BICUBIC).resize((64,64),Image.Resampling.BICUBIC)
    if kind=='blur' and value in (.5,1,2): return x.filter(ImageFilter.GaussianBlur(value))
    if kind=='noise' and value in (5,15,25):
        a=np.asarray(x,dtype=float)+np.random.default_rng(noise_seed(image_id,condition,seed)).normal(0,value,(64,64,3))
        return Image.fromarray(np.rint(a).clip(0,255).astype(np.uint8))
    if kind=='jpeg' and value in (95,70,40,15):
        b=BytesIO(); x.save(b,format='JPEG',quality=int(value),subsampling=0)
        b.seek(0)
        with Image.open(b) as decoded: return decoded.convert('RGB').copy()
    raise ValueError('Invalid corruption')
