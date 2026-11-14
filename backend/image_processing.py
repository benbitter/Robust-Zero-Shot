import base64
import hashlib
import warnings
from io import BytesIO
import numpy as np
from PIL import Image,ImageFilter,ImageOps
from src.e6_corruptions import corrupt,noise_seed
MAX_BYTES=10*1024*1024
MAX_PIXELS=16_000_000
LEVELS={'resolution':[64,48,32,16,8],'blur':[0,1,2,3],'noise':[0,5,10,15],'jpeg':[100,80,60,40]}
def decode(data):
    if not data or len(data)>MAX_BYTES: raise ValueError('Upload must be between 1 byte and 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ('JPEG','PNG'): raise ValueError('Only JPEG and PNG are supported.')
                if image.width*image.height>MAX_PIXELS or max(image.size)>8192: raise ValueError('Image exceeds dimension limits.')
                image.verify()
            with Image.open(BytesIO(data)) as image:
                return ImageOps.exif_transpose(image).convert('RGB').copy()
    except (OSError,SyntaxError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise ValueError('Invalid, corrupt or oversized image.') from None

def degrade(image,kind,value):
    if kind not in LEVELS or value not in LEVELS[kind]: raise ValueError('Unsupported corruption setting.')
    # Explicit research-compatible 64px canvas for BOTH clean and degraded lab inputs.
    original=image.resize((64,64),Image.Resampling.BICUBIC)
    identity=hashlib.sha256(original.tobytes()).hexdigest()
    if (kind=='resolution' and value==64) or (kind in ('blur','noise') and value==0): return original,original.copy()
    condition=f'{kind}:{value}'
    if condition in ('resolution:48','resolution:32','resolution:16','resolution:8','blur:1','blur:2','noise:5','noise:15','jpeg:40'):
        return original,corrupt(original,condition,identity,42)
    if kind=='blur': changed=original.filter(ImageFilter.GaussianBlur(value))
    elif kind=='noise':
        a=np.asarray(original,dtype=float)+np.random.default_rng(noise_seed(identity,condition,42)).normal(0,value,(64,64,3))
        changed=Image.fromarray(np.rint(a).clip(0,255).astype(np.uint8))
    else:
        out=BytesIO(); original.save(out,format='JPEG',quality=value,subsampling=0); out.seek(0)
        with Image.open(out) as decoded: changed=decoded.convert('RGB').copy()
    return original,changed

def preview(image):
    b=BytesIO(); image.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
