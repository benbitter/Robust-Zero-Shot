"""Dataset-free deployment of the frozen six-model configuration."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from production_20.common import ROOT, read, sha, identity, load_clip, backbone, vectors
from production_20.service import UnifiedService

BUNDLE=ROOT/'deployment'

def available(): return (BUNDLE/'manifest.json').is_file()

def verify():
    m=read(BUNDLE/'manifest.json')
    if m.get('version')!=1: raise ValueError('Unsupported deployment bundle')
    for relative,digest in m['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:
            raise ValueError('Deployment file missing or changed: '+relative+'; run git lfs pull')
    if backbone()!=m['backbone']: raise ValueError('CLIP files differ from deployment provenance')
    taxonomy=read(BUNDLE/'taxonomy.json')
    if identity(taxonomy)!=m['taxonomy_id']: raise ValueError('Deployment taxonomy differs')
    from production_20.taxonomy import build
    if taxonomy!=build()[0]: raise ValueError('Expected fixed 20-class taxonomy')
    with np.load(BUNDLE/'prototypes.npz',allow_pickle=False) as z:
        vectors(z['prototypes'],20);vectors(z['simple'],20)
    return m

def report(name):
    m=read(BUNDLE/'manifest.json');relative='deployment/'+name
    if relative not in m['files'] or sha(ROOT/relative)!=m['files'][relative]:
        raise ValueError('Deployment report integrity failure')
    return read(ROOT/relative)

def final_results():
    s=report('final_summary.json')
    m=read(BUNDLE/'manifest.json')
    if s['selection_id']!=m['selection_id']:raise ValueError('Wrong report selection')
    s['plot_urls']=['/api/unified/final-benchmark/plots/'+Path(p).name for p in m['files'] if p.startswith('deployment/plots/') and p.endswith('.png')]
    return s

def plot(name):
    if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Invalid plot name')
    m=read(BUNDLE/'manifest.json');relative='deployment/plots/'+name
    if relative not in m['files'] or sha(ROOT/relative)!=m['files'][relative]:raise ValueError('Plot missing or changed')
    return ROOT/relative

class PortableService(UnifiedService):
    # Inherit the existing prediction path, TTA and score aggregation verbatim.
    def startup(self):
        import torch
        from src.visual_adapter import VisualAdapter
        m=verify();self.taxonomy=read(BUNDLE/'taxonomy.json');self.classes=[c['name'] for c in self.taxonomy['classes']]
        device=os.getenv('TERRACLIP_DEVICE','auto')
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda'):raise ValueError('Use auto, cpu or cuda')
        if device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
            import socket
            self.gpu_lock=socket.socket()
            try:self.gpu_lock.bind(('127.0.0.1',18764))
            except OSError:self.gpu_lock.close();raise RuntimeError('Stop other TerraCLIP GPU processes first') from None
        model,processor=load_clip(device)
        self.clip=SimpleNamespace(model=model,processor=processor,device=torch.device(device))
        self.adapters={}
        for key in ('m3','m4'):
            payload=torch.load(BUNDLE/(key+'.pt'),map_location='cpu',weights_only=True)
            cfg=payload['config']
            if identity(cfg)!=m['checkpoints'][key]['config_id'] or cfg['taxonomy_id']!=m['taxonomy_id'] or cfg['classes']!=self.taxonomy['classes']:
                raise ValueError('Checkpoint configuration mismatch')
            if cfg['text_sha256']!=m['files']['deployment/prototypes.npz'] or cfg['backbone']!=m['backbone']:
                raise ValueError('Checkpoint feature provenance mismatch')
            adapter=VisualAdapter(**cfg['architecture']);adapter.load_state_dict(payload['state_dict'],strict=True)
            if any(not torch.isfinite(p).all() for p in adapter.parameters()):raise ValueError('Nonfinite adapter weights')
            self.adapters[key]=adapter.to(device).eval().requires_grad_(False)
        with np.load(BUNDLE/'prototypes.npz',allow_pickle=False) as z:
            self.e2=torch.from_numpy(z['prototypes'].copy()).to(device)
            self.simple=torch.from_numpy(z['simple'].copy()).to(device);self.scale=float(z['scale'])
        self.recipe=copy.deepcopy(m['recipe']);self.ready=True

if __name__=='__main__':
    m=verify();print('Deployment integrity OK: 20 classes, six configurations, local CLIP and two adapters. No inference performed.')
