"""Unified live inference with one encoder and exactly six compatible models."""
from threading import Lock
from time import perf_counter
import os
import numpy as np
from .common import OUT, artifacts, read, write, identity, sha, backbone, load_clip
from .taxonomy import MODELS
from .features import VIEWS, TEST_CONDITIONS, load_text
from .benchmark import frozen, checkpoints, check_report

def tensor_scores(features,simple,e2,adapters,recipe,names):
    """features: batch x view x512; adapt each view before averaging cosine scores."""
    import torch
    x=torch.nn.functional.normalize(features.float(),dim=-1)
    simple=torch.nn.functional.normalize(simple.float(),dim=-1); e2=torch.nn.functional.normalize(e2.float(),dim=-1)
    result={}; cached={}
    def adapter_scores(name,tta):
        key=(name,tta)
        if key not in cached:
            z=adapters[name](x if tta else x[:,:1])
            cached[key]=(z@e2.T).mean(dim=1)
        return cached[key]
    if ('m2' in names or ('m5' in names and recipe['tta'])) and x.shape[1]!=8: raise ValueError('Eight views required')
    for name in names:
        if name=='m0': result[name]=x[:,0]@simple.T
        elif name=='m1': result[name]=x[:,0]@e2.T
        elif name=='m2': result[name]=(x@e2.T).mean(1)
        elif name in ('m3','m4'): result[name]=adapter_scores(name,False)
        elif name=='m5': result[name]=sum(w*adapter_scores(k,recipe['tta']) for k,w in recipe['weights'].items())
        else: raise ValueError('Unknown model')
    return result

def verify_activation():
    s=frozen(); active=read(OUT/'active.json')
    if active['selection_id']!=identity(s): raise ValueError('Activation differs from frozen benchmark')
    for condition in TEST_CONDITIONS:
        root=OUT/'evaluation/test'/condition.replace(':','_')
        stamp=check_report(root)
        if stamp['protocol']['selection_id']!=identity(s) or stamp['protocol']['models']!=MODELS: raise ValueError('Test report differs from activated benchmark')
    return s

class UnifiedService:
    def __init__(self): self.ready=False;self.lock=Lock();self.classes=[];self.error=None
    def startup(self):
        import torch
        from types import SimpleNamespace
        manifest,taxonomy=artifacts(); self.selection=verify_activation(); self.taxonomy=taxonomy
        e2,self.scale,cfg=load_text(taxonomy); simple,_,_=load_text(taxonomy,'simple')
        if backbone()!=cfg['backbone']: raise ValueError('Local backbone differs')
        device=os.getenv('TERRACLIP_DEVICE','auto')
        if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda'): raise ValueError('Invalid device')
        if device=='cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
            if os.getenv('TERRACLIP_GPU_LOCK_HELD')!='1':
                import socket
                self.gpu_lock=socket.socket()
                try: self.gpu_lock.bind(('127.0.0.1',18764))
                except OSError: self.gpu_lock.close();raise RuntimeError('Stop other GPU TerraCLIP processes first') from None
        model,processor=load_clip(device)
        self.clip=SimpleNamespace(model=model,processor=processor,device=torch.device(device))
        self.adapters,_=checkpoints(manifest,taxonomy,device)
        self.e2=torch.from_numpy(e2.copy()).to(device);self.simple=torch.from_numpy(simple.copy()).to(device)
        self.classes=[c['name'] for c in taxonomy['classes']];self.recipe=self.selection['recipe'];self.ready=True
    def listing(self):
        from .taxonomy import build
        taxonomy=getattr(self,'taxonomy',build()[0])
        return [{'id':k,'name':v,'available':self.ready,'default':k=='m4','reason':None if self.ready else 'Unified artifacts unavailable or invalid',
          'taxonomy':'terraclip20-v1','sources':['NWPU-RESISC45','EuroSAT Forest/River'], 'categories':taxonomy['classes']} for k,v in MODELS.items()]
    def predict(self,image,names):
        import torch
        from src.tta import make_view
        from src.e3_runtime import encode_views
        if not self.ready or not names or set(names)-set(MODELS): raise ValueError('Unified model unavailable')
        needs_tta='m2' in names or ('m5' in names and self.recipe['tta'])
        with self.lock,torch.inference_mode():
            started=perf_counter()
            images=[make_view(image,v) for v in (VIEWS if needs_tta else ['r0'])]
            # Eight images maximum; same single CLIP model and official processor.
            features=torch.from_numpy(encode_views(self.clip,images)).to(self.clip.device)[None]
            if self.clip.device.type=='cuda': torch.cuda.synchronize()
            encoding_ms=1000*(perf_counter()-started);results=[]
            for name in names:
                start=perf_counter(); scores=tensor_scores(features,self.simple,self.e2,self.adapters,self.recipe,[name])[name][0]
                relative=(scores*self.scale).softmax(-1)
                if self.clip.device.type=='cuda': torch.cuda.synchronize()
                elapsed=1000*(perf_counter()-start);order=torch.argsort(scores,descending=True,stable=True)[:5].tolist()
                top=[{'class_name':self.classes[i],'description':self.taxonomy['classes'][i]['description'],'similarity':float(scores[i]),'relative_score':float(relative[i])} for i in order]
                results.append({'model':name,'model_name':MODELS[name],'predicted_class':top[0]['class_name'],'top5':top,'inference_ms':elapsed})
            return {'results':results,'shared_encoding_ms':encoding_ms,'agreement':len({r['predicted_class'] for r in results})==1}

def activate(args):
    manifest,taxonomy=artifacts();s=frozen();checkpoints(manifest,taxonomy,'cpu');_,_,cfg=load_text(taxonomy)
    if backbone()!=cfg['backbone']: raise ValueError('Local CLIP changed')
    for condition in TEST_CONDITIONS:
        root=OUT/'evaluation/test'/condition.replace(':','_');stamp=check_report(root)
        if stamp['protocol']['selection_id']!=identity(s) or stamp['protocol']['models']!=MODELS: raise ValueError('Missing compatible six-model test report')
    write(OUT/'active.json',{'selection_id':identity(s),'models':MODELS,'default':'m4'})
    print('Unified six-model interface activated. Restart backend. Prior work remains in Archive.')

def reports():
    s=frozen();rows=[]
    for condition in TEST_CONDITIONS:
        root=OUT/'evaluation/test'/condition.replace(':','_')
        if not (root/'complete.json').exists(): continue
        stamp=check_report(root)
        if stamp['protocol']['selection_id']!=identity(s): raise ValueError('Wrong benchmark report')
        for key,result in read(root/'metrics.json').items():
            rows.append({'model':key,'name':MODELS[key],'condition':condition,**result})
    latency=OUT/'benchmark/report.json'
    timing=read(latency) if latency.exists() else None
    if timing is not None and timing['selection_id']!=identity(s): raise ValueError('Latency report belongs to another selection')
    return {'rows':rows,'taxonomy':'terraclip20-v1','ultimate':s['recipe'],'improved_over_m4':s['improved_over_m4'],
      'latency':timing,
      'note':'Same20 classes and frozen test images for all six methods. Ultimate selected using validation only. Old experiments are excluded.'}

def benchmark(args):
    """Manual small validation-image live latency check, never an accuracy experiment."""
    import torch
    from .data import Images
    manifest,_=artifacts();os.environ['TERRACLIP_DEVICE']=args.device;os.environ['TERRACLIP_GPU_LOCK_HELD']='1'
    # This manual measurement runs after activation; user owns all real inference.
    service=UnifiedService();service.startup(); loader=Images(manifest)
    pool=[r for r in manifest['rows'] if r['split']=='val'];rng=np.random.default_rng(42)
    rows=[pool[i] for i in rng.choice(len(pool),min(args.samples,len(pool)),replace=False)]
    result={}
    for name in MODELS:
        values=[];peaks=[]
        for row in rows:
            image=loader.get(row)
            if args.device=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=perf_counter(); service.predict(image,[name])
            if args.device=='cuda': torch.cuda.synchronize()
            values.append(1000*(perf_counter()-start)); peaks.append(torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
        result[name]={'median_ms':float(np.median(values)),'mean_ms':float(np.mean(values)),
          'samples_ms':values,'gpu_peak_allocated_bytes':max(peaks) if args.device=='cuda' else None}
    write(OUT/'benchmark/report.json',{'selection_id':identity(service.selection),'device':args.device,'models':result,
      'image_ids':[r['image_id'] for r in rows],'scope':'Per-model end-to-end processor+CLIP+adapter+TTA, batch1, no disk decode, all adapters resident; first call included; validation images; not accuracy'})
