from time import perf_counter
from threading import Lock
import numpy as np
from backend.model_registry import metadata,validate,NAMES
from src.config import ROOT
class InferenceService:
    def __init__(self):
        self.lock=Lock(); self.ready=False; self.models={}; self.error=None
    def startup(self):
        from backend.portable import available,PortableService
        if available():
            self.unified=PortableService();self.unified.startup()
            self.classes=self.unified.classes;self.clip=self.unified.clip
            self.ready=True
            return
        if (ROOT/'results/terraclip_20/active.json').exists():
            from production_20.service import UnifiedService
            self.unified=UnifiedService(); self.unified.startup()
            self.classes=self.unified.classes; self.clip=self.unified.clip
            self.ready=True
            return
        import os,torch
        from src.clip_baseline import ClipBaseline,parameter_hash
        from src.prompt_ensemble import encode_text_prompts,build_class_prototypes
        from src.visual_adapter import VisualAdapter
        self.meta=metadata(); self.classes=self.meta['classes']
        import json
        e2=json.loads((ROOT/'results/e2_prompt_ensemble/medium/experiment_config.json').read_text())
        if self.meta['expanded_prompts']!=e2['expanded_prompts'] or self.classes!=e2['class_order']: raise ValueError('E2 prototype mapping mismatch')
        device=os.getenv('TERRACLIP_DEVICE','auto')
        if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda'): raise ValueError('Invalid device')
        if device=='cuda':
            if not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
            import socket
            if os.getenv('TERRACLIP_GPU_LOCK_HELD')!='1':
                self.gpu_lock=socket.socket()
                try: self.gpu_lock.bind(('127.0.0.1',18764))
                except OSError: raise RuntimeError('Another TerraCLIP GPU process is active; stop it first') from None
            print(f'TerraCLIP GPU free memory: {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB',flush=True)
        self.clip=ClipBaseline(self.classes,device,model_path=ROOT/'models/clip-vit-base-patch16',local_files_only=True)
        if parameter_hash(self.clip.model)!=self.meta['weights_sha256_before'] or self.clip.processor.image_processor.to_dict()!=self.meta['preprocessing']: raise ValueError('CLIP provenance mismatch')
        groups=[self.meta['expanded_prompts'][c] for c in self.classes]
        self.text=build_class_prototypes(groups,encode_text_prompts(self.clip,groups)).to(self.clip.device)
        self.scale=float(self.clip.model.logit_scale.detach().exp().cpu())
        self.models['clip']={'adapter':None,'scale':self.scale}
        self.errors={}
        for name in ('e4','augmentation','consistency','final'):
            try:
                path,cfg=validate(name,self.meta)
                payload=torch.load(path,map_location='cpu',weights_only=True)
                recorded=payload.get('architecture') if name=='e4' else payload['config']['architecture']
                if recorded!=cfg['architecture']: raise ValueError('Architecture mismatch')
                adapter=VisualAdapter(**recorded).to(self.clip.device)
                adapter.load_state_dict(payload['state_dict'],strict=True)
                if any(not torch.isfinite(p).all() for p in adapter.parameters()): raise ValueError('Invalid weights')
                adapter.eval(); adapter.requires_grad_(False)
                self.models[name]={'adapter':adapter,'scale':cfg['frozen_logit_scale'] if name=='e4' else cfg['scale']}
            except (OSError,ValueError,KeyError,RuntimeError): self.errors[name]='Checkpoint missing or incompatible; check local artifacts.'
        if (ROOT/'results/terraclip_v2/active.json').exists():
            try:
                from production_v2.deployment import active_models
                self.models.update(active_models(self.clip.device))
            except (OSError,ValueError,KeyError,RuntimeError) as exc:
                self.errors['v2']=self.errors['v2_clip']='V2 activation invalid: '+str(exc)
        self.ready=True
    def listing(self):
        if hasattr(self,'unified'): return self.unified.listing()
        return [{'id':n,'name':label,'available':n in self.models and self.ready,'default':n==__import__('os').getenv('TERRACLIP_DEFAULT_MODEL','augmentation'),'reason':None if n in self.models and self.ready else getattr(self,'errors',{}).get(n,'Model service unavailable.'),
          'taxonomy':self.models.get(n,{}).get('taxonomy','v2' if n.startswith('v2') else 'eurosat10'),
          'sources':self.models.get(n,{}).get('sources',['EuroSAT RGB','NWPU-RESISC45'] if n.startswith('v2') else ['EuroSAT RGB']),
          'categories':self.models.get(n,{}).get('categories',[{'id':i,'name':c,'description':c} for i,c in enumerate(getattr(self,'classes',[]))])} for n,label in NAMES.items()]
    def predict(self,image,names):
        if hasattr(self,'unified'): return self.unified.predict(image,names)
        import torch
        from src.e3_runtime import encode_views
        if not self.ready: raise ValueError('Model service unavailable.')
        if not names or any(n not in self.models for n in names): raise ValueError('Requested model unavailable.')
        if len({self.models[n].get('taxonomy','eurosat10') for n in names})>1: raise ValueError('Choose models with compatible taxonomies.')
        with self.lock,torch.inference_mode():
            start=perf_counter(); f=torch.from_numpy(encode_views(self.clip,[image])).to(self.clip.device)
            if self.clip.device.type=='cuda': torch.cuda.synchronize()
            encoding_ms=1000*(perf_counter()-start); results=[]
            for name in names:
                start=perf_counter(); model=self.models[name]
                z=f if model['adapter'] is None else model['adapter'](f)
                classes=model.get('classes',self.classes); text=model.get('text',self.text)
                scores=(z@text.T)[0]; relative=(scores*model['scale']).softmax(-1)
                if self.clip.device.type=='cuda': torch.cuda.synchronize()
                ms=1000*(perf_counter()-start)
                order=torch.argsort(scores,descending=True,stable=True)[:5].tolist()
                descriptions={c['name']:c['description'] for c in model.get('categories',[])}
                top=[{'class_name':classes[i],'description':descriptions.get(classes[i],classes[i]),'similarity':float(scores[i]),'relative_score':float(relative[i])} for i in order]
                results.append({'model':name,'predicted_class':top[0]['class_name'],'top5':top,'inference_ms':ms})
            return {'results':results,'shared_encoding_ms':encoding_ms,'agreement':len({r['predicted_class'] for r in results})==1}
