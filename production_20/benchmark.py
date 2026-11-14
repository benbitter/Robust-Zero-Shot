"""Six fixed methods, validation-only Ultimate selection, sealed test protocol."""
import time
import numpy as np
from .common import OUT, read, write, sha, identity, artifacts, log
from .features import VIEWS, TEST_CONDITIONS, load_features, load_text
from .learning import load_adapter, metrics
from .taxonomy import MODELS

def candidates():
    # M4 wins ties. No parameter grid is inferred from test results.
    result=[{'name':'m4','tta':False,'weights':{'m4':1.}},
            {'name':'m3','tta':False,'weights':{'m3':1.}},
            {'name':'m4_tta','tta':True,'weights':{'m4':1.}},
            {'name':'m3_tta','tta':True,'weights':{'m3':1.}}]
    for tta in (False,True):
        for w in (.25,.5,.75): result.append({'name':f'ensemble_{w}_tta{tta}','tta':tta,'weights':{'m3':w,'m4':1-w}})
    return result

def blend(recipe,parts):
    if set(recipe['weights'])-{'m3','m4'} or not np.isclose(sum(recipe['weights'].values()),1) or any(w<0 for w in recipe['weights'].values()): raise ValueError('Invalid ensemble weights')
    return sum(w*parts[(key,recipe['tta'])] for key,w in recipe['weights'].items())

def objective(results):
    if set(results)!=set(TEST_CONDITIONS): raise ValueError('Missing selection conditions')
    return .5*results['clean']['macro_f1']+.5*np.mean([results[c]['macro_f1'] for c in TEST_CONDITIONS if c!='clean'])

def view_features(manifest,split,condition):
    arrays=[]; rows=None
    for view in VIEWS:
        f,rr=load_features(manifest,split,condition,view)
        if rows is not None and rows!=rr: raise ValueError('TTA image ordering mismatch')
        arrays.append(f); rows=rr
    return np.stack(arrays,axis=1),rows

def view_score(adapter,features,text,device,batch=128):
    import torch
    outputs=[]
    with torch.inference_mode():
        t=torch.nn.functional.normalize(torch.from_numpy(text.copy()).to(device),dim=-1)
        for start in range(0,len(features),batch):
            x=torch.from_numpy(features[start:start+batch].copy()).to(device)
            z=torch.nn.functional.normalize(x,dim=-1) if adapter is None else adapter(x)
            outputs.append((z@t.T).cpu().numpy())
    return np.concatenate(outputs)

def parts(features,text,adapters,device):
    result={}
    for key,adapter in adapters.items():
        views=view_score(adapter,features,text,device)
        result[(key,False)]=views[:,0,:]; result[(key,True)]=views.mean(axis=1)
    return result

def checkpoints(manifest,taxonomy,device):
    models={}; refs={}; configs={}
    for name in ('m3','m4'):
        root=OUT/'checkpoints'/name; complete=read(root/'complete.json')
        if sha(root/'best.pt')!=complete['best_sha256']: raise ValueError('Checkpoint digest differs')
        model,payload=load_adapter(root/'best.pt',manifest,taxonomy,device)
        if payload['config']['recipe']!=('clean' if name=='m3' else 'robust'): raise ValueError('Wrong training method')
        models[name]=model; refs[name]={'path':str((root/'best.pt').resolve()),'sha256':sha(root/'best.pt'),'epoch':payload['epoch'],'config_id':identity(payload['config'])}
        configs[name]=payload['config']
    for key in ('architecture','seed','lr','weight_decay','optimizer','epochs','patience','batch_size','source_weighting','selection','normalization'):
        if configs['m3'][key]!=configs['m4'][key]: raise ValueError('Controlled M3/M4 hyperparameter differs: '+key)
    return models,refs

def freeze(args):
    manifest,taxonomy=artifacts(); text,_,cfg=load_text(taxonomy)
    if (OUT/'ultimate.json').exists(): frozen(); log('Ultimate already frozen; preserved'); return
    models,refs=checkpoints(manifest,taxonomy,args.device); options=candidates(); reports={r['name']:{} for r in options}
    for condition in TEST_CONDITIONS:
        f,rows=view_features(manifest,'val',condition); p=parts(f,text,models,args.device)
        for recipe in options: reports[recipe['name']][condition]=metrics(blend(recipe,p),rows,20)
        log('Validated Ultimate candidates: '+condition)
    values={name:float(objective(results)) for name,results in reports.items()}
    chosen=options[0]
    for recipe in options[1:]:
        if values[recipe['name']]>values[chosen['name']]+1e-12: chosen=recipe
    selection={'version':'terraclip20-v1','manifest_id':identity(manifest),'taxonomy_id':identity(taxonomy),
      'prototype_sha256':sha(OUT/'text/prototypes.npz'),'prompts':cfg,'checkpoints':refs,'recipe':chosen,
      'candidates':options,'validation_objective':values,'validation_metrics':reports,
      'criterion':'0.5 clean macro F1 + 0.5 mean macro F1 across resolution32,resolution16,blur2,noise15,jpeg40; M4 wins ties',
      'conditions':TEST_CONDITIONS,'views':VIEWS,'improved_over_m4':values[chosen['name']]>values['m4']+1e-12,
      'test_policy':'all six methods on identical frozen test rows; no test-based reselection'}
    write(OUT/'ultimate.json',selection)
    log(f'Ultimate frozen: {chosen}; objective={values[chosen["name"]]:.6f}; improves over M4={selection["improved_over_m4"]}')

def frozen():
    manifest,taxonomy=artifacts(); s=read(OUT/'ultimate.json')
    if s['manifest_id']!=identity(manifest) or s['taxonomy_id']!=identity(taxonomy) or s['prototype_sha256']!=sha(OUT/'text/prototypes.npz'): raise ValueError('Frozen experiment identity changed')
    if s['candidates']!=candidates() or s['recipe'] not in candidates() or s['conditions']!=TEST_CONDITIONS or s['views']!=VIEWS: raise ValueError('Frozen protocol differs')
    for name,ref in s['checkpoints'].items():
        if name not in ('m3','m4') or sha(OUT/'checkpoints'/name/'best.pt')!=ref['sha256']: raise ValueError('Frozen weights changed')
    if set(s['checkpoints'])!={'m3','m4'}: raise ValueError('Incomplete checkpoint references')
    return s

def six_scores(features,simple,e2,adapters,recipe,device):
    m0=view_score(None,features[:,:1],simple,device)[:,0,:]
    base=view_score(None,features,e2,device)
    p=parts(features,e2,adapters,device)
    return {'m0':m0,'m1':base[:,0,:],'m2':base.mean(1),'m3':p[('m3',False)],'m4':p[('m4',False)],'m5':blend(recipe,p)}

def evaluate(args):
    import torch
    manifest,taxonomy=artifacts(); s=frozen(); text,_,_=load_text(taxonomy); simple,_,_=load_text(taxonomy,'simple')
    models,_=checkpoints(manifest,taxonomy,args.device)
    for condition in TEST_CONDITIONS:
        root=OUT/'evaluation'/args.split/condition.replace(':','_')
        protocol={'selection_id':identity(s),'condition':condition,'split':args.split,'manifest_id':identity(manifest),'models':MODELS}
        if (root/'complete.json').exists():
            check_report(root,protocol);log('Verified existing '+str(root));continue
        if root.exists(): raise FileExistsError('Partial report preserved; inspect before recovery: '+str(root))
        f,rows=view_features(manifest,args.split,condition)
        if args.device=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter(); outputs=six_scores(f,simple,text,models,s['recipe'],args.device)
        if args.device=='cuda': torch.cuda.synchronize()
        timing={'all_six_cached_scoring_seconds':time.perf_counter()-started,
          'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
          'note':'Shared cached scoring includes transfer/aggregation, excludes image decoding and CLIP extraction. Use manual benchmark for per-model end-to-end latency.'}
        reports={}
        for key,values in outputs.items():
            reports[key]={'name':MODELS[key],**metrics(values,rows,20)}
            write(root/f'{key}_predictions.json',[{'image_id':r['image_id'],'source':r['source'],'truth':r['unified_id'],
              'prediction':int(v.argmax()),'scores':v.tolist(),'top5':np.argsort(-v,kind='stable')[:5].tolist()} for r,v in zip(rows,values)])
        write(root/'metrics.json',reports);write(root/'timing.json',timing);write(root/'protocol.json',protocol)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5)); ax.bar(list(MODELS),[reports[k]['top1'] for k in MODELS]);ax.set(ylim=(0,1),ylabel='Top-1',title=args.split+' / '+condition);fig.savefig(root/'accuracy.png');plt.close(fig)
        for name in MODELS:
            fig,ax=plt.subplots(figsize=(9,8));ax.imshow(reports[name]['confusion_matrix']);ax.set(xlabel='Predicted ID',ylabel='True ID',title=MODELS[name]);fig.savefig(root/f'{name}_confusion.png');plt.close(fig)
        write(root/'complete.json',{'protocol':protocol,'files':{p.name:sha(p) for p in root.iterdir() if p.is_file()}})
        log('Saved six-model '+args.split+' / '+condition)

def check_report(root,protocol=None):
    stamp=read(root/'complete.json')
    if protocol is not None and stamp['protocol']!=protocol: raise ValueError('Report protocol differs')
    if any(sha(root/name)!=digest for name,digest in stamp['files'].items()): raise ValueError('Report files changed')
    return stamp
