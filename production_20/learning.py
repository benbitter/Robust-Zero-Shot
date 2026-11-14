"""Adapter-only training and validation-selected, explicitly gated test evaluation."""
import time
import numpy as np
from .common import OUT, artifacts, identity, read, write, sha, log, vectors
from .features import TRAIN_CONDITIONS, load_features, load_text

def source_weights(rows):
    from collections import Counter
    counts=Counter(r['source'] for r in rows)
    if set(counts)!= {'eurosat','nwpu'}: raise ValueError('Both sources required')
    return np.array([len(rows)/(2*counts[r['source']]) for r in rows],dtype=np.float32)

def metrics(scores, rows, count):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
    y=np.array([r['unified_id'] for r in rows]); pred=scores.argmax(1)
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=list(range(count)),zero_division=0)
    matrix=confusion_matrix(y,pred,labels=list(range(count)))
    source={name:{'count':int(sum(x['source']==name for x in rows)),
      'top1':float(accuracy_score(y[[x['source']==name for x in rows]],pred[[x['source']==name for x in rows]])),
      'macro_f1':float(f1_score(y[[x['source']==name for x in rows]],pred[[x['source']==name for x in rows]],average='macro',zero_division=0))} for name in sorted({x['source'] for x in rows})}
    for name in source:
        indices=[i for i,row in enumerate(rows) if row['source']==name]
        yy=y[indices]; pp=pred[indices]
        source[name]['top5']=float(np.mean([y[i] in np.argsort(-scores[i],kind='stable')[:5] for i in indices]))
        source[name]['per_class_accuracy']={str(c):float(np.mean(pp[yy==c]==c)) for c in np.unique(yy)}
        source[name]['confusion_matrix']=confusion_matrix(yy,pp,labels=list(range(count))).tolist()
    return {'top1':float(accuracy_score(y,pred)),'top5':float(np.mean([truth in np.argsort(-v,kind='stable')[:5] for truth,v in zip(y,scores)])),'macro_f1':float(f1_score(y,pred,labels=list(range(count)),average='macro',zero_division=0)),
      'source':source,'source_balanced_top1':float(np.mean([v['top1'] for v in source.values()])),
      'per_class':[{'id':i,'precision':float(p[i]),'recall':float(r[i]),'accuracy':float(r[i]),'f1':float(f[i]),'support':int(s[i])} for i in range(count)],'confusion_matrix':matrix.tolist()}

def score(model,features,text,device,batch=256):
    import torch
    outputs=[]
    with torch.inference_mode():
        t=torch.nn.functional.normalize(torch.from_numpy(text.copy()).to(device),dim=-1)
        for start in range(0,len(features),batch):
            x=torch.from_numpy(features[start:start+batch].copy()).to(device)
            z=torch.nn.functional.normalize(x,dim=-1) if model is None else model(x)
            outputs.append((z@t.T).cpu().numpy())
    return np.concatenate(outputs)

def save_checkpoint(path,payload):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
    torch.save(payload,temp); temp.replace(path)

def load_adapter(path,manifest,taxonomy,device):
    import torch
    from src.visual_adapter import VisualAdapter
    payload=torch.load(path,map_location='cpu',weights_only=True); cfg=payload['config']
    if cfg['manifest_id']!=identity(manifest) or cfg['taxonomy_id']!=identity(taxonomy) or cfg['text_sha256']!=sha(OUT/'text/prototypes.npz'): raise ValueError('Checkpoint provenance mismatch')
    if cfg['classes']!=taxonomy['classes'] or cfg.get('benchmark')!='terraclip20-v1' or cfg['architecture']!={'dimension':512,'hidden_dim':128,'alpha':.2}: raise ValueError('Checkpoint taxonomy/architecture mismatch')
    model=VisualAdapter(**cfg['architecture']); model.load_state_dict(payload['state_dict'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.parameters()): raise ValueError('Nonfinite checkpoint weights')
    return model.to(device).eval().requires_grad_(False),payload

def train(args):
    import torch
    from tqdm import tqdm
    from src.visual_adapter import VisualAdapter
    manifest,taxonomy=artifacts(); text,scale,textcfg=load_text(taxonomy)
    conditions=['clean'] if args.recipe=='clean' else TRAIN_CONDITIONS
    features={}; rows=None
    for c in conditions:
        features[c],rr=load_features(manifest,'train',c)
        if rows is not None and rr!=rows: raise ValueError('Condition/label alignment failure')
        rows=rr
    val,valrows=load_features(manifest,'val','clean')
    cfg={'version':1,'benchmark':'terraclip20-v1','recipe':args.recipe,'conditions':conditions,'manifest_id':identity(manifest),
      'taxonomy_id':identity(taxonomy),'classes':taxonomy['classes'],'text_sha256':sha(OUT/'text/prototypes.npz'),
      'backbone':textcfg['backbone'],'scale':scale,'architecture':{'dimension':512,'hidden_dim':128,'alpha':.2},
      'seed':42,'lr':args.learning_rate,'weight_decay':.0001,'optimizer':'AdamW','epochs':args.epochs,
      'patience':args.patience,'batch_size':args.batch_size,'source_weighting':'N/(2*N_source), uniform shuffled epochs',
      'selection':'mean EuroSAT/NWPU clean validation accuracy','normalization':'input L2; residual x+alpha*MLP(x); output L2',
      'device':args.device,'feature_configs':{c:identity(read(OUT/'cache/train'/c.replace(':','_')/'r0/config.json')) for c in conditions}}
    dest=OUT/'checkpoints'/args.run
    if dest.exists() and not args.resume: raise FileExistsError(f'Run exists; use --resume or new --run: {dest}')
    if (OUT/'ultimate.json').exists() and args.run in read(OUT/'ultimate.json')['checkpoints']: raise ValueError('Run frozen for test; choose a new run name')
    if args.resume and not (dest/'last.pt').exists(): raise ValueError('No committed epoch to resume; choose a fresh run name')
    write(dest/'config.json',cfg)
    torch.manual_seed(42); np.random.seed(42)
    if args.device=='cuda': torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    model=VisualAdapter(**cfg['architecture']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(42); history=[]; best=-1.; bad=0; first=1; duration=0.
    if args.resume:
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if state['config']!=cfg: raise ValueError('Resume configuration differs')
        model.load_state_dict(state['state_dict'],strict=True); optimizer.load_state_dict(state['optimizer'])
        # Optimizer state follows parameter device after load_state_dict.
        rng.bit_generator.state=state['rng']; history=state['history']; best=state['best']; bad=state['bad']; first=state['epoch']+1; duration=state['seconds']
        if first>args.epochs or bad>=args.patience:
            if not (dest/'complete.json').exists(): write(dest/'complete.json',{'config_id':identity(cfg),'best_sha256':sha(dest/'best.pt'),'last_sha256':sha(dest/'last.pt'),'epochs_completed':state['epoch'],'seconds':duration})
            log('Run already finished'); return
    y=torch.tensor([r['unified_id'] for r in rows],device=args.device)
    weights=torch.from_numpy(source_weights(rows)).to(args.device); t=torch.from_numpy(text.copy()).to(args.device)
    started=time.perf_counter()
    for epoch in tqdm(range(first,args.epochs+1),desc='20-class epochs'):
        model.train(); order=rng.permutation(len(rows)); total=0.; source_loss={'eurosat':0.,'nwpu':0.}; source_n={'eurosat':0,'nwpu':0}
        condition='clean' if args.recipe=='clean' else conditions[1:][(epoch-1)%(len(conditions)-1)]
        for start in tqdm(range(0,len(rows),args.batch_size),desc=f'Epoch{epoch} {condition}',leave=False):
            idx=order[start:start+args.batch_size]; labels=y[idx]
            x=torch.from_numpy(features['clean'][idx].copy()).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss_vector=torch.nn.functional.cross_entropy(scale*(model(x)@t.T),labels,reduction='none')
            if condition!='clean':
                d=torch.from_numpy(features[condition][idx].copy()).to(args.device)
                loss_vector=.5*(loss_vector+torch.nn.functional.cross_entropy(scale*(model(d)@t.T),labels,reduction='none'))
            loss=(loss_vector*weights[idx]).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            if any(not torch.isfinite(v).all() for v in model.parameters()): raise ValueError('Nonfinite weights')
            total+=float(loss.detach())*len(idx)
            values=loss_vector.detach().cpu().tolist()
            for i,v in zip(idx,values): source_loss[rows[i]['source']]+=v; source_n[rows[i]['source']]+=1
        model.eval(); result=metrics(score(model,val,text,args.device),valrows,len(taxonomy['classes'])); value=result['source_balanced_top1']
        history.append({'epoch':epoch,'condition':condition,'loss':total/len(rows),'source_loss':{s:source_loss[s]/source_n[s] for s in source_n},'source_count':source_n.copy(),'lr':cfg['lr'],'validation':result})
        if value>best: best=value; bad=0; improved=True
        else: bad+=1; improved=False
        payload={'config':cfg,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch}
        if improved: save_checkpoint(dest/'best.pt',payload)
        save_checkpoint(dest/'last.pt',{**payload,'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,'history':history,'best':best,'bad':bad,'seconds':duration+time.perf_counter()-started})
        write(dest/'history.json',history,update=True); log(f'Epoch{epoch}: loss={total/len(rows):.5f}; source-balanced val={value:.4%}; best={best:.4%}')
        if bad>=args.patience: break
    write(dest/'complete.json',{'config_id':identity(cfg),'best_sha256':sha(dest/'best.pt'),'last_sha256':sha(dest/'last.pt'),'epochs_completed':history[-1]['epoch'],'seconds':duration+time.perf_counter()-started})
