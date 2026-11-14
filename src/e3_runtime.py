"""Inference implementation imported only by an explicit manual evaluation command."""
import json
import importlib.metadata
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
from .config import ROOT, DATA, SPLITS, RESULTS
from .utils import read_csv,write_csv,write_json,digest
from .tta import CONFIGURATIONS,VIEW_VERSION,select_tta,make_view,average_scores,identity,read_cache,save_cache,require_empty,safe_output


def encode_views(model,images):
    with torch.inference_mode():
        try:
            pixels=model.processor(images=images,return_tensors='pt')['pixel_values'].to(model.device)
            vectors=model.model.get_image_features(pixel_values=pixels)
            return F.normalize(vectors.float(),dim=-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if 'pixels' in locals(): del pixels
            if 'vectors' in locals(): del vectors
            torch.cuda.empty_cache()
            if len(images)==1: raise RuntimeError('OOM for one view; run on CPU or free GPU memory') from None
    half=len(images)//2
    return np.concatenate([encode_views(model,images[:half]),encode_views(model,images[half:])])


def evaluate(args,out,cache_dir):
    from .clip_baseline import ClipBaseline,parameter_hash
    from .prompt_ensemble import build_ensemble_prompts,encode_text_prompts,build_class_prototypes
    from .evaluate import compute_metrics
    from .visualization import plots
    from .prepare_dataset import load_splits
    selected=select_tta(args.tta)
    for name in selected:
        require_empty(safe_output(out/name,ROOT/'results/e3_test_time_augmentation'))
    base_dir=RESULTS/'validation'; e2_dir=ROOT/'results/e2_prompt_ensemble/medium'
    base=json.loads((base_dir/'experiment_config.json').read_text())
    e2=json.loads((e2_dir/'experiment_config.json').read_text())
    classes=base['selected_classes']
    groups=build_ensemble_prompts(classes,base['fixed_prompt_mapping'],'medium')
    if base['status']!='complete' or base['mode']!='validation' or e2['status']!='complete' or e2['mode']!='validation' or e2['class_order']!=classes or e2['expanded_prompts']!=dict(zip(classes,groups)):
        raise ValueError('E2 Medium provenance differs; E3 fixes text prototypes to completed E2 Medium')
    splits,_=load_splits(SPLITS,base['seed']); val=splits['val']
    if val!=read_csv(base_dir/'evaluation_manifest.csv') or val!=read_csv(e2_dir/'evaluation_manifest.csv'):
        raise ValueError('Baseline validation manifests differ')
    rows=read_csv(SPLITS/'smoke.csv') if args.mode=='smoke' else val
    index={r['image_id']:r for r in val}
    if len(rows)!=(50 if args.mode=='smoke' else 4050) or len({r['image_id'] for r in rows})!=len(rows) or any(index.get(r['image_id'])!=r for r in rows):
        raise ValueError('Invalid frozen evaluation manifest')
    protected_dirs=[RESULTS,ROOT/'results/e1_domain_prompts',ROOT/'results/e2_prompt_ensemble',SPLITS,args.model_path]
    protected={str(p):digest(p) for d in protected_dirs for p in d.rglob('*') if p.is_file()}
    versions={lib:importlib.metadata.version(lib) for lib in base['library_versions']}
    if versions!=e2['library_versions']: raise ValueError('Library versions differ from E2 Medium')
    torch.manual_seed(base['seed']); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    model=ClipBaseline(classes,e2['device'],model_path=args.model_path,local_files_only=True)
    if str(model.device)!=e2['device']: raise ValueError('Device differs from saved E2 Medium; comparison settings must match')
    before=parameter_hash(model.model)
    if before!=base['weights_sha256_before'] or before!=e2['weights_sha256_before'] or model.processor.image_processor.to_dict()!=base['preprocessing']:
        raise ValueError('Checkpoint or official preprocessing differs')
    text={p:f.cpu() for p,f in zip(model.prompts,model.text_features)}
    text.update(encode_text_prompts(model,[[p for g in groups for p in g if p not in text]]))
    prototypes=build_class_prototypes(groups,text).numpy()
    # Union has only eight geometric views; all configurations share these encodings.
    needed={v for name in selected for v in CONFIGURATIONS[name]}
    views=[v for v in CONFIGURATIONS['rotations_flips'] if v in needed]
    ids=[r['image_id'] for r in rows]
    metadata={'checkpoint_sha256':before,'manifest_sha256':digest(SPLITS/('smoke.csv' if args.mode=='smoke' else 'val.csv')),
              'mode':args.mode,'preprocessing':base['preprocessing'],'embedding_dimension':model.model.config.projection_dim,
              'view_version':VIEW_VERSION,'views':views,'device':str(model.device),'dtype':'float32','library_versions':versions,
              'image_contents':identity({r['image_id']:digest(DATA/r['path']) for r in rows})}
    cache_path=cache_dir/(identity(metadata)+'.npz') if cache_dir else None
    features=read_cache(cache_path,metadata,ids,views) if cache_path and cache_path.exists() else None
    cache_hit=features is not None
    if features is None:
        features=np.empty((len(rows),len(views),metadata['embedding_dimension']),dtype=np.float32)
        pending=[]; positions=[]
        def flush():
            if pending:
                vectors=encode_views(model,pending)
                for position,vector in zip(positions,vectors): features[position]=vector
                pending.clear(); positions.clear()
        for i,row in enumerate(tqdm(rows,desc='E3 shared augmented encoding')):
            path=(DATA/row['path']).resolve()
            if not path.is_relative_to(DATA.resolve()): raise ValueError('Unsafe image path')
            with Image.open(path) as image:
                image=image.convert('RGB')
                for j,view in enumerate(views):
                    pending.append(make_view(image,view)); positions.append((i,j))
                    if len(pending)==args.batch_size: flush()
        flush()
        if cache_path: save_cache(cache_path,metadata,ids,views,features)
    after=parameter_hash(model.model)
    if before!=after or model.model.training or any(p.requires_grad for p in model.model.parameters()): raise ValueError('Frozen model invariant failed')
    for name in selected:
        dest=out/name; dest.mkdir(parents=True,exist_ok=True)
        scores=average_scores(features,prototypes,views,name)
        ranked=model.rank(torch.from_numpy(scores)); predictions=[]
        for row,score,top in zip(rows,scores,ranked):
            pred=classes[int(score.argmax())]
            predictions.append({'image_id':row['image_id'],'path':row['path'],'image_path':row['path'],'ground_truth':row['class'],
                                'prediction':pred,'predicted_class':pred,'top5_classes':json.dumps([t['class'] for t in top]),
                                'top_predictions':json.dumps(top),'similarity_scores':json.dumps(score.tolist()),'correct':pred==row['class']})
        metrics,matrix,pc=compute_metrics([classes.index(r['class']) for r in rows],scores,classes)
        config={'status':'running','experiment':'E3','mode':args.mode,'tta':name,'views':CONFIGURATIONS[name],
                'aggregation':'arithmetic mean of per-view cosine class scores; no probability averaging',
                'text_baseline':'E2 Medium','expanded_prompts':dict(zip(classes,groups)),'class_order':classes,
                'weights_sha256_before':before,'weights_sha256_after':after,'weights_unchanged':True,'model_path':str(args.model_path.resolve()),
                'preprocessing':base['preprocessing'],'device':str(model.device),'batch_size_views':args.batch_size,
                'cache_identity':metadata,'cache_reused':cache_hit,'seed':base['seed'],'library_versions':versions}
        write_json(dest/'experiment_config.json',config)
        write_csv(dest/'evaluation_manifest.csv',rows); write_json(dest/'class_mapping.json',{c:i for i,c in enumerate(classes)})
        write_json(dest/'metrics.json',metrics); write_csv(dest/'predictions.csv',predictions); write_csv(dest/'per_class_accuracy.csv',pc)
        np.savetxt(dest/'confusion_matrix.csv',matrix,fmt='%d',delimiter=',',header=','.join(classes),comments='')
        plots(DATA,dest,classes,matrix,pc,predictions,rows)
        config['status']='complete'; write_json(dest/'experiment_config.json',config)
    if any(digest(path)!=sha for path,sha in protected.items()): raise ValueError('Protected artifacts changed')
    write_json(out/('preservation_'+args.tta+'.json'),{'unchanged':True,'files':protected})
    print('E3 evaluation saved. Run --compare-only separately; no comparisons were automatically generated.')
