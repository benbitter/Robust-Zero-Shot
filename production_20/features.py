"""Manual frozen CLIP features. Native-resolution corruptions, atomic chunks."""
import hashlib
import time
from io import BytesIO
import numpy as np
from .common import OUT, ROOT, artifacts, backbone, load_clip, identity, write, read, save_npz, sha, vectors, log
TRAIN_CONDITIONS=['clean','resolution:48','resolution:32','resolution:16','blur:1']
TEST_CONDITIONS=['clean','resolution:32','resolution:16','blur:2','noise:15','jpeg:40']
CORRUPTION_VERSION='20-native-canvas-bicubic-blur-noise-jpeg-v1'
from src.tta import CONFIGURATIONS, VIEW_VERSION, make_view
VIEWS=list(CONFIGURATIONS['rotations_flips'])

def corrupt(image, condition, image_id):
    from PIL import Image,ImageFilter
    image=image.convert('RGB')
    if condition=='clean': return image
    kind,value=condition.split(':'); v=int(value)
    if kind=='resolution': return image.resize((v,v),Image.Resampling.BICUBIC).resize(image.size,Image.Resampling.BICUBIC)
    if kind=='blur': return image.filter(ImageFilter.GaussianBlur(v))
    if kind=='noise':
        seed=int.from_bytes(hashlib.sha256(f'42|{image_id}|{condition}'.encode()).digest()[:8],'little')
        a=np.asarray(image).astype(np.float32); noise=np.random.default_rng(seed).normal(0,v,a.shape)
        return Image.fromarray(np.clip(np.rint(a+noise),0,255).astype(np.uint8))
    if kind=='jpeg':
        b=BytesIO(); image.save(b,format='JPEG',quality=v,subsampling=0); b.seek(0)
        with Image.open(b) as im: return im.convert('RGB').copy()
    raise ValueError('Unknown corruption '+condition)

def text(args):
    import torch
    manifest,taxonomy=artifacts(); root=OUT/'text'
    cfg={'taxonomy_id':identity(taxonomy),'backbone':backbone(),'dimension':512,
         'simple_prompts':[[c['simple_prompt']] for c in taxonomy['classes']],
         'prompts':[c['prompts'] for c in taxonomy['classes']],'normalization':'normalize each, mean, normalize'}
    write(root/'config.json',cfg)
    if (root/'complete.json').exists(): load_text(taxonomy); log('Verified existing text prototypes'); return
    if (root/'prototypes.npz').exists(): raise FileExistsError('Uncommitted prototype artifact; preserve and use a new 20-class output root')
    model,processor=load_clip(args.device); result=[]
    arrays={}
    with torch.inference_mode():
        for key,groups in [('prototypes',cfg['prompts']),('simple',cfg['simple_prompts'])]:
            result=[]
            for prompts in groups:
                tokens=processor(text=prompts,padding=True,return_tensors='pt').to(args.device)
                z=model.get_text_features(**tokens).float(); z=torch.nn.functional.normalize(z,dim=-1)
                result.append(torch.nn.functional.normalize(z.mean(0),dim=-1).cpu().numpy())
            arrays[key]=np.stack(result).astype(np.float32); vectors(arrays[key],20)
        scale=float(model.logit_scale.exp().cpu())
    save_npz(root/'prototypes.npz',**arrays,scale=np.float32(scale))
    write(root/'complete.json',{'config_id':identity(cfg),'sha256':sha(root/'prototypes.npz')})

def load_text(taxonomy,kind="e2"):
    root=OUT/'text'; cfg=read(root/'config.json'); stamp=read(root/'complete.json')
    if cfg['taxonomy_id']!=identity(taxonomy) or cfg['prompts']!=[c['prompts'] for c in taxonomy['classes']] or stamp['config_id']!=identity(cfg) or sha(root/'prototypes.npz')!=stamp['sha256']: raise ValueError('Prototype provenance mismatch')
    if cfg['simple_prompts']!=[[c['simple_prompt']] for c in taxonomy['classes']]: raise ValueError('Simple prompts differ')
    if kind not in ('e2','simple'): raise ValueError('Invalid prototype family')
    with np.load(root/'prototypes.npz',allow_pickle=False) as z: t=z['prototypes' if kind=='e2' else 'simple'].copy(); scale=float(z['scale'])
    vectors(t,len(taxonomy['classes']))
    if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid logit scale')
    return t,scale,cfg

def cache_root(split,condition,view='r0'): return OUT/'cache'/split/condition.replace(':','_')/view

def read_chunk(path, rows, config_id):
    stamp=read(path.with_suffix('.json'))
    if stamp['sha256']!=sha(path) or stamp['config_id']!=config_id: raise ValueError('Chunk checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        a=z['features'].copy()
        if z['ids'].tolist()!=[r['image_id'] for r in rows] or z['labels'].tolist()!=[r['unified_id'] for r in rows]: raise ValueError('Cache label/order mismatch')
    vectors(a,len(rows)); return a

def load_features(manifest,split,condition,view="r0"):
    rows=[r for r in manifest['rows'] if r['split']==split]; root=cache_root(split,condition,view)
    cfg=read(root/'config.json'); stamp=read(root/'complete.json')
    if cfg['manifest_id']!=identity(manifest) or cfg['view']!=view or cfg['view_version']!=VIEW_VERSION or cfg['condition']!=condition or cfg['corruption_version']!=CORRUPTION_VERSION or stamp['config_id']!=identity(cfg): raise ValueError('Cache provenance mismatch')
    if cfg['backbone']!=read(OUT/'text/config.json')['backbone']: raise ValueError('Cache/text backbone mismatch')
    pieces=[]
    for start in range(0,len(rows),cfg['chunk_size']):
        p=root/f'{start:06d}.npz'
        if stamp['files'].get(p.name)!=sha(p): raise ValueError('Cache changed')
        pieces.append(read_chunk(p,rows[start:start+cfg['chunk_size']],identity(cfg)))
    return np.concatenate(pieces),rows

class EuroReuse:
    """Clean image features only, independent of the old ten-class prototypes."""
    def __init__(self, model, processor):
        from src.clip_baseline import parameter_hash
        from src.utils import read_csv
        self.root=ROOT/'results/final_production_training/cache'
        meta=read(ROOT/'results/e4_visual_adapter/cache/bundle.json')
        if parameter_hash(model)!=meta['weights_sha256_before'] or processor.image_processor.to_dict()!=meta['preprocessing']: raise ValueError('Old EuroSAT encoder/processor incompatible')
        stamp=read(self.root/'complete.json')
        if sha(self.root/'clean.npy')!=stamp['features']['clean.npy'] or sha(self.root/'full_manifest.csv')!=stamp['manifest_sha256']: raise ValueError('Old cache damaged')
        self.features=np.load(self.root/'clean.npy',mmap_mode='r')
        self.rows=read_csv(self.root/'full_manifest.csv'); self.lookup={r['image_id']:i for i,r in enumerate(self.rows)}
        self.fingerprints={}
        for s in ('train','val','test'):
            p=ROOT/'results/e6_robustness/cache'/s/'clean/config.json'
            if not p.exists(): p=ROOT/'results/final_test_evaluation/cache'/s/'clean/config.json'
            self.fingerprints.update(read(p)['fingerprints'])
    def get(self,row):
        if self.fingerprints.get(row['original_image_id'])!=row['bytes_sha256']: raise ValueError('Old cache image fingerprint differs')
        i=self.lookup[row['original_image_id']]
        if self.rows[i]['class']!=row['source_name']: raise ValueError('Old cache source-label mismatch')
        return np.array(self.features[i],copy=True)

def extract(args):
    import torch
    from tqdm import tqdm
    from .data import Images
    manifest,taxonomy=artifacts(); _,_,textcfg=load_text(taxonomy)
    if 'test' in args.splits:
        from .benchmark import frozen
        frozen()
    weights=backbone()
    if weights!=textcfg['backbone']: raise ValueError('Local CLIP differs from prototype encoder')
    if sha(manifest['parquet'])!=manifest['parquet_sha256']: raise ValueError('Parquet changed')
    images=Images(manifest); model=processor=reuse=None
    for split in args.splits:
        rows=[r for r in manifest['rows'] if r['split']==split]
        from itertools import product
        for condition,view in product(args.conditions,VIEWS if args.views=='eight' else ['r0']):
            root=cache_root(split,condition,view)
            cfg={'manifest_id':identity(manifest),'backbone':weights,'condition':condition,
              'split':split,'view':view,'view_version':VIEW_VERSION,'corruption_version':CORRUPTION_VERSION,'chunk_size':128,'dimension':512,'dtype':'float32'}
            write(root/'config.json',cfg); config_id=identity(cfg); hashes={}; started=time.perf_counter()
            with tqdm(total=len(rows),desc=split+'/'+condition+'/'+view,unit='image') as bar:
                for start in range(0,len(rows),128):
                    part=rows[start:start+128]; dest=root/f'{start:06d}.npz'
                    if dest.exists():
                        # A crash between data and commit can recover only after full array/ID validation.
                        if not dest.with_suffix('.json').exists():
                            with np.load(dest,allow_pickle=False) as z:
                                vectors(z['features'],len(part))
                                if z['ids'].tolist()!=[r['image_id'] for r in part] or z['labels'].tolist()!=[r['unified_id'] for r in part]: raise ValueError('Uncommitted cache order differs')
                            write(dest.with_suffix('.json'),{'sha256':sha(dest),'config_id':config_id})
                        read_chunk(dest,part,config_id)
                    else:
                        if model is None:
                            model,processor=load_clip(args.device)
                            if args.reuse_eurosat_clean: reuse=EuroReuse(model,processor)
                        pieces=[]
                        for b in range(0,len(part),args.batch_size):
                            batch=part[b:b+args.batch_size]; array=np.empty((len(batch),512),np.float32); missing=[]; loaded=[]
                            for j,row in enumerate(batch):
                                if reuse is not None and row['source']=='eurosat' and condition=='clean' and view=='r0': array[j]=reuse.get(row)
                                else: missing.append(j); loaded.append(make_view(corrupt(images.get(row),condition,row['image_id']),view))
                            if missing:
                                with torch.inference_mode():
                                    pixels=processor(images=loaded,return_tensors='pt')['pixel_values'].to(args.device)
                                    z=model.get_image_features(pixel_values=pixels).float()
                                    array[missing]=torch.nn.functional.normalize(z,dim=-1).cpu().numpy()
                            pieces.append(array)
                        array=np.concatenate(pieces); vectors(array,len(part))
                        save_npz(dest,features=array,ids=np.array([r['image_id'] for r in part]),labels=np.array([r['unified_id'] for r in part]))
                        write(dest.with_suffix('.json'),{'sha256':sha(dest),'config_id':config_id})
                    hashes[dest.name]=sha(dest); bar.update(len(part))
                    if start%1024==0:
                        log(f'{split}/{condition} {start+len(part)}/{len(rows)}; {(start+len(part))/max(.001,time.perf_counter()-started):.1f} cached-or-encoded images/s')
                        if args.device=='cuda': log(f'GPU allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
            write(root/'complete.json',{'config_id':config_id,'files':hashes,'rows':len(rows)})
