"""Manual image validation, duplicate grouping, and reproducible manifests."""
import hashlib
from collections import Counter, defaultdict
from io import BytesIO
import numpy as np
from .common import OUT, ROOT, PARQUET, read, write, sha, identity, log
from .taxonomy import NWPU, EURO, SOURCE_SHA256, SOURCE_IDS, EURO_IDS, build, REPORT

def parquet_file(path):
    try: import pyarrow.parquet as pq
    except ImportError: raise RuntimeError('Install only the 20-class extra: python -m pip install "pyarrow>=17,<24". Do not reinstall torch.') from None
    f = pq.ParquetFile(path)
    import pyarrow as pa
    schema = f.schema_arrow
    if 'image' not in schema.names or 'label' not in schema.names: raise ValueError('Missing image/label columns')
    t = schema.field('image').type
    if not pa.types.is_struct(t) or 'bytes' not in [x.name for x in t] or not pa.types.is_binary(t.field('bytes').type): raise ValueError('Expected image struct with binary bytes')
    if not pa.types.is_integer(schema.field('label').type): raise ValueError('Labels must be integers')
    return f

def image_info(raw):
    from PIL import Image, ImageOps
    with Image.open(BytesIO(raw)) as im:
        fmt, mode, size = im.format, im.mode, im.size
        im.load(); rgb = ImageOps.exif_transpose(im).convert('RGB')
        pixels = rgb.tobytes()
        pixel_sha = hashlib.sha256(str(rgb.size).encode()+pixels).hexdigest()
        thumb = np.asarray(rgb.convert('L').resize((17,16), Image.Resampling.LANCZOS))
        dhash = np.packbits(thumb[:,1:] > thumb[:,:-1]).tobytes().hex()
    return {'bytes_sha256':hashlib.sha256(raw).hexdigest(),'pixel_sha256':pixel_sha,
            'visual_hash':dhash,'size':list(size),'mode':mode,'format':fmt}

def inspect(path=PARQUET, decode=False):
    from production_v2.footer import footer, validate_schema
    info = footer(path)
    validate_schema(info)
    if not decode:
        print(info); return
    # Explicit manual --decode validates every image, resumable by row group.
    from tqdm import tqdm
    fingerprint = sha(path)
    if fingerprint != SOURCE_SHA256: raise ValueError('Parquet differs from pinned source; cannot trust its label mapping')
    pf = parquet_file(path)
    if pf.metadata.num_rows != 31500: raise ValueError('Expected31500 source rows')
    root = OUT/'inspection'; write(root/'source.json', {'path':str(path.resolve()),'sha256':fingerprint,'schema':str(pf.schema_arrow)})
    all_rows = []; errors = []
    offset = 0
    for g in tqdm(range(pf.num_row_groups), desc='Validate NWPU row groups'):
        dest = root/f'group_{g:04d}.json'
        if dest.exists():
            saved = read(dest)
            if saved['source'] != fingerprint: raise ValueError('Inspection provenance differs')
        else:
            records = []; bad = []
            table = pf.read_row_group(g, columns=['image','label']).to_pylist()
            for j, r in enumerate(table):
                idx = offset+j
                try:
                    label = int(r['label'])
                    if not 0 <= label < 45: raise ValueError('Label outside mapping')
                    if label not in SOURCE_IDS: continue
                    raw = r['image']['bytes']
                    if not raw: raise ValueError('Embedded image bytes missing; external paths are not implicitly trusted')
                    image = image_info(raw)
                    records.append({'image_id':f'nwpu:{idx:05d}','source':'nwpu','source_label':label,
                      'source_name':NWPU[label],'parquet_row':idx,'row_group':g,'group_row':j,
                      'embedded_path':r['image'].get('path'),'previously_used_by_final':False,**image})
                except Exception as e: bad.append({'row':idx,'error':str(e)})
            saved = {'source':fingerprint,'rows':records,'errors':bad}; write(dest,saved)
        all_rows.extend(saved['rows']); errors.extend(saved['errors'])
        offset += pf.metadata.row_group(g).num_rows
        log(f'NWPU validated {offset}/{pf.metadata.num_rows}')
    counts = Counter(r['source_name'] for r in all_rows)
    summary = {'source_sha256':fingerprint,'rows':len(all_rows),'class_counts':dict(counts),
      'sizes':dict(Counter(str(r['size']) for r in all_rows)), 'modes':dict(Counter(r['mode'] for r in all_rows)),
      'formats':dict(Counter(r['format'] for r in all_rows)), 'errors':errors,
      'exact_pixel_duplicates':len(all_rows)-len({r['pixel_sha256'] for r in all_rows})}
    write(root/'summary.json',summary)
    if errors or len(counts)!=20 or any(counts[NWPU[i]]!=700 for i in SOURCE_IDS): raise ValueError('Validation failed; inspect inspection/summary.json. No silent dropping.')
    write(root/'complete.json',{'source_sha256':fingerprint,'group_files':{p.name:sha(p) for p in sorted(root.glob('group_*.json'))},'summary_sha256':sha(root/'summary.json')})
    print(summary)

def group_splits(rows, seed=42):
    """Initial source/class stratification, then co-locate exact and visual-hash groups."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i,r in enumerate(rows): strata[(r['source'],r['unified_id'])].append(i)
    for (source,_), indices in sorted(strata.items()):
        order = rng.permutation(indices); n=len(order); a=int(n*.7); b=a+int(n*.15)
        for j,i in enumerate(order): rows[i]['split']='train' if j<a else 'val' if j<b else 'test'
    parent = list(range(len(rows)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    exact={}; visual={}; candidates=[]
    for i,r in enumerate(rows):
        for key, table in [('pixel_sha256',exact),('visual_hash',visual)]:
            h=r[key]
            if h in table:
                j=table[h]
                if key=='pixel_sha256' and r['unified_id']!=rows[j]['unified_id']: raise ValueError(f'Exact duplicate with conflicting labels: {r["image_id"]} / {rows[j]["image_id"]}')
                parent[find(i)]=find(j)
                if key=='visual_hash': candidates.append([rows[j]['image_id'],r['image_id']])
            else: table[h]=i
    groups=defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    priority={'train':0,'val':1,'test':2}; moved=0
    for indices in groups.values():
        split=max((rows[i]['split'] for i in indices),key=priority.get)
        for i in indices:
            moved += rows[i]['split']!=split; rows[i]['split']=split
            rows[i]['duplicate_group']=rows[min(indices)]['image_id']
    return {'moved_to_prevent_leakage':moved,'visual_candidates':candidates,
            'note':'Equal256-bit difference hashes conservatively co-located, not proof of visual duplication. Nonidentical-hash near duplicates/geographic overlap remain unverified.'}

def validate_manifest(rows, count):
    if len({r['image_id'] for r in rows})!=len(rows): raise ValueError('Duplicate image ID')
    groups={}
    for r in rows:
        if not 0<=r['unified_id']<count or r['split'] not in ('train','val','test'): raise ValueError('Invalid label/split')
        for key in ('pixel_sha256','duplicate_group'):
            group=(key,r[key])
            if group in groups and groups[group]!=r['split']: raise ValueError('Duplicate leakage')
            groups[group]=r['split']
    for split in ('train','val','test'):
        if {r['unified_id'] for r in rows if r['split']==split}!=set(range(count)): raise ValueError(f'Class missing in {split}; review grouping')

def prepare(path=PARQUET):
    from src.prepare_dataset import load_splits
    from src.config import SPLITS, DATA
    from tqdm import tqdm
    inspection=OUT/'inspection'; complete=read(inspection/'complete.json')
    if sha(path)!=complete['source_sha256']: raise ValueError('Parquet changed')
    nwpu=[]
    for name,digest in complete['group_files'].items():
        if sha(inspection/name)!=digest: raise ValueError('Inspection changed')
        nwpu.extend(read(inspection/name)['rows'])
    taxonomy,emap,nmap=build()
    root=OUT/'taxonomy'
    write(root/'taxonomy.json',taxonomy); write(root/'eurosat_to_unified.json',emap); write(root/'nwpu_to_unified.json',nmap)
    report=root/'taxonomy_report.md'
    if report.exists() and report.read_text(encoding='utf-8')!=REPORT: raise FileExistsError(report)
    report.write_text(REPORT,encoding='utf-8')
    splits,split_meta=load_splits(SPLITS); euro=[]
    for s in ('train','val','test'):
        for r in tqdm(splits[s],desc='Validate EuroSAT '+s):
            label=int(r['class_index'])
            if label not in EURO_IDS: continue
            if EURO[label]!=r['class']: raise ValueError('EuroSAT mapping differs')
            image_path=(DATA/r['path']).resolve()
            if not image_path.is_relative_to(DATA.resolve()): raise ValueError('Unsafe image path')
            euro.append({'image_id':'eurosat:'+r['image_id'],'source':'eurosat','source_label':label,
              'source_name':r['class'],'path':str(image_path),'original_image_id':r['image_id'],
              'original_split':s,'previously_used_by_final':True,**image_info(image_path.read_bytes())})
    rows=euro+nwpu
    for r in rows: r['unified_id']=(emap if r['source']=='eurosat' else nmap)[str(r['source_label'])]['unified_id']
    dedup=group_splits(rows); validate_manifest(rows,len(taxonomy['classes']))
    rows.sort(key=lambda r:(r['split'],r['source'],r['image_id']))
    manifest={'version':1,'seed':42,'taxonomy_id':identity(taxonomy),'parquet':str(path.resolve()),
      'parquet_sha256':complete['source_sha256'],'original_split_hashes':split_meta['manifest_sha256'],'rows':rows}
    write(OUT/'manifests/manifest.json',manifest)
    stats=Counter((r['source'],r['split'],r['source_name']) for r in rows)
    write(OUT/'manifests/statistics.json',{'rows':len(rows),'strata':[{'source':s,'split':sp,'class':c,'count':n} for (s,sp,c),n in sorted(stats.items())],
      'split_counts':dict(Counter(r['split'] for r in rows)),'deduplication':dedup,
      'evaluation_note':'20-class is trained only on 20-class train. All EuroSAT rows were previously used by the old Final; they are not independent for that old model.'})
    log(f'Prepared {len(rows)} rows, {len(taxonomy["classes"])} classes; inspect manifests/statistics.json')

class Images:
    def __init__(self, manifest): self.manifest=manifest; self.pf=None; self.group=None; self.table=None
    def get(self,row):
        from PIL import Image,ImageOps
        if row['source']=='eurosat':
            from pathlib import Path
            raw=Path(row['path']).read_bytes()
        else:
            if self.pf is None: self.pf=parquet_file(self.manifest['parquet'])
            if self.group!=row['row_group']:
                self.table=self.pf.read_row_group(row['row_group'],columns=['image']).to_pylist(); self.group=row['row_group']
            raw=self.table[row['group_row']]['image']['bytes']
        if hashlib.sha256(raw).hexdigest()!=row['bytes_sha256']: raise ValueError('Source image changed: '+row['image_id'])
        with Image.open(BytesIO(raw)) as im: return ImageOps.exif_transpose(im).convert('RGB').copy()
