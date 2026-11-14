import json
from src.config import ROOT
from src.utils import digest
from src.tta import identity
E4=ROOT/'results/e4_visual_adapter'
NAMES={'clip':'CLIP + E2 Medium','e4':'E4 Visual Adapter','augmentation':'E6-A Robust Adapter','consistency':'E6-B Consistency Adapter','final':'TerraCLIP Final — Full Dataset'}
NAMES.update(v2='TerraCLIP V2 - Multi-Dataset',v2_clip='CLIP - V2 Unified Taxonomy')

def metadata(): return json.loads((E4/'cache/bundle.json').read_text())
def artifact(name):
    if name=='final':
        base=ROOT/'results/final_production_training/model'
        return base/'final.pt',base/'config.json'
    if name=='e4': return E4/'training/best_adapter.pt',E4/'training/training_config.json'
    base=ROOT/'results/e6_robustness/training'/name
    return base/'best.pt',base/'config.json'
def validate(name,meta):
    path,cfg_path=artifact(name)
    cfg=json.loads(cfg_path.read_text())
    if cfg['status']!='complete' or digest(path)!=cfg['checkpoint_sha256'] or cfg['cache_id']!=identity(meta): raise ValueError('Checkpoint integrity or cache identity mismatch')
    if cfg['architecture']['dimension']!=512: raise ValueError('Embedding dimension mismatch')
    if name=='e4':
        if cfg['class_order']!=meta['classes'] or cfg['expanded_prompts']!=meta['expanded_prompts']: raise ValueError('Class/prompt mapping mismatch')
    elif name=='final':
        if cfg['variant']!='final' or cfg['clip']!=meta['weights_sha256_before'] or cfg['class_order']!=meta['classes'] or cfg['expanded_prompts']!=meta['expanded_prompts'] or cfg['train_count']!=27000 or cfg['independent_test_accuracy'] is not None: raise ValueError('Production metadata mismatch')
    elif cfg['variant']!=name or cfg['clip']!=meta['weights_sha256_before']: raise ValueError('Adapter identity mismatch')
    return path,cfg
