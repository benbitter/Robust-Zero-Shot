import json
from src.config import ROOT
from backend.model_registry import NAMES

def read_results():
    rows=[]
    paths=[('E0','e0_original_clip_eurosat/validation','Zero-shot'),('E1','e1_domain_prompts','Zero-shot'),('E2 Medium','e2_prompt_ensemble/medium','Zero-shot'),('E3 TTA','e3_test_time_augmentation/validation/rotations_flips','Zero-shot'),('E4','e4_visual_adapter/e4_a','Supervised')]
    for name,path,category in paths:
        p=ROOT/'results'/path/'metrics.json'
        if p.exists():
            m=json.loads(p.read_text()); rows.append({'experiment':name,'condition':'clean','top1_accuracy':m['top1_accuracy'],'macro_f1':m.get('macro_f1'),'category':category,'source':p.relative_to(ROOT).as_posix()})
    for name in NAMES:
        base=ROOT/'results/e6_robustness/evaluations'/name
        for p in sorted(base.glob('*/metrics.json')):
            cfg=json.loads((p.parent/'config.json').read_text()); m=json.loads(p.read_text())
            rows.append({'experiment':NAMES[name],'condition':cfg['condition'],'top1_accuracy':m['top1_accuracy'],'macro_f1':m.get('macro_f1'),'category':'Zero-shot' if name=='clip' else 'Supervised','source':p.relative_to(ROOT).as_posix()})
    return {'rows':rows,'note':'Historical EuroSAT validation results; no TTA except E3. Validation selected adapter checkpoints. Not independent test accuracy or upload accuracy.'}
