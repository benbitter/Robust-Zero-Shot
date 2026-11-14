"""Measured V2 reports only, kept separate from historical ten-class results."""
from production_v2.common import OUT, read, sha

def results():
    rows=[]
    for split in ('val','test'):
        for stamp_path in sorted((OUT/'evaluation'/split).glob('*/*/complete.json')):
            stamp=read(stamp_path); root=stamp_path.parent
            if any(sha(root/name)!=digest for name,digest in stamp['files'].items()):
                raise ValueError('V2 measured report changed')
            m=read(root/'metrics.json'); p=stamp['protocol']
            rows.append({'model':p['model'],'split':split,'condition':p['condition'],
              'top1':m['top1'],'macro_f1':m['macro_f1'],'sources':m['source']})
    return {'rows':rows,'note':'V2 unified taxonomy only. These are not directly comparable to historical ten-class accuracies. No measurements are shown until evaluation completes.'}
