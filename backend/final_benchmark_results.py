"""Serve only committed, integrity-checked final benchmark summaries."""
from production_20.common import OUT,read,sha,identity
ROOT=OUT/'final_benchmark'
from production_20.benchmark import frozen

def results():
    from backend.portable import available,final_results
    if available():return final_results()
    marker=ROOT/'report_complete.json'
    if not marker.exists():return {'available':False,'note':'Final benchmark report has not been generated. No placeholder metrics.'}
    commit=read(marker);summary_path=ROOT/'summary.json'
    if sha(summary_path)!=commit['files']['summary.json']:raise ValueError('Final summary checksum differs')
    s=read(summary_path)
    if s['protocol_id']!=commit['protocol_id'] or s['selection_id']!=identity(frozen()):raise ValueError('Final benchmark uses different frozen artifacts')
    # Return plot links only for files in the report's committed export set.
    s['plot_urls']=['/api/unified/final-benchmark/plots/'+name.split('/')[-1] for name in commit['files'] if name.startswith('plots/') and name.endswith('.png')]
    return s

def plot(name):
    from backend.portable import available,plot as portable_plot
    if available():return portable_plot(name)
    if '/' in name or '\\' in name or not name.endswith(('.png','.svg')):raise ValueError('Invalid plot name')
    commit=read(ROOT/'report_complete.json');relative='plots/'+name
    if relative not in commit['files'] or sha(ROOT/relative)!=commit['files'][relative]:raise ValueError('Plot not committed or changed')
    return ROOT/relative
