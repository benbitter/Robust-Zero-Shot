"""Fixed user-specified order, pinned NWPU source IDs, conservative harmonization."""
from production_v2.taxonomy import NWPU, DETAILS, EURO, SOURCE, SOURCE_SHA256

CLASSES = ['airplane','airport','beach','bridge','circular_farmland','commercial_area',
 'dense_residential','desert','forest','freeway','harbor','industrial_area','lake',
 'mountain','parking_lot','railway','rectangular_farmland','river','runway','wetland']
SOURCE_IDS = [0,1,4,5,8,10,11,12,13,14,17,18,21,25,28,29,31,32,34,44]
EURO_IDS = {1:8,8:17}
MODELS = {'m0':'Original CLIP','m1':'CLIP + E2','m2':'CLIP + E3',
          'm3':'CLIP + E4','m4':'CLIP + E6','m5':'TerraCLIP Ultimate'}

def build():
    categories=[]
    for i,(name,source_id) in enumerate(zip(CLASSES,SOURCE_IDS)):
        if NWPU[source_id].replace(' ','_')!=name: raise ValueError('Pinned NWPU mapping changed')
        detail=DETAILS[source_id]; readable=name.replace('_',' ')
        categories.append({'id':i,'name':name,'description':detail,'sources':['nwpu']+(['eurosat'] if i in EURO_IDS.values() else []),
          'simple_prompt':f'A photo of {readable}.',
          'prompts':[f'An aerial satellite image of {readable}.',f'An overhead view showing {detail}.',f'A remote sensing scene with {detail}.',f'A satellite photograph showing {detail}.',f'An aerial view of {readable} from above.']})
    emap={str(k):{'original_name':EURO[k],'unified_id':v} for k,v in EURO_IDS.items()}
    nmap={str(k):{'original_name':NWPU[k],'unified_id':i} for i,k in enumerate(SOURCE_IDS)}
    return {'version':'terraclip20-v1','source_mapping':SOURCE,'source_parquet_sha256':SOURCE_SHA256,'classes':categories},emap,nmap

REPORT='''# Conservative 20-class harmonization, v1
The20 class IDs are fixed in the requested order; source NWPU integers come from its pinned dataset card, never a guessed sort.
Included EuroSAT: Forest -> forest (ID8), River -> river (ID17). Both name the same broad scene/land-cover concept, but spatial resolution and annotation scale differ; provenance and source-specific metrics remain mandatory.
Excluded EuroSAT Industrial: EuroSAT describes industrial buildings/land cover; NWPU describes an industrial-area scene. Plausible overlap is not sufficient evidence of equivalent annotation scope. No Industrial rows are included without a future separately versioned mapping review.
Also excluded: Residential, AnnualCrop, PermanentCrop, Highway, SeaLake, HerbaceousVegetation, Pasture. None is automatically equated to residential density, farmland geometry, freeway, lake or another selected label.
Sources: https://github.com/phelber/EuroSAT ; https://gcheng-nwpu.github.io/ ; pinned NWPU card in taxonomy.json.
No per-image semantic audit was performed automatically. The mappings are documented broad-semantic research choices, not proof of identical labeling policies.
New seeded70/15/15 source/class splits retain original EuroSAT split provenance. Exact decoded-pixel duplicates and identical256-bit visual-hash candidates are co-located. This does not prove spatial independence or eliminate every near duplicate.
All included EuroSAT rows were used by old TerraCLIP Final. Old models/results are excluded from this new benchmark.
'''
