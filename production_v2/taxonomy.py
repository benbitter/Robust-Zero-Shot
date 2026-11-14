"""Pinned source-label mapping; never derive numerical labels by sorting names."""
SOURCE = 'https://huggingface.co/datasets/yangjiahao-x/NWPU-RESISC45/blob/966bc7b/README.md'
SOURCE_SHA256 = '66b7da94f92c3435e3c6f8230f17efe8961b6f6b53dadfada62167da4f683e29'
NWPU = ['airplane','airport','baseball diamond','basketball court','beach','bridge','chaparral','church','circular farmland','cloud','commercial area','dense residential','desert','forest','freeway','golf course','ground track field','harbor','industrial area','intersection','island','lake','meadow','medium residential','mobile home park','mountain','overpass','palace','parking lot','railway','railway station','rectangular farmland','river','roundabout','runway','sea ice','ship','snowberg','sparse residential','stadium','storage tank','tennis court','terrace','thermal power station','wetland']
DETAILS = [
 'aircraft parked on the ground', 'airport runways, taxiways and terminals',
 'a baseball field with a diamond-shaped infield', 'a marked basketball playing court',
 'a sandy beach beside water', 'a bridge spanning a river or road',
 'dense low shrubs and scrubland', 'a church building and surrounding grounds',
 'circular crop fields formed by center-pivot irrigation', 'cloud cover viewed from above',
 'commercial buildings, shops and paved lots', 'closely packed residential buildings',
 'barren sandy or rocky desert terrain', 'continuous tree-covered forest',
 'a broad multi-lane freeway', 'golf fairways, greens and sand bunkers',
 'an oval running track surrounding a sports field', 'a harbor with docks and sheltered water',
 'factory buildings and industrial infrastructure', 'intersecting roads and junctions',
 'an island surrounded by water', 'an inland lake', 'open grass-covered meadow',
 'moderately spaced residential buildings', 'rows of mobile homes in a residential park',
 'mountain ridges and elevated rocky terrain', 'a road overpass crossing another road',
 'a palace complex with formal grounds', 'a paved parking lot with marked spaces',
 'parallel railway tracks', 'a railway station with tracks and platforms',
 'rectangular agricultural fields', 'a flowing river channel',
 'a circular road junction with a central island', 'a long aircraft runway',
 'floating or continuous sea ice', 'a vessel on water', 'an iceberg or snow-covered ice formation',
 'widely spaced residential houses', 'a large sports stadium with surrounding stands',
 'circular industrial storage tanks', 'marked tennis courts', 'terraced hillside fields',
 'a thermal power plant with cooling and generation infrastructure', 'marshy wetlands and shallow water']
EURO = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
EURO_DETAIL = {
 'AnnualCrop':'agricultural fields of annually planted crops',
 'HerbaceousVegetation':'land covered by herbaceous plants and low vegetation',
 'Highway':'highway transport corridors in a land-cover patch',
 'Industrial':'industrial land cover including factory roofs and paved yards',
 'Pasture':'grazing pasture and managed grassland',
 'PermanentCrop':'perennial crop plantations, orchards or vineyards',
 'Residential':'residential land cover without a specified housing density',
 'SeaLake':'open water land cover including both seas and lakes'}

def build():
    categories = []
    for i, (name, detail) in enumerate(zip(NWPU, DETAILS)):
        categories.append({'id':i, 'name':name, 'description':detail, 'sources':['nwpu'],
          'prompts':[f'An aerial satellite image of {name}.', f'An overhead view showing {detail}.', f'A remote sensing scene with {detail}.']})
    emap = {}
    for i, name in enumerate(EURO):
        if name in ('Forest','River'):
            target = {'Forest':13,'River':32}[name]; categories[target]['sources'].append('eurosat')
        else:
            target = len(categories); detail = EURO_DETAIL[name]
            categories.append({'id':target,'name':'EuroSAT '+name,'description':detail,'sources':['eurosat'],
              'prompts':[f'A satellite image of {detail}.', f'An overhead land-cover patch showing {detail}.', f'A remote sensing image of {detail}.']})
        emap[str(i)] = {'original_name':name,'unified_id':target}
    nmap = {str(i):{'original_name':name,'unified_id':i} for i,name in enumerate(NWPU)}
    return {'version':1,'source_mapping':SOURCE,'source_parquet_sha256':SOURCE_SHA256,'classes':categories}, emap, nmap

REPORT = '''# V2 taxonomy decisions
53 stable classes: NWPU IDs0-44 are retained from the pinned dataset card; eight EuroSAT-only IDs follow.
Forest and River denote the same broad land-cover concept and are merged (13 and32), while source provenance remains recorded. This is a modeling decision, not proof that annotation policies or spatial scales are identical.
EuroSAT Residential is NOT assigned a housing-density category. SeaLake is NOT equivalent to Lake. Pasture, HerbaceousVegetation, AnnualCrop and PermanentCrop are NOT equated to meadow, chaparral or field geometry. Highway/freeway and Industrial/industrial area remain source-qualified because scope and annotation scale are unverified.
These intentionally overlapping labels create a harder, potentially ambiguous closed-set task. Review source-qualified categories before any experiment; changing taxonomy requires a fresh output root/version, not editing trained artifacts.
Prompt descriptions are newly authored. No old ten-class prototype matrix is reused.
'''
