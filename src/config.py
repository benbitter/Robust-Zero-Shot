from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'EuroSAT_RGB' / 'EuroSAT_RGB'
SPLITS = ROOT / 'data' / 'splits'
RESULTS = ROOT / 'results' / 'e0_original_clip_eurosat'
EUROSAT_COUNTS = {'AnnualCrop': 3000, 'Forest': 3000, 'HerbaceousVegetation': 3000,
                  'Highway': 2500, 'Industrial': 2500, 'Pasture': 2000,
                  'PermanentCrop': 2500, 'Residential': 3000, 'River': 2500, 'SeaLake': 3000}
EUROSAT_PROMPTS = dict(zip(sorted(EUROSAT_COUNTS), [
    'a photo of an annual crop field', 'a photo of a forest',
    'a photo of herbaceous vegetation', 'a photo of a highway',
    'a photo of an industrial area', 'a photo of a pasture',
    'a photo of a permanent crop field', 'a photo of a residential area',
    'a photo of a river', 'a photo of a sea or lake']))
MODEL = 'openai/clip-vit-base-patch16'
PROMPT = 'fixed EuroSAT baseline mapping (see EUROSAT_PROMPTS)'
