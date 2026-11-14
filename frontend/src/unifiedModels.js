export const UNIFIED_MODELS = [
  ['m0','Original CLIP'], ['m1','CLIP + E2'], ['m2','CLIP + E3'],
  ['m3','CLIP + E4'], ['m4','CLIP + E6'], ['m5','TerraCLIP Ultimate'],
];
export function unifiedOptions(models) {
  return UNIFIED_MODELS.map(([id,name])=>({id,name,available:models.find(m=>m.id===id)?.available===true}));
}
