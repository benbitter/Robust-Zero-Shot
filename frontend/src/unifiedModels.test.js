import test from 'node:test';
import assert from 'node:assert/strict';
import {UNIFIED_MODELS,unifiedOptions} from './unifiedModels.js';
test('selector has exactly six fixed models and no archive entries',()=>{
  const options=unifiedOptions([{id:'final',available:true},{id:'m0',available:true}]);
  assert.equal(options.length,6);
  assert.deepEqual(options.map(m=>m.name),['Original CLIP','CLIP + E2','CLIP + E3','CLIP + E4','CLIP + E6','TerraCLIP Ultimate']);
  assert.deepEqual(options.map(m=>m.id),UNIFIED_MODELS.map(m=>m[0]));
  assert.equal(options[0].available,true);assert.equal(options[3].available,false);
});
