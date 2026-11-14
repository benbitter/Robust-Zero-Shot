import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI,File,Form,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.inference_service import InferenceService
from backend.image_processing import decode,degrade,preview,MAX_BYTES
from backend.model_registry import metadata
from backend.schemas import Health,ModelInfo,Classes,Predictions,Robustness,Research
from backend.research import read_results
service=InferenceService()
@asynccontextmanager
async def lifespan(app):
    try: await asyncio.to_thread(service.startup)
    except Exception: service.error='Local model startup failed. Verify dependencies and checkpoint compatibility.'
    yield
app=FastAPI(title='TerraCLIP',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
@app.middleware('http')
async def limit_request(request,call_next):
    length=request.headers.get('content-length')
    if length:
        try:
            if int(length)>11*1024*1024: return JSONResponse({'detail':'Request exceeds upload limit.'},status_code=413)
        except ValueError: return JSONResponse({'detail':'Invalid request length.'},status_code=400)
    return await call_next(request)

@app.get('/api/health',response_model=Health)
def health(): return {'ready':service.ready,'message':service.error or 'Local model service ready'}
@app.get('/api/models',response_model=list[ModelInfo])
def models(): return service.listing()
@app.get('/api/project')
def project(): return {'unified':hasattr(service,'unified'),'ready':service.ready}
@app.get('/api/unified/final-benchmark')
def final_benchmark_results():
    from backend.final_benchmark_results import results
    try: return results()
    except (OSError,ValueError,KeyError): raise HTTPException(503,'Final benchmark report integrity check failed') from None
@app.get('/api/unified/final-benchmark/plots/{name}')
def final_benchmark_plot(name:str):
    from backend.final_benchmark_results import plot
    from fastapi.responses import FileResponse
    try: return FileResponse(plot(name))
    except (OSError,ValueError,KeyError): raise HTTPException(404,'Committed benchmark plot unavailable') from None
@app.get('/api/unified/research-results')
def unified_research():
    from backend.portable import available,report
    if available():
        try:return report('research.json')
        except (OSError,ValueError,KeyError):raise HTTPException(503,'Deployment research integrity check failed') from None
    if not hasattr(service,'unified'): return {'rows':[],'note':'Unified benchmark is not activated.'}
    from production_20.service import reports
    try: return reports()
    except (OSError,ValueError,KeyError): raise HTTPException(503,'Unified report integrity check failed') from None
@app.get('/api/classes',response_model=Classes)
def classes(): return {'classes':service.classes if hasattr(service,'unified') else metadata()['classes']}
@app.get('/api/research-results',response_model=Research)
def research(): return read_results()
@app.get('/api/v2/research-results')
def v2_research():
    from backend.v2_results import results
    try: return results()
    except (OSError,ValueError,KeyError): raise HTTPException(503,'V2 report integrity check failed') from None
async def image(file):
    try:
        data=await file.read(MAX_BYTES+1)
        return decode(data)
    except ValueError as e: raise HTTPException(422,str(e)) from None
    finally: await file.close()
async def predict_image(img,names):
    try: return await asyncio.to_thread(service.predict,img,names)
    except ValueError as e: raise HTTPException(503,str(e)) from None
    except Exception as exc:
        import torch
        if isinstance(exc,torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            raise HTTPException(503,'CUDA out of memory. Stop other GPU processes and retry; no CPU fallback was performed.') from None
        raise HTTPException(500,'Inference failed; verify model compatibility and available memory.') from None
@app.post('/api/predict',response_model=Predictions)
async def predict(file:UploadFile=File(...),model:str=Form('augmentation')):
    if hasattr(service,'unified') and model=='augmentation': model='m4'
    return await predict_image(await image(file),[model])
@app.post('/api/compare',response_model=Predictions)
async def compare(file:UploadFile=File(...),model:str=Form('augmentation')):
    if hasattr(service,'unified'):
        from production_20.taxonomy import MODELS
        return await predict_image(await image(file),list(MODELS))
    names=['v2_clip','v2'] if model.startswith('v2') else [x['id'] for x in service.listing() if x['available'] and not x['id'].startswith('v2')]
    return await predict_image(await image(file),names)
@app.post('/api/robustness',response_model=Robustness)
async def robustness(file:UploadFile=File(...),kind:str=Form('resolution'),severity:int=Form(32),model:str=Form('augmentation')):
    try:
        img=await image(file)
        if model.startswith('v2') or hasattr(service,'unified'):
            if hasattr(service,'unified'):
                from production_20.features import corrupt
            else:
                from production_v2.features import corrupt
            levels={'resolution':[64,48,32,16,8],'blur':[0,1,2,3],'noise':[0,5,10,15],'jpeg':[100,80,60,40]}
            if kind not in levels or severity not in levels[kind]: raise ValueError('Invalid corruption setting')
            original=img; changed=corrupt(img,f'{kind}:{severity}','web-upload')
        else: original,changed=degrade(img,kind,severity)
    except ValueError as e: raise HTTPException(422,str(e)) from None
    names=[model if model!='augmentation' else 'm4'] if hasattr(service,'unified') else (['v2_clip','v2'] if model.startswith('v2') else ['e4','augmentation'])
    return {'original_preview':preview(original),'degraded_preview':preview(changed),'original':await predict_image(original,names),'degraded':await predict_image(changed,names),'note':('Uses native upload dimensions.' if model.startswith('v2') or hasattr(service,'unified') else 'Both inputs use a 64x64 bicubic canvas.')+' Relative scores are not calibrated probabilities. No ground truth supplied.'}
