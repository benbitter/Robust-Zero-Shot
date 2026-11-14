# TerraCLIP — 20-Class Satellite Scene Classifier

React + FastAPI app: upload a JPEG/PNG, inspect Top-5 predictions, compare six models, apply corruptions, and view measured research results. All six share frozen CLIP ViT-B/16 and the same class order.

**No dataset, training or feature extraction is required.** Weights use Git LFS. Downloading GitHub's source ZIP alone may leave weight pointers instead of actual models.

## Prerequisites

Windows 10/11, Python **3.11**, Node.js **20 or 22 LTS**, Git and **Git LFS**. CPU inference works; NVIDIA CUDA is optional. Development hardware: RTX 3050 Laptop GPU, 6 GB VRAM, 16 GB RAM. Allow several GB for dependencies and LFS's local object copy. Model/artifact downloads are approximately 610 MB.

## 1. Clone and fetch weights (PowerShell)

Install Git LFS first if `git lfs version` is not recognized.

```powershell
git lfs install
git clone https://github.com/benbitter/Robust-Zero-Shot.git
Set-Location .\Robust-Zero-Shot
git lfs pull
git lfs ls-files
py -3.11 -m venv .venv
```

For an existing clone, run `git pull` and `git lfs pull`. The CLIP `pytorch_model.bin` should be approximately 599 MB, not a small text pointer. Resolve any GitHub LFS quota/access errors before startup.

## 2. Install dependencies

Choose **one** PyTorch command for the new virtual environment.

CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.6 (compatible NVIDIA driver required):

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m backend.portable
```

The integrity check hashes files and checks prototypes without loading or executing a model. No Anaconda, datasets, training manifests, PyArrow or feature caches are required. If your existing CUDA environment already works, do not reinstall its PyTorch. Inference uses local weights only; no model download occurs at startup.

## 3. Start backend — terminal 1, repository root

```powershell
$env:TERRACLIP_DEVICE = 'auto'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`auto` chooses CUDA when available, otherwise CPU; `cpu` forces CPU, and `cuda` requires CUDA without silent fallback. Startup loads real models. Use one process, without multiple workers or `--reload`. Stop other TerraCLIP GPU processes first.

Wait for `ready: true` at http://127.0.0.1:8000/api/health. API docs: http://127.0.0.1:8000/docs.

## 4. Start frontend — terminal 2

From the repository root:

```powershell
Set-Location .\frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to port 8000. Keep both terminals running; Ctrl+C stops each server. If PowerShell blocks npm.ps1, use `npm.cmd ci` and `npm.cmd run dev` instead.

## Six models

| ID | Name | Method |
|---|---|---|
| M0 | Original CLIP | One simple prompt per class |
| M1 | CLIP + E2 | Five domain-specific prompts per class |
| M2 | CLIP + E3 | E2 with eight rotations/flips |
| M3 | CLIP + E4 | Clean-trained residual adapter |
| M4 | CLIP + E6 | Corruption-trained residual adapter; default |
| M5 | TerraCLIP Ultimate | Eight-view score ensemble: 25% M3 + 75% M4 |

Categories: airplane, airport, beach, bridge, circular farmland, commercial area, dense residential, desert, forest, freeway, harbor, industrial area, lake, mountain, parking lot, railway, rectangular farmland, river, runway, wetland.

M3/M4 are supervised, not zero-shot. Saved test metrics concern 2,945 images (825 EuroSAT Forest/River and 2,120 NWPU images), not arbitrary uploaded images. Relative scores are not calibrated probabilities. Historical ten-class weights/results are not distributed; the separate archive can be empty in a fresh clone.

## Included files

- `frontend/`, `backend/`: app and API.
- `deployment/`: selected adapter checkpoints, text prototypes, taxonomy, checksums and measured dashboard assets.
- `models/clip-vit-base-patch16/`: local frozen CLIP; large weights use Git LFS.
- Selected `src/`, `production_20/`, `production_v2/` modules: shared runtime definitions. Importing them does not start training.
- `requirements-app.txt`: app-only requirements.

Excluded: datasets, feature caches, full research runs, optimizer checkpoints, virtual environments, node_modules, build output and credentials. Original local research files are preserved. Only M3/M4 require adapter checkpoints; M5 reuses both.

## Lightweight verification

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_portable
npm --prefix frontend run build
node --test frontend/src/unifiedModels.test.js
```

Tests mock heavy components; no pretrained inference occurs. Actual inference after cloning remains a manual check. The app shares one CLIP encoder and the necessary view embeddings across selected models.

## Troubleshooting

- Missing weights/hash mismatch: run `git lfs pull`, then the integrity command. Do not bypass checks by editing hashes.
- CUDA out of memory: stop competing GPU processes, use a single-view model, or restart with `TERRACLIP_DEVICE=cpu`.
- Frontend cannot classify: check the backend terminal and health endpoint; allow model startup time.
- Backend import errors: use Python 3.11, the documented venv and the repository root.

This is a localhost research demo, not a hardened public service. Uploaded images are processed in memory; no external inference service is used. Authentication and rate limiting are not included. Upstream CLIP attribution and artifact provenance are in `deployment/README.md`.
