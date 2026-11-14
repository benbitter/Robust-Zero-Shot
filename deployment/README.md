# Frozen application artifacts

This directory packages the existing 20-class six-model deployment without dataset images, dataset split manifests, feature caches, training histories, optimizer checkpoints or training jobs.

- `m3.pt`: original selected clean adapter, unchanged bytes, selected epoch 14.
- `m4.pt`: original selected robustness adapter, unchanged bytes, selected epoch 48.
- `prototypes.npz`: original simple and five-prompt E2 matrices and frozen logit scale.
- `taxonomy.json`, `prompts.json`: fixed categories and exact text configuration.
- `manifest.json`: file checksums, original frozen selection identity and checkpoint configuration hashes.
- `research.json`, `final_summary.json`, `plots/`: saved measured results only; no new evaluations.

The original checkpoints and results remain in their local research directories. Deployment copies are bound to the same artifacts by SHA-256. CLIP remains under `models/clip-vit-base-patch16/` and is tracked using Git LFS. Only M3/M4 need adapter weights: M0/M1/M2 use frozen CLIP, and M5 composes the same M3/M4 adapters with 0.25/0.75 weights and eight-view TTA.

Verify without running a model: `python -m backend.portable`.

Original CLIP is an OpenAI pretrained model. Its upstream source and MIT license are at https://github.com/openai/CLIP. This repository does not redistribute EuroSAT or NWPU imagery. Dataset names and metrics identify the training/evaluation sources, not a new independent accuracy guarantee on arbitrary uploads.
