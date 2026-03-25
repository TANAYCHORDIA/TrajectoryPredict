# TrajectoryPredict

TrajectoryPredict is a 7-day hackathon project focused on **predicting future trajectories of pedestrians and cyclists** in urban scenes using the **nuScenes** dataset.

Our objective is simple: ship a **working, reproducible, demo-ready** system that achieves qualifying metrics for Round 1.

---

## Executive Summary

Autonomous systems need to predict not only where people are now, but where they are likely to be in the next few seconds.  
Given 2 seconds of observed trajectory, we predict 3 seconds into the future with **multimodal outputs** (3 possible futures), because human behavior is uncertain.

This repo is the team’s **single source of truth** for:
- technical decisions,
- daily execution,
- ownership,
- coding standards,
- GitHub workflow.

---

## Problem Definition

### Input
- Observed trajectory: `[(x1,y1), ... , (x8,y8)]`
- 2 seconds at 4Hz = 8 timesteps
- Features per timestep: `(x, y, dx, dy)` → shape `[8, 4]`

### Output
- `K=3` predicted trajectories
- Each trajectory: 12 timesteps (3 seconds at 4Hz)
- Output shape: `[3, 12, 2]` + confidence scores `[3]`

### Target agents
- ✅ Pedestrian
- ✅ Cyclist
- ❌ Vehicles (out of scope)

---

## Finalized Technical Stack (Locked)

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| Deep Learning | PyTorch | 2.11.0 |
| Dataset SDK | nuscenes-devkit | 1.2.0 |
| Experiment Tracking | Weights & Biases | 0.25.1 |
| Visualization | Matplotlib | 3.10.8 |
| Environment (local) | Conda | via environment.yml |
| Environment (Kaggle) | pip | via requirements.txt |
| Version Control | Git + GitHub | latest |
| Compute (training) | Kaggle (GPU T4) | primary |
| Compute (local) | VSCode + Conda | Data Engineer only |

> **Rule:** stack decisions are fixed for sprint execution. No tech pivots after Day 1 unless explicitly approved.

---

## Non-Negotiable Data Decisions

These must be implemented exactly to avoid leakage and ensure reproducibility:

- Sampling rate: **4 Hz** (linear interpolation from nuScenes 2Hz)
- Input window: **8 timesteps (2s)**
- Prediction window: **12 timesteps (3s)**
- Normalization: **agent-centric** (last observed position = `(0,0)`)
- Features: append velocity `(dx, dy)` → input `[8,4]`
- Split: **scene-level only** (never frame-level)
- Ratio: **70/15/15** train/val/test

---

## Functional Requirements

### Must Have (qualification-critical)
- Temporal model (LSTM/GRU/Transformer)
- Social context handling
- Multimodal output (`K >= 3`)
- ADE and FDE reporting
- End-to-end inference pipeline (raw input → predictions)
- Working visualization demo on real nuScenes scenes

### Should Have
- Physically plausible trajectories
- Validation ADE no worse than 30% over train ADE
- <100ms single-agent CPU inference
- Reproducibility with fixed seed

### Out of Scope (Do Not Build)
- HD maps / map conditioning
- GAN / diffusion trajectories
- ONNX or real-time optimization
- Vehicle prediction
- Continual/online learning

---

## Model Plan (by Day)

### Baseline (Day 2 target)
- Encoder: LSTM (`hidden=128`, `layers=2`, `dropout=0.1`)
- Decoder: 3-mode decoding (`K=3`)
- Loss: Winner-Takes-All (WTA)
- Output: `[batch, 3, 12, 2]` + `[batch, 3]` confidences

### Improved (Day 4–5 target)
- Add social pooling:
  - neighbors within 2m radius
  - pool hidden states (max/sum)
  - concat pooled social vector before decode

### Final optional upgrade (Day 6 only if ahead)
Choose one:
- **Option A:** small Transformer encoder (4 heads, 2 layers, d_model=128)
- **Option B:** Temporal Conv encoder  
Decision by ML Lead, max 4-hour validation window.

---

## Team Ownership


### ML Lead
- Own architecture and checkpoints
- Baseline by Day 2
- WTA loss
- Social pooling integration
- Hyperparameter tuning
- Model card + architecture decisions

### Data Engineer
- nuScenes raw → clean DataLoader
- Track extraction for ped/cyclist
- Agent-centric normalization
- Velocity features
- Scene-level split + leakage checks
- Neighbor extraction for social pooling
- Data pipeline documentation

### ML Engineer #2 (Experiment Tracker)
- ADE/FDE implementations + unit tests
- W&B setup and logging
- Training loop scaffold
- Ablations (K=1 vs K=3, with/without velocity/social)
- Multimodal diversity checks
- Final results table

### Integration & Demo Engineer
- `inference.py` ownership (must run daily)
- Visualization pipeline
- Repo setup + `requirements.txt`
- Final demo on 5 curated scenes
- Reproduction-ready README
- Ensure end-to-end runtime <30s

---

## 7-Day Execution Schedule

### Day 1 — Foundation
- Data extraction pipeline
- Dataset class + tensor shape validation
- ADE/FDE + unit tests
- Repo scaffold runs clean

**Checkpoint:** anyone can clone and run `python train.py`

### Day 2 — Baseline running
- LSTM + K=3 + WTA training
- No leakage verified
- First 10-epoch run logged
- `inference.py` outputs 3 trajectories

**Checkpoint:** first val ADE/FDE numbers available

### Day 3 — E2E lock (hard deadline)
- Stable longer training
- LR trials
- Velocity features integrated
- K=1 vs K=3 ablation
- Visualization screenshots

**Checkpoint:** full fallback submission pipeline is ready

### Day 4 — Social context
- Social pooling integrated + tested
- Multi-agent inference supported
- Compare with vanilla baseline

### Day 5 — Tuning only
- No new major features
- Optimize ADE/FDE
- Failure case analysis
- Stress test inference edge cases

### Day 6 — Demo polish
- Model card, pipeline docs, final comparison table
- Demo polished on 5 scenes, under 30s

### Day 7 — Buffer + submit
- Fix breakages, freeze scope
- Final submission package
- Validate fallback checkpoint

---

## Metrics and Targets

### Reported Metrics
- **ADE**: average displacement error over trajectory
- **FDE**: final-point displacement error
- For multimodal (`K=3`): report **minADE** and **minFDE**

### Qualification Targets
| Metric | Unacceptable | Qualifies (Target) | Competitive |
|---|---:|---:|---:|
| ADE (3s) | > 2.0m | 1.0–1.5m | < 0.8m |
| FDE (3s) | > 4.0m | 2.0–3.0m | < 1.5m |

---

## Definition of Done

Project is done when all are true:

- 3 distinct plausible trajectory predictions per agent
- Val ADE < 1.5m
- Val FDE < 3.0m
- Social context implemented + documented
- Reproducible training with fixed seed
- Inference pipeline works end-to-end
- Demo runs on 5 real scenes in <30s
- README is fully reproducible for newcomers
- Code merged into runnable `main`
- Baseline vs Improved vs Final comparison table completed

---

## Repository Structure (Expected)

```text
TrajectoryPredict/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── metrics/
│   └── visualization/
├── tests/
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── logs/
├── train.py
├── inference.py
├── env.yml
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Kaggle — ML Training (ML Lead, ML #2, Integration Engineer)

1. Upload the dataset to your Kaggle notebook or use the nuScenes dataset from Kaggle's public datasets
2. In your Kaggle notebook, run the following in the first cell:
```python
  import torch
  print(torch.__version__)        # confirm version
  print(torch.cuda.is_available()) # must print True — if False, go to Settings → Accelerator → GPU T4
```

3. Install project dependencies:
```bash
  pip install -r requirements.txt
```

4. Do NOT install torch separately — Kaggle pre-installs it with CUDA. Reinstalling will break GPU support silently.

---

### Local / VSCode — Data Engineering & Demo (Data Engineer)

**Prerequisites:** Anaconda or Miniconda installed. CUDA-capable GPU recommended but not required for data work.

**First time setup:**
```bash
# 1. Clone the repo
git clone 
cd trajectorypredict

# 2. Check your CUDA version — look at top right of output
nvidia-smi

# 3. If CUDA version is 11.x, open environment.yml and change
#    pytorch-cuda=12.1  →  pytorch-cuda=11.8
#    before running the next command

# 4. Create and activate the environment
conda env create -f environment.yml

The environment.yml file pins the following:
- Python 3.10
- PyTorch 2.11.0 with CUDA support
- torchvision 0.26.0
- torchaudio 2.11.0
- All pip packages from requirements.txt (installed automatically)

conda activate trajpredict

# 5. Verify torch and CUDA are working
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**VSCode interpreter setup:**
- Open VSCode
- Bottom left corner → click the Python interpreter selector
- Choose the interpreter that shows `trajpredict` in the path
- If it doesn't appear, press `Ctrl+Shift+P` → `Python: Select Interpreter` → `Enter interpreter path` → paste the output of `conda run -n trajpredict which python`

**Subsequent runs:**
```bash
conda activate trajpredict
# then run whatever script you need
```

---

### Adding New Dependencies

If you need to add a package during the sprint:

1. Add it to `requirements.txt` with a pinned version
2. Kaggle users run `pip install <package>==version` in their notebook
3. Local users run `pip install <package>==version` inside the active conda env
4. Commit the updated `requirements.txt` — never commit environment changes without updating this file

---

### Known Issues

**nuScenes map rendering error on local setup:**
If you see a shapely or descartes related error when rendering nuScenes maps, run:
```bash
pip install shapely==2.0.7 descartes==1.1.0
```
Both are already in requirements.txt but some systems need a manual reinstall.

**wandb login:**
First time using wandb, run `wandb login` and paste your API key from wandb.ai/authorize. This is a one-time step per machine.

---

## GitHub Workflow & Push Guidelines

### Branching policy
- `main` is always runnable
- never push directly to `main`
- create feature branches:
  - `feat/<short-name>`
  - `fix/<short-name>`
  - `exp/<short-name>`

Examples:
- `feat/social-pooling`
- `fix/data-split-leakage`
- `exp/k3-vs-k1`

### Commit message format
Use clear, scoped commits:
- `feat(model): add k=3 decoder heads`
- `fix(data): enforce scene-level split`
- `test(metrics): add ADE/FDE unit tests`
- `docs(readme): update run instructions`

### Pull Request checklist (required)
Before requesting review:
- [ ] code runs locally
- [ ] no broken imports
- [ ] tests added/updated where relevant
- [ ] ADE/FDE impact noted (if model/data change)
- [ ] W&B run link added (if experiment)
- [ ] no secrets/API keys committed
- [ ] README/docs updated if behavior changed

### Merge rule
- minimum 1 reviewer approval
- all critical checks pass
- integration owner confirms end-to-end run if touching inference/train pipeline

### Hard rules
- no broken code on `main`
- no force-push to `main`
- no silent schema changes
- no untracked experiment assumptions (log in W&B or PR notes)

---

## Coding Guidelines

- Keep functions small and testable
- Prefer explicit shapes in code comments/docstrings
- Seed all random components for reproducibility
- Validate tensor shapes at data/model boundaries
- Add unit tests for metrics and data transforms
- Keep inference backward-compatible with latest stable checkpoint format

---



## Risk Register (Top 5)

1. nuScenes parsing complexity blocks Day 1  
2. Mode collapse (3 trajectories become identical)  
3. Data leakage via wrong split  
4. No GPU access / slow training  
5. Integration breakage near demo day  

Each risk must have active mitigation tracked during standup.

---


