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
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | 2.x |
| Dataset SDK | nuscenes-devkit | latest |
| Experiment Tracking | Weights & Biases | free tier |
| Visualization | Matplotlib | 3.x |
| Environment | conda / venv | team choice |
| Version Control | Git + GitHub | latest |
| Packaging | `env.yml` (+ optional `requirements.txt`) | required |
| Compute fallback | Kaggle / Colab Pro | as needed |

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
├── requirments.txt
└── README.md
```

---

## Setup and Usage

### 1) Clone repository
```bash
git clone <repo-url>
cd TrajectoryPredict
```

### 2) Create Conda environment (recommended)
```bash
conda env create -f env.yml
conda activate trajpredict
python -c "import torch; print(torch.__version__)"
```

For dependency notes and quick checks, see `requirments.txt`.

### 3) Optional venv setup (if not using Conda)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Train baseline
```bash
python train.py
```

### 5) Run inference
```bash
python inference.py
```

### 6) Run tests
```bash
pytest -q
```

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


