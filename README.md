# TrajectoryPredict

TrajectoryPredict is a 7-day hackathon project focused on **predicting future trajectories of pedestrians and cyclists** in urban scenes using the **nuScenes** dataset.

Our objective is simple: ship a **working, reproducible, demo-ready** system that achieves qualifying metrics for Round 1.

---

## Executive Summary

Autonomous systems need to predict not only where people are now, but where they are likely to be in the next few seconds.  
Given 2 seconds of observed trajectory, we predict 3 seconds into the future with **multimodal outputs** (3 possible futures), because human behavior is uncertain.

This repo is the team’s **single source of truth** for:
- technical decisions
- daily execution and ownership
- coding standards
- GitHub workflow

---

## Problem Definition

### Input
- Observed trajectory: `[(x1,y1), ... , (x4,y4)]`
- 2 seconds at native 2Hz = 4 timesteps
- Features per timestep: `(x, y, dx, dy)` → shape `[4, 4]`

### Output
- `K=3` predicted trajectories
- Each trajectory: 6 timesteps (3 seconds at native 2Hz)
- Output shape: `[3, 6, 2]` + confidence scores `[3]`

### Target agents
- ✅ Pedestrian
- ✅ Cyclist
- ❌ Vehicles (out of scope)

---

## Finalized Technical Stack (Locked)

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| DL Framework | PyTorch | 2.11.0 |
| Data SDK | nuscenes-devkit | 1.2.0 |
| Exp Tracking | Weights & Biases | 0.25.1 |
| Visualization | Matplotlib | 3.10.8 |
| Environment (local) | Conda | via environment.yml |
| Environment (Kaggle) | pip | via requirements.txt |
| Version Control | Git + GitHub | latest |
| Compute (training) | Kaggle (GPU T4) | primary |
| Compute (local) | VSCode + Conda | Data Engineer only |

> **Rule:** Stack decisions are fixed for sprint execution. No tech pivots after Day 1 unless explicitly approved.

---

## Non-Negotiable Data Decisions

These must be implemented exactly to avoid leakage, prevent mode collapse, and ensure the model learns actual kinematics:

- **Sampling rate:** **2 Hz** (Native nuScenes rate. Do NOT interpolate to 4Hz. It creates fake data and exacerbates vanishing gradients).
- **Input window:** **4 timesteps (2s)**
- **Prediction window:** **6 timesteps (3s)**
- **Normalization:** **Translation AND Rotation** (Translate last observed position to `(0,0)`, AND rotate so the agent's last known heading points along the positive X-axis. This makes the model rotation-invariant and prevents the model wasting capacity learning basic geometry).
- **Features:** Append velocity `(dx, dy)` → input `[4,4]`
- **Split:** **Scene-level only** (never frame-level)
- **Ratio:** **70/15/15** train/val/test

---

## Functional Requirements

### Must Have (Qualification-Critical)
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

## Team Ownership & Explicit Tasks

### ML Lead
**Goal:** Own the architecture, loss functions, and final checkpoints.
- **Day 1-2:** Build the LSTM encoder/decoder baseline accepting `[batch, 4, 4]` and outputting `[batch, 3, 6, 2]`. 
- **CRITICAL CORRECTION (WTA Loss):** Do *not* implement a naive Winner-Takes-All loss, or two of your heads will die by Epoch 2. Implement a **warmup phase**: for the first 10 epochs, backpropagate loss to ALL 3 heads equally to keep them alive, then transition to strict WTA. 5 epochs is not enough warmup — heads can still die on smaller datasets.
- **Day 4:** Integrate social pooling (concat max/sum pooled neighbor hidden states).
- **Day 5:** Hyperparameter tuning (dropout, LR scheduling).

### Data Engineer
**Goal:** Own the data pipeline from raw nuScenes to PyTorch DataLoader. 
- **Day 1 (Extraction):** Run the dedicated extraction script to pull contiguous pedestrian/cyclist tracks into a clean CSV. 
- **Day 1-2 (Normalization):** Implement the Dataset class. You MUST implement both translation to `(0,0)` AND rotation to the Y-axis. 
- **Day 1 (evening):** Append `(dx, dy)` velocity features. This must be done on Day 1 — the ML Lead's baseline expects input shape [batch, 4, 4] from Day 2 onwards. Velocity at frame 0 is always (0, 0), never NaN.
- **Day 3:** Enforce the 70/15/15 scene-level split and mathematically prove to the team there is no data leakage.
- **Day 4:** Build the neighbor extraction logic (agents within 2m) for the ML Lead's social pooling.

### ML Engineer #2 (Experiment Tracker)
**Goal:** Own the metrics, experiment tracking, and scientific rigor.
- **Day 1:** Write the ADE and FDE functions. Write unit tests passing dummy tensors through them to prove they calculate distance correctly. Set up W&B.
- **Day 3:** Monitor for mode collapse. Plot the K=3 outputs on a graph and visually confirm the model is actually predicting diverse futures. If they are identical, flag the ML Lead immediately. 
- **Day 4-6:** Run ablations (with/without velocity, with/without social) and build the final results comparison table.

### Integration & Demo Engineer
**Goal:** Own the end-to-end pipeline and what the judges actually see.
- **Day 1:** Set up repo structure, `requirements.txt`, and verify the empty train loop runs on Kaggle and local.
- **Day 2-7:** Own `inference.py`. Run it daily. If the ML Lead breaks inference, force a fix.
- **Day 3 (Visualizer):** Build the plotting tool. *Warning:* Because the Data Engineer rotated the inputs, you must reverse the rotation during inference so the predictions map correctly back onto the nuScenes visualizer.
- **Day 6:** Build the final 30-second demo on 5 curated scenes. Make it look professional.

---

## 7-Day Execution Schedule

- **Day 1 — Foundation:** Data extraction pipeline, Dataset class + tensor shape validation, ADE/FDE unit tests. Repo scaffold runs clean. *(Checkpoint: anyone can clone and run `train.py`)*
- **Day 2 — Baseline:** LSTM + K=3 + WTA (with warmup) training. First validation ADE/FDE numbers available.
- **Day 3 — E2E Lock (HARD DEADLINE):** Stable training, velocity integrated, K=1 vs K=3 ablation, visual check. *(Checkpoint: Fallback submission ready)*
- **Day 4 — Social Context:** Social pooling integrated + tested. Multi-agent inference supported.
- **Day 5 — Tuning Only:** No new major features. Optimize ADE/FDE, failure case analysis.
- **Day 6 — Demo Polish:** Model card, pipeline docs, final comparison table, 30s demo polished on 5 scenes.
- **Day 7 — Buffer + Submit:** Fix breakages, freeze scope, prepare final submission package.

---

## Metrics and Targets

- **ADE**: Average Euclidean distance across the 3-second horizon.
- **FDE**: Euclidean distance of the final 3-second endpoint.
- For multimodal (`K=3`), we report **minADE** and **minFDE**.

**Qualification Target:**
| Metric | Unacceptable | Qualifies (Target) | Competitive |
|---|---:|---:|---:|
| ADE (3s) | > 2.0m | 1.0–1.5m | < 0.8m |
| FDE (3s) | > 4.0m | 2.0–3.0m | < 1.5m |

---

## Definition of Done

Project is done when all are true:
1. 3 distinct plausible trajectory predictions per agent
2. Val ADE < 1.5m, Val FDE < 3.0m
3. Social context implemented + documented
4. Reproducible training with fixed seed
5. Inference pipeline works end-to-end
6. Demo runs on 5 real scenes in <30s
7. README is fully reproducible for newcomers
8. Code merged into runnable `main`
9. Baseline vs Improved vs Final comparison table completed

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
├── environment.yml
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


