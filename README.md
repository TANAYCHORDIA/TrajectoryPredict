# Autonomous Trajectory & Intent Prediction (L4)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Predicts where pedestrians will be before they get there.

[Performance](#performance) • [Setup](#setup--installation) • [Demo](#running-the-project) • [Help](#troubleshooting)

</div>

---

## Overview

Given 2 seconds of motion, the model predicts 3 possible futures (3 seconds ahead), capturing real-world uncertainty in human behavior.

- Not deterministic
- Multimodal, socially-aware prediction

```
Input:   [(x₁,y₁), (x₂,y₂), (x₃,y₃), (x₄,y₄)]
         ↓
    LSTM Model + Social Pooling
         ↓
Output:  3 possible future trajectories
```

---

## Why This Matters

Autonomous systems cannot rely on reaction alone.

They must:

- Anticipate pedestrian intent
- Handle uncertainty
- Avoid unsafe last-moment decisions

This model shifts from reaction to prediction.

---

## Key Highlights

- Multimodal trajectory prediction (K=3)
- Social interaction modeling
- Rotation-invariant coordinates
- Winner-Takes-All loss
- End-to-end pipeline

---

## Model Architecture

<table>
<tr>
<td width="50%" valign="top">

### Pipeline Steps

1. **Coordinate Normalization**
   Transforms coordinates into agent-centric frame

2. **Social Context Pooling**
   Encodes neighbors within 2m radius

3. **Multimodal LSTM**
   - 2-layer LSTM (128)
   - 3 decoder heads

4. **Inverse Rotation**

</td>
<td width="50%" valign="top">

### Architecture Flow

```
Input: [batch, 4, 4]

        ↓
LSTM Encoder (128)

        ↓
Social Features (64)

        ↓
Concatenate (192)

        ↓
3 Decoder Heads

        ↓
Output: [batch, 3, 6, 2]
```

</td>
</tr>
</table>

---

## Performance

Evaluated on 74 samples / 8 scenes (nuScenes mini)

| Model | Social | Multimodal | Mean minADE | Mean minFDE |
|---|---|---|---|---|
| LSTM + Social Pooling | Yes | Yes (K=3) | 0.2252 m | 0.4016 m |

**Result:** Beats threshold (ADE < 1.5m, FDE < 3.0m)

---

## Dataset

<https://www.kaggle.com/datasets/tanaychordia/trajectorypredict>

Place files inside:

```
data/raw/
```

---

## Setup & Installation

### Conda

```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict

conda create -n trajpredict python=3.11 -y
conda activate trajpredict

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r requirements.txt
```

### Python venv

```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict

python3.11 -m venv trajpredict
source trajpredict/bin/activate

# Windows:
# trajpredict\Scripts\activate.bat

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r requirements.txt
```

---

## Data Preprocessing (IMPORTANT)

Run this before inference:

```bash
python -m src.data.preprocess
```

This will generate:

```
data/processed/val_inputs.npy
```

> If you skip this step, inference will fail.

---

## Add Model Checkpoint

Place file:

```
outputs/checkpoints/best_model_final.pth
```

### Verify

```bash
python -m src.inference --sample-idx 0
```

---

## Running the Project

### Demo

```bash
python -m src.demo
```

### Evaluate

```bash
python -m src.evaluate_full_dataset
```

### Custom Input

```bash
python -m src.test_custom_input
```

---

## Project Structure

```
TrajectoryPredict/
├── data/
├── src/
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| File not found | Run preprocessing step |
| env not active | Activate environment |
| module errors | Run from project root |
| missing model | Add checkpoint |
| torch missing | Reinstall |

---

<div align="center">

Built for ML Hackathon using PyTorch

</div>
