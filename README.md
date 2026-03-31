# 🧭 Autonomous Trajectory & Intent Prediction (L4)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predicts where pedestrians will be — before they get there.**

[📊 Performance](#-performance) • [🚀 Setup](#-setup--installation) • [🎬 Demo](#-running-the-project) • [❓ Help](#-troubleshooting)

</div>

---

## 🎬 Demo Preview

<p align="center">
  <img src="assets/demo.gif" width="700"/>
</p>

> ⚠️ If the GIF doesn't load, generate it from `outputs/demo/` screenshots.

---

## 📌 Overview

Given **2 seconds of motion**, the model predicts **3 possible futures (3 seconds ahead)** — capturing real-world uncertainty in human behavior.

> 🚫 Not deterministic  
> ✅ Multimodal, socially-aware prediction

```
Input:   [(x₁,y₁), (x₂,y₂), (x₃,y₃), (x₄,y₄)]
↓
LSTM Model
+ Social Pooling Layer
↓
Output:  Mode 1 → future trajectory (most likely)
Mode 2 → alternative future
Mode 3 → alternative future
```

---

## 🎯 Why This Matters

Autonomous systems cannot rely on reaction alone.

They must:
- Anticipate pedestrian intent  
- Handle uncertainty  
- Avoid unsafe last-moment decisions  

This model shifts from **reaction → prediction**.

---

## ⚡ Key Highlights

- Multimodal trajectory prediction (K=3)
- Social interaction modeling (neighbor-aware)
- Rotation-invariant coordinate system
- Winner-Takes-All loss to prevent mode collapse
- End-to-end pipeline (data → model → evaluation)

---

## 🧠 Model Architecture

<table>
<tr>
<td width="50%">

### Pipeline Steps

**① Coordinate Normalization**  
Transforms coordinates into an agent-centric frame  
→ model learns motion, not location

**② Social Context Pooling**  
Encodes neighbors within **2m radius**  
→ captures interactions & avoids collisions

**③ Multimodal LSTM Backbone**  
- 2-layer LSTM (hidden = 128)  
- 3 decoder heads  
- Output: `[3, 6, 2]`

**④ Inverse Rotation**  
Transforms predictions back to global coordinates

</td>
<td width="50%">

### Architecture Summary

```
Input: [batch, 4, 4]
```
```
    LSTM Encoder (128)
           ↓
    Social Features (64)
           ↓
    Concatenate (192)
           ↓
    3× Decoder Heads
           ↓
```
```
Output: [batch, 3, 6, 2]
```

**Key Idea:** Model behavior, not position  
**Social Radius:** 2m  
**Loss:** Winner-Takes-All  

</td>
</tr>
</table>

---

## 📊 Performance

> Evaluated on **74 samples / 8 scenes (nuScenes mini)**

<div align="center">

| Model | Social | Multimodal | Mean minADE | Mean minFDE |
|------|--------|------------|------------|------------|
| **LSTM + Social Pooling** | ✅ | ✅ (K=3) | **0.2252 m** | **0.4016 m** |

</div>

🔥 **Result:** Easily beats hackathon threshold (ADE < 1.5m, FDE < 3.0m)

---

## 📂 Dataset

🔗 https://www.kaggle.com/datasets/tanaychordia/trajectorypredict  

Place files inside:
```
data/raw/
```

---

## ⚙️ Setup & Installation

> ⏱️ 15–30 minutes  
> 💻 Windows / macOS / Linux

---

### 🟢 Conda (Recommended)

```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict

conda create -n trajpredict python=3.11 -y
conda activate trajpredict

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r requirements.txt
```

---

### 🔵 Python venv

```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict

python3.11 -m venv trajpredict
source trajpredict/bin/activate   # macOS/Linux

# Windows:
# trajpredict\Scripts\activate.bat

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r requirements.txt
```

---

## 🧩 Add Model Checkpoint

```
outputs/checkpoints/best_model_final.pth
```

Create folder if needed:

```bash
mkdir -p outputs/checkpoints
```

---

## ✅ Verify

```bash
python -m src.inference --sample-idx 0
```

---

## 🚀 Run

### 🎬 Demo

```bash
python -m src.demo
```

### 📈 Evaluate

```bash
python -m src.evaluate_full_dataset
```

### 🧪 Custom Input

```bash
python -m src.test_custom_input
```

---

## 🗂️ Structure

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

## ❓ Troubleshooting

| Issue           | Fix                          |
| --------------- | ---------------------------- |
| conda not found | restart terminal             |
| env not active  | `conda activate trajpredict` |
| module errors   | run from root                |
| missing model   | place checkpoint             |
| torch missing   | reinstall                    |
| no GPU          | CPU works fine               |

---

<div align="center">

Built for ML Hackathon • LSTM + Social Pooling • PyTorch

</div>