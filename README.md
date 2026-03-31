````markdown
# 🧭 Autonomous Trajectory & Intent Prediction (L4)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-ready Behavioral AI pipeline for autonomous vehicle systems.**  
Predicts where pedestrians and cyclists will be — before they get there.

[📊 View Metrics](#-performance) • [🚀 Quick Start](#-setup--installation) • [🎮 Run Demo](#-running-the-project) • [❓ Help](#-troubleshooting)

</div>

---

## 📌 Overview

Given just **2 seconds** of an agent's past movement, this system predicts their **3 most likely paths** over the next **3 seconds** — accounting for uncertainty in human behavior and the influence of nearby people.

```
Input:   [(x₁,y₁), (x₂,y₂), (x₃,y₃), (x₄,y₄)]   ← 2 seconds of history
                              ↓
                         LSTM Model
                    + Social Pooling Layer
                              ↓
Output:  Mode 1 → [(x̂₁,ŷ₁) ... (x̂₆,ŷ₆)]          ← most likely path
         Mode 2 → [(x̂₁,ŷ₁) ... (x̂₆,ŷ₆)]          ← alternative path
         Mode 3 → [(x̂₁,ŷ₁) ... (x̂₆,ŷ₆)]          ← alternative path
```

Unlike naive constant-velocity extrapolation, this model understands **social dynamics** — it knows that people slow down, turn, and respond to those around them.

---

## 🧠 Model Architecture

<table>
<tr>
<td width="50%">

### Pipeline Steps

**① Coordinate Normalization**  
Raw global `(x, y)` coordinates are translated to an agent-centric origin and rotated so the heading aligns with the positive X-axis. The model learns *movement patterns*, not map locations.

**② Social Context Pooling**  
All neighboring agents within a **2-metre radius** are identified at each timestep. Their relative positions are encoded and fed into the model, enabling it to understand group dynamics and avoid predicted collisions.

**③ Multimodal LSTM Backbone**  
A 2-layer LSTM encoder (`hidden=128`) processes the 4-timestep history. Three independent decoder heads each produce one predicted trajectory, outputting a `[3, 6, 2]` tensor.

**④ Inverse Rotation**  
Predictions are mathematically rotated back into global map coordinates, ready for downstream AV simulators or visualization tools.

</td>
<td width="50%">

### Architecture Summary

```
Input: [batch, 4, 4]
  └─ (x, y, dx, dy) per timestep

        ┌─────────────────┐
        │   LSTM Encoder  │
        │  hidden=128     │
        │  layers=2       │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  Social MLP     │
        │  neighbors→64d  │
        └────────┬────────┘
                 │ concat [128+64=192]
        ┌────────▼────────┐
        │ 3× Decoder Head │
        │ 192→128→12      │
        └────────┬────────┘
                 │ reshape
Output: [batch, 3, 6, 2]
  └─ 3 modes × 6 timesteps × (x,y)
```

**Loss:** Winner-Takes-All with 10-epoch warmup  
**Optimizer:** Adam  
**Social radius:** 2.0 metres  

</td>
</tr>
</table>

---

## 📊 Performance

> Evaluated on **74 trajectory samples** across **8 scenes** from the nuScenes mini split.

<div align="center">

| Architecture | Social Context | Multimodal Output | Mean minADE ↓ | Mean minFDE ↓ |
| :--- | :---: | :---: | :---: | :---: |
| LSTM + Social Pooling | ✅ | ✅ K=3 | **0.2252 m** | **0.4016 m** |

</div>

> 💡 **What these numbers mean:**  
> - **minADE** — average distance between the best predicted path and ground truth, across all timesteps  
> - **minFDE** — distance between the predicted final position and where the agent actually ended up  
> - Lower is better. Hackathon qualification threshold: ADE < 1.5m, FDE < 3.0m  

> ⚠️ *Trained on nuScenes mini split (8 scenes). Full dataset performance expected in the 0.8–1.5m ADE range based on published baselines.*

---

## 📂 Dataset

This project uses the **nuScenes Mini Split** — pedestrian and cyclist tracks only.

<div align="center">

🔗 **[Download Dataset from Kaggle](https://www.kaggle.com/datasets/tanaychordia/trajectorypredict)**

</div>

After downloading, place the raw JSON files inside:
```
TrajectoryPredict/
└── data/
    └── raw/        ← files go here
```

---

## ⚙️ Setup & Installation

> ⏱️ **Estimated time:** 15–30 minutes  
> ✅ **Supported OS:** Windows, macOS, Linux

---

### Option A — Conda (Recommended)

<details>
<summary><b>Click to expand Conda setup instructions</b></summary>

<br>

**1. Install Miniconda**  
Download from https://docs.conda.io/en/latest/miniconda.html and run the installer.  
After installing, **close and reopen your terminal**.

**2. Clone the repository**
```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

> If `git` is not found:  
> Windows → download from https://git-scm.com/download/win  
> macOS → run `xcode-select --install`  
> Linux → run `sudo apt install git`

**3. Create and activate the environment**
```bash
conda create -n trajpredict python=3.11 -y
conda activate trajpredict
```

Your terminal prompt should now show `(trajpredict)`.  
If it doesn't, run `conda activate trajpredict` again.

**4. Install PyTorch**
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```
> ⏱️ Large download (~1GB). Do not close the terminal.

**5. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

</details>

---

### Option B — Python venv (No Conda)

<details>
<summary><b>Click to expand venv setup instructions</b></summary>

<br>

**1. Install Python 3.11**  
Download from https://www.python.org/downloads/  
> ⚠️ Windows users: check **"Add Python to PATH"** during installation.

**2. Clone the repository**
```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

**3. Create and activate the environment**

Windows (Command Prompt):
```bash
python -m venv trajpredict
trajpredict\Scripts\activate.bat
```

macOS / Linux:
```bash
python3.11 -m venv trajpredict
source trajpredict/bin/activate
```

> ⚠️ Windows PowerShell: if activation fails, run `Set-ExecutionPolicy RemoteSigned` in PowerShell as Administrator, then retry.

**4. Install PyTorch**
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

**5. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

</details>

---

### Final Step — Add the Model Checkpoint

Place `best_model_final.pth` inside the `outputs/checkpoints/` folder.

Create the folder if it doesn't exist:

```bash
# Windows
mkdir outputs\checkpoints

# macOS / Linux
mkdir -p outputs/checkpoints
```

Then copy `best_model_final.pth` into that folder using your file explorer or terminal.

**Verify everything works:**
```bash
python -m src.inference --sample-idx 0
```

✅ Expected output:
```
Sample 0 | minADE=X.XXXX | minFDE=X.XXXX
Saved prediction to: outputs/predictions/latest_prediction.npz
```

---

## 🚀 Running the Project

> ⚠️ Always activate your environment before running commands:
> ```bash
> # Conda
> conda activate trajpredict
>
> # venv — Windows
> trajpredict\Scripts\activate.bat
>
> # venv — macOS/Linux
> source trajpredict/bin/activate
> ```

---

### 🎬 Generate Visual Demo
```bash
python -m src.demo
```
Renders **6 scene dashboards** showing past trajectory, ground truth, and 3 multimodal predicted futures.  
Output saved to `outputs/demo/` as `.png` files.  
> ⏱️ Runs in under 30 seconds.

---

### 📈 Evaluate Full Dataset
```bash
python -m src.evaluate_full_dataset
```
Runs inference across the entire validation split with social features and reports ADE and FDE.

Expected output:
```
True Mean ADE : 0.2252 meters
True Mean FDE : 0.4016 meters
✅ QUALIFIED: Model passes the hackathon constraints.
```

---

### 🧪 Test Custom Coordinates
```bash
python -m src.test_custom_input
```
Pass any raw `(x, y)` history into the production API and visualise 3 predicted futures live on screen.  
> ℹ️ A plot window will open. Close it to exit.

---

## 🗂️ Repository Structure

```
TrajectoryPredict/
├── data/
│   ├── raw/                         ← place nuScenes JSON files here
│   └── processed/                   ← auto-generated by preprocessing
├── src/
│   ├── data/                        ← parsing, preprocessing, dataset classes
│   ├── model.py                     ← LSTM + Social Pooling architecture
│   ├── metrics.py                   ← ADE and FDE implementations
│   ├── inference.py                 ← single-sample inference + EndToEndPredictor
│   ├── evaluate_full_dataset.py     ← full validation set evaluation
│   ├── demo.py                      ← dashboard visualization generator
│   └── test_custom_input.py         ← custom coordinate API test
├── outputs/
│   ├── checkpoints/                 ← model weights (best_model_final.pth)
│   ├── predictions/                 ← saved .npz inference outputs
│   └── demo/                        ← generated dashboard .png images
├── tests/                           ← unit tests
├── environment.yml                  ← conda environment definition
├── requirements.txt                 ← pip dependencies
└── README.md
```

---

## ❓ Troubleshooting

| Problem | Fix |
|:---|:---|
| `conda: command not found` | Close and reopen terminal after installing Miniconda |
| `(trajpredict)` not in prompt | Run `conda activate trajpredict` |
| PowerShell activation error | Run `Set-ExecutionPolicy RemoteSigned` as Administrator |
| `No module named 'src'` | Navigate to project root: `cd TrajectoryPredict` |
| `FileNotFoundError: best_model_final.pth` | Place checkpoint file in `outputs/checkpoints/` |
| `No module named 'torch'` | Activate environment and reinstall PyTorch |
| `CUDA available: False` | No GPU detected — project runs correctly on CPU |
| PyTorch download keeps failing | Retry with `--retries 10 --timeout 120` or use mobile hotspot |
| Plot window does not open | Run on local machine, not a remote server |

---

<div align="center">

Built for the 7-Day ML Hackathon Sprint  
**nuScenes Mini Split · PyTorch 2.11.0 · LSTM + Social Pooling**

</div>
````