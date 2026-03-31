````markdown
# 🧭 Autonomous Trajectory & Intent Prediction (L4)

## 📌 Project Overview
A production-ready Behavioral AI pipeline that predicts multimodal pedestrian and cyclist trajectories in urban environments. Given a 2-second history of an agent's coordinates, the model predicts their 3 most likely paths over the next 3 seconds.

Instead of naive constant-velocity assumptions, the system captures human movement uncertainty by understanding the social dynamics of the surrounding environment.

---

## 🧠 Model Architecture

**Multimodal LSTM + Social Pooling**

1. **Coordinate Normalization** — Translates and rotates each scene to an agent-centric frame, making the model position and direction invariant.
2. **Social Context Pooling** — Encodes neighboring agents within a 2m radius so the model understands group dynamics and avoids predicted collisions.
3. **Multimodal LSTM Backbone** — A 2-layer LSTM (hidden=128) with 3 independent decoder heads outputs `[3, 6, 2]` — the 3 most likely trajectories over 6 future timesteps. Winner-Takes-All loss with 10-epoch warmup prevents mode collapse.
4. **Inverse Rotation** — Predictions are rotated back to global map coordinates for downstream use.

---

## 📊 Performance

Evaluated on 74 trajectory samples across 8 scenes from the nuScenes mini split.

| Architecture | Social | Multimodal | Mean minADE | Mean minFDE |
| :--- | :---: | :---: | :---: | :---: |
| LSTM + Social Pooling (Final) | ✅ | ✅ K=3 | **0.2252m** | **0.4016m** |

*Evaluated on nuScenes mini split. Full dataset performance expected in the 0.8–1.5m ADE range.*

---

## 📂 Dataset

🔗 **[Download from Kaggle](https://www.kaggle.com/datasets/tanaychordia/trajectorypredict)**

Place the downloaded files inside `data/raw/` before running any preprocessing scripts.

---

## ⚙️ Setup & Installation

> ⏱️ Estimated time: 15–30 minutes  
> ✅ Works on Windows, macOS, and Linux

---

### Option A — Using Conda (Recommended)

**1. Install Miniconda** if you don't have it: https://docs.conda.io/en/latest/miniconda.html  
After installing, close and reopen your terminal.

**2. Clone the repo**
```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

**3. Create and activate environment**
```bash
conda create -n trajpredict python=3.11 -y
conda activate trajpredict
```

> Your terminal prompt should now show `(trajpredict)`. If not, run `conda activate trajpredict` again.

**4. Install PyTorch**
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

> ⏱️ Large download (~1GB). Do not close the terminal.

**5. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

---

### Option B — Using Python venv (No Conda)

**1. Install Python 3.11** from https://www.python.org/downloads/  
> ⚠️ Windows users: check **"Add Python to PATH"** during installation.

**2. Clone the repo**
```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

**3. Create and activate environment**

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

> ⚠️ Windows PowerShell users: if activation fails with a policy error, run `Set-ExecutionPolicy RemoteSigned` in PowerShell as Administrator, then retry.

**4. Install PyTorch**
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

**5. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

---

### Final Step — Add the Model Checkpoint

Place `best_model_final.pth` inside `outputs/checkpoints/`.

If the folder doesn't exist, create it:

Windows:
```bash
mkdir outputs\checkpoints
```

macOS / Linux:
```bash
mkdir -p outputs/checkpoints
```

Then copy the file into that folder manually using your file explorer, or via terminal.

**Verify everything works:**
```bash
python -m src.inference --sample-idx 0
```

Expected output:
```
Sample 0 | minADE=X.XXXX | minFDE=X.XXXX
Saved prediction to: outputs/predictions/latest_prediction.npz
```

---

## 🚀 Running the Project

> ⚠️ Always activate your environment before running commands:  
> Conda: `conda activate trajpredict`  
> venv Windows: `trajpredict\Scripts\activate.bat`  
> venv macOS/Linux: `source trajpredict/bin/activate`

**Generate Visual Demo**
```bash
python -m src.demo
```
Renders 6 scene dashboards showing past trajectory, ground truth, and 3 predicted futures.  
Output saved to `outputs/demo/`.

**Evaluate Full Dataset**
```bash
python -m src.evaluate_full_dataset
```
Reports Mean ADE and FDE across the full validation split.

**Test Custom Coordinates**
```bash
python -m src.test_custom_input
```
Pass any raw `(x, y)` history and visualise predictions live on screen.

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `conda: command not found` | Close and reopen terminal after installing Miniconda |
| `(trajpredict)` not in prompt | Run `conda activate trajpredict` |
| PowerShell activation error | Run `Set-ExecutionPolicy RemoteSigned` as Administrator |
| `No module named 'src'` | Navigate to project root: `cd TrajectoryPredict` |
| `FileNotFoundError: best_model_final.pth` | Place the checkpoint in `outputs/checkpoints/` |
| `No module named 'torch'` | Activate your environment and reinstall torch |
| `CUDA available: False` | No GPU detected — project still runs correctly on CPU |
| PyTorch download failing | Retry with `--retries 10 --timeout 120` or switch to mobile hotspot |
````