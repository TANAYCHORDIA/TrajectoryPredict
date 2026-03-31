# 🧭 Autonomous Trajectory & Intent Prediction (L4)

## 📌 Project Overview
This project is a production-ready Behavioral AI pipeline designed to predict multimodal pedestrian and cyclist trajectories in complex urban environments. Built for autonomous vehicle systems, the model takes a 2-second history of an agent's coordinates and accurately predicts their most likely paths 3 seconds into the future.

Instead of relying on naive constant-velocity assumptions, this system successfully captures the uncertainty of human movement and avoids collisions by understanding the social dynamics of the surrounding environment.

---

## 🧠 Model Architecture
The system utilizes a **Multimodal LSTM + Social Pooling** architecture to generate context-aware predictions.

1. **Coordinate Normalization (Rotational Invariance):** Raw global `(x, y)` coordinates are translated to an agent-centric origin. The entire scene is then rotated so the primary agent's velocity vector aligns with the positive X-axis. This ensures the model learns movement patterns rather than memorizing global map locations.
2. **Social Context Pooling:** For every primary agent, the relative coordinates of all neighboring agents within a 2-metre radius are extracted and encoded. This prevents predicted collisions and allows the model to understand group dynamics.
3. **Multimodal LSTM Backbone:** A 2-layer LSTM encoder (hidden=128) processes the temporal sequence. Three independent decoder heads each output a predicted trajectory, producing a `[3, 6, 2]` tensor representing the **3 most likely predicted paths** over the next 6 timesteps (3 seconds). A Winner-Takes-All loss with 10-epoch warmup prevents mode collapse.
4. **Inverse Rotation API:** Predictions are mathematically un-rotated back into the global map frame for seamless integration into downstream AV simulators or visualization dashboards.

---

## 📊 Performance Metrics
The model was evaluated on 74 trajectory samples across 8 scenes from the nuScenes mini split.

| Architecture | Social Context | Multimodal | Mean minADE (m) | Mean minFDE (m) |
| :--- | :---: | :---: | :---: | :---: |
| LSTM Baseline | ❌ | ❌ | — | — |
| LSTM + Social Pooling (Final) | ✅ | ✅ K=3 | **0.2252** | **0.4016** |

*ADE and FDE are measured in metres. minADE/minFDE report the best of K=3 predicted modes.*  
*Trained and evaluated on nuScenes mini split (8 scenes). Full dataset performance expected in the 0.8–1.5m ADE range based on published baselines.*

---

## 📂 Dataset
Trained and evaluated on the **nuScenes Mini Split** — pedestrian and cyclist tracks only.

🔗 **[Download the Dataset on Kaggle](https://www.kaggle.com/datasets/tanaychordia/trajectorypredict)**

Place raw dataset files in `data/raw/` before running preprocessing scripts.

---

## ⚙️ Setup & Installation

> ⏱️ **Estimated setup time:** 15–30 minutes depending on your internet speed.  
> 💻 **Operating System:** These instructions are written for **Linux and macOS**. Windows users should use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) and follow the same steps inside the WSL terminal.

---

### Step 0 — Install Miniconda (Skip if already installed)

Miniconda is a lightweight tool that manages Python environments. It prevents package conflicts between different projects.

**Check if you already have it:**
```bash
conda --version
```

If this prints a version number (e.g. `conda 23.x.x`), skip to Step 1.

If you see `command not found`, install it:

**On Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
- Press `Enter` to scroll through the license
- Type `yes` when asked to accept the license
- Press `Enter` to confirm the install location
- Type `yes` when asked to initialize Miniconda

**On macOS (Apple Silicon — M1/M2/M3):**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**On macOS (Intel):**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

> ⚠️ **After installation, close your terminal completely and reopen it.** This is required for the `conda` command to become available. If `conda --version` still shows `command not found` after reopening, run `source ~/.bashrc` (Linux) or `source ~/.zshrc` (macOS) and try again.

---

### Step 1 — Clone the Repository

This downloads the project code to your machine.

```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

> ⚠️ **If you see `command not found: git`**, install it first:
> - Linux: `sudo apt install git`
> - macOS: `xcode-select --install`

After running `cd TrajectoryPredict`, your terminal should show you are inside the project folder. You can confirm with:
```bash
pwd
```
It should print something ending in `/TrajectoryPredict`.

---

### Step 2 — Create the Python Environment

This creates an isolated Python environment so that project dependencies do not interfere with anything else on your machine.

```bash
conda create -n trajpredict python=3.11 -y
```

This may take 1–2 minutes. You will see conda downloading packages.

Once it finishes, activate the environment:

```bash
conda activate trajpredict
```

> ✅ **How to confirm it worked:** Your terminal prompt should now start with `(trajpredict)`. For example:
> ```
> (trajpredict) your-name@your-machine:~/TrajectoryPredict$
> ```
> If you do not see `(trajpredict)`, the environment is not active. Run `conda activate trajpredict` again.

> ⚠️ **Every time you open a new terminal**, you must run `conda activate trajpredict` again before running any project commands. The environment does not stay active automatically.

---

### Step 3 — Install PyTorch

PyTorch is the deep learning framework used by this project. It must be installed separately because the correct version depends on your hardware.

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

> ⏱️ This download is approximately 500MB–1.5GB depending on your system. It may take 5–20 minutes on a normal connection. Do not close the terminal while it is running.

> ⚠️ **If the download keeps failing mid-way** (broken pipe or SSL error), your internet connection is dropping. Try:
> ```bash
> pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --retries 10 --timeout 120
> ```
> If it continues to fail, switch to a more stable connection (wired or mobile hotspot) and retry.

Once installed, verify it worked:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch version: 2.11.0
CUDA available: True
```

> ℹ️ If `CUDA available` prints `False`, PyTorch installed correctly but will use your CPU instead of GPU. The project will still run — it will just be slower for training. Inference and demo generation are fast on CPU too.

---

### Step 4 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ This takes 2–5 minutes. You will see a list of packages being downloaded and installed.

> ⚠️ **If you see an error like `No matching distribution found for X==Y.Z`**, your Python version may be wrong. Confirm you are on Python 3.11:
> ```bash
> python --version
> ```
> It should print `Python 3.11.x`. If it prints 3.9 or 3.10, delete the environment and recreate it:
> ```bash
> conda deactivate
> conda env remove -n trajpredict
> conda create -n trajpredict python=3.11 -y
> conda activate trajpredict
> ```
> Then repeat Steps 3 and 4.

---

### Step 5 — Add the Model Checkpoint

The pre-trained model weights are not stored in the repository (they are too large for GitHub). You need to place them manually.

1. Download `best_model_final.pth` from the release or shared drive link provided
2. Place it in the following folder inside the project:
```
TrajectoryPredict/
└── outputs/
	└── checkpoints/
		└── best_model_final.pth   ← file goes here
```

If the `outputs/checkpoints/` folder does not exist yet, create it:
```bash
mkdir -p outputs/checkpoints
```

Then move your downloaded file into it:
```bash
mv ~/Downloads/best_model_final.pth outputs/checkpoints/best_model_final.pth
```

Verify it is in the right place:
```bash
ls outputs/checkpoints/
```

You should see `best_model_final.pth` listed.

---

### Step 6 — Verify Full Setup

Run this to confirm everything is installed and the model loads correctly:

```bash
python -m src.inference --sample-idx 0
```

Expected output:
```
Sample 0 | minADE=X.XXXX | minFDE=X.XXXX
Saved prediction to: outputs/predictions/latest_prediction.npz
```

If you see this, your setup is complete.

> ⚠️ **Common errors and fixes:**
>
> `ModuleNotFoundError: No module named 'src'`  
> → You are not in the project root. Run `cd ~/TrajectoryPredict` and try again.
>
> `FileNotFoundError: Checkpoint not found`  
> → The model file is missing or in the wrong folder. Repeat Step 5.
>
> `ModuleNotFoundError: No module named 'torch'`  
> → Your conda environment is not active. Run `conda activate trajpredict` and try again.

---

## 🚀 How to Run

> ⚠️ **Before running any command**, make sure you are in the project root and your environment is active:
> ```bash
> cd ~/TrajectoryPredict
> conda activate trajpredict
> ```

---

### 1. Generate Visual Demo

Runs inference on 6 curated scenes and renders autonomous vehicle dashboards showing the past trajectory, ground truth, and 3 multimodal predicted futures.

```bash
python -m src.demo
```

Output images are saved to `outputs/demo/`. Open any `.png` file in that folder to view the dashboard.

> ⏱️ Expected runtime: under 30 seconds for all 6 scenes.

---

### 2. Evaluate Full Dataset Metrics

Runs inference across the entire validation split and reports mean ADE and FDE with social features enabled.

```bash
python -m src.evaluate_full_dataset
```

Expected output:
```
True Mean ADE : 0.2252 meters
True Mean FDE : 0.4016 meters
✅ QUALIFIED: Model passes the hackathon constraints.
```

---

### 3. Test Custom Inputs

Tests the production API with a custom history of raw `(x, y)` coordinates and renders a live visualization on screen.

```bash
python -m src.test_custom_input
```

> ℹ️ A plot window will open on your screen. Close it to exit the script.  
> If running on a remote server without a display, this script will fail with a display error — run it on your local machine instead.

---

## 🗂️ Repository Structure

```
TrajectoryPredict/
├── data/
│   ├── raw/               ← place nuScenes JSON files here
│   └── processed/         ← generated by preprocessing scripts
├── src/
│   ├── data/              ← data pipeline (parsing, preprocessing, dataset)
│   ├── model.py           ← LSTM + Social Pooling architecture
│   ├── metrics.py         ← ADE and FDE metric functions
│   ├── inference.py       ← single-sample inference
│   ├── evaluate_full_dataset.py ← full val set evaluation
│   ├── demo.py            ← dashboard visualization
│   └── test_custom_input.py ← custom coordinate API test
├── outputs/
│   ├── checkpoints/       ← model weights (.pth files)
│   ├── predictions/       ← saved inference outputs (.npz files)
│   └── demo/              ← generated dashboard images (.png files)
├── tests/                 ← unit tests
├── environment.yml        ← local conda environment (Data Engineer)
├── requirements.txt       ← pip dependencies
└── README.md
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `conda: command not found` | Close and reopen terminal after installing Miniconda |
| `(trajpredict)` not showing in prompt | Run `conda activate trajpredict` |
| `No module named 'src'` | Run `cd ~/TrajectoryPredict` first |
| `FileNotFoundError: best_model_final.pth` | Complete Step 5 — place checkpoint in `outputs/checkpoints/` |
| `No module named 'torch'` | Run `conda activate trajpredict` then reinstall torch |
| PyTorch download keeps failing | Switch to mobile hotspot and retry with `--retries 10 --timeout 120` |
| Plot window does not open | Run on local machine, not a remote server |
| `CUDA available: False` | No GPU detected — project still runs on CPU, just slower |