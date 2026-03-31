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
> 💻 **These instructions support Windows, macOS, and Linux natively. No WSL required.**

---

### Step 0 — Open Your Terminal

**Windows:**
- Press `Windows key + R`, type `cmd`, press Enter
- OR search for **Command Prompt** or **PowerShell** in the Start menu
- Right-click and select **Run as Administrator** for best results

**macOS:**
- Press `Cmd + Space`, type `Terminal`, press Enter

**Linux:**
- Press `Ctrl + Alt + T`

> ⚠️ **Keep this terminal open for all following steps.** Do not close it until setup is complete.

---

### Step 1 — Install Miniconda (Skip if already installed)

Miniconda manages Python environments and prevents package conflicts between projects.

**Check if you already have it:**
```bash
conda --version
```

If this prints a version number (e.g. `conda 23.x.x`), skip to Step 2.

If you see `command not found` or `'conda' is not recognized`, install it:

**Windows:**
1. Download the installer: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. Double-click the downloaded `.exe` file
3. Click Next → I Agree → Next → Next
4. ✅ **Check the box that says "Add Miniconda3 to my PATH environment variable"** — this is unchecked by default but you need it
5. Click Install → Finish
6. **Close Command Prompt completely and reopen it** — this is required

**macOS (Apple Silicon — M1/M2/M3):**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**macOS (Intel):**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

For macOS and Linux installers:
- Press `Enter` to scroll through the license
- Type `yes` to accept
- Press `Enter` to confirm the install location
- Type `yes` when asked to initialize Miniconda

> ⚠️ **After installation on any OS, close your terminal completely and reopen it before continuing.** If `conda --version` still shows an error after reopening:
> - Windows: Search for **Anaconda Prompt** in the Start menu and use that instead of Command Prompt
> - macOS/Linux: Run `source ~/.bashrc` or `source ~/.zshrc` and try again

---

### Step 2 — Install Git (Skip if already installed)

Git is used to download the project code.

**Check if you already have it:**
```bash
git --version
```

If this prints a version number, skip to Step 3.

**Windows:**
1. Download the installer: https://git-scm.com/download/win
2. Run the `.exe` file and click Next through all screens — default options are fine
3. Close and reopen your terminal after installation

**macOS:**
```bash
xcode-select --install
```
A popup will appear — click Install and wait for it to finish.

**Linux:**
```bash
sudo apt install git
```

---

### Step 3 — Clone the Repository

This downloads the project to your machine.

**Windows (Command Prompt or PowerShell):**
```bash
cd %USERPROFILE%\Desktop
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

**macOS and Linux:**
```bash
cd ~/Desktop
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

> ✅ **Confirm you are in the right folder:**
> ```bash
> # Windows
> echo %CD%
> # macOS/Linux
> pwd
> ```
> The output should end with `TrajectoryPredict`.

---

### Step 4 — Create the Python Environment

This creates an isolated Python environment for the project.
```bash
conda create -n trajpredict python=3.11 -y
```

This may take 1–2 minutes. Once complete, activate it:

**Windows:**
```bash
conda activate trajpredict
```

**macOS and Linux:**
```bash
conda activate trajpredict
```

> ✅ **How to confirm it worked:** Your terminal prompt should now start with `(trajpredict)`:
> ```
> (trajpredict) C:\Users\YourName\Desktop\TrajectoryPredict>
> ```
> If you do not see `(trajpredict)`, the environment is not active. Run `conda activate trajpredict` again.

> ⚠️ **Every time you open a new terminal**, you must run `conda activate trajpredict` again before running any project commands.

> ⚠️ **Windows only — if `conda activate` gives an error about execution policy:**
> Open PowerShell as Administrator and run:
> ```bash
> Set-ExecutionPolicy RemoteSigned
> ```
> Type `Y` and press Enter. Then retry `conda activate trajpredict`.

---

### Step 5 — Install PyTorch

PyTorch is the deep learning framework used by this project.
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

> ⏱️ This download is approximately 500MB–1.5GB. It may take 5–20 minutes. Do not close the terminal.

> ⚠️ **If the download keeps failing** (broken pipe or connection error):
> ```bash
> pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --retries 10 --timeout 120
> ```
> If it still fails, switch to a more stable connection (wired or mobile hotspot) and retry.

Once installed, verify it:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch version: 2.11.0
CUDA available: True
```

> ℹ️ If `CUDA available` prints `False`, PyTorch installed correctly but will use your CPU. The project will still run — inference and demo generation work fine on CPU.

---

### Step 6 — Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

> ⏱️ This takes 2–5 minutes.

> ⚠️ **If you see `No matching distribution found for X==Y.Z`**, confirm your Python version:
> ```bash
> python --version
> ```
> It must print `Python 3.11.x`. If it shows anything else, recreate the environment:
> ```bash
> conda deactivate
> conda env remove -n trajpredict
> conda create -n trajpredict python=3.11 -y
> conda activate trajpredict
> ```
> Then repeat Steps 5 and 6.

---

### Step 7 — Add the Model Checkpoint

The pre-trained model weights must be placed manually as they are too large for GitHub.

**Create the folder if it does not exist:**

Windows:
```bash
mkdir outputs\checkpoints
```

macOS/Linux:
```bash
mkdir -p outputs/checkpoints
```

**Move the downloaded checkpoint into it:**

Windows (if downloaded to Downloads folder):
```bash
move %USERPROFILE%\Downloads\best_model_final.pth outputs\checkpoints\best_model_final.pth
```

macOS/Linux:
```bash
mv ~/Downloads/best_model_final.pth outputs/checkpoints/best_model_final.pth
```

**Verify it is in the right place:**

Windows:
```bash
dir outputs\checkpoints\
```

macOS/Linux:
```bash
ls outputs/checkpoints/
```

You should see `best_model_final.pth` listed.

---

### Step 8 — Verify Full Setup

Run this to confirm everything works end to end:
```bash
python -m src.inference --sample-idx 0
```

Expected output:
```
Sample 0 | minADE=X.XXXX | minFDE=X.XXXX
Saved prediction to: outputs/predictions/latest_prediction.npz
```

If you see this, your setup is complete.

---

## 🚀 How to Run

> ⚠️ **Before running any command**, confirm you are in the project folder and the environment is active:
>
> Windows:
> ```bash
> cd %USERPROFILE%\Desktop\TrajectoryPredict
> conda activate trajpredict
> ```
>
> macOS/Linux:
> ```bash
> cd ~/Desktop/TrajectoryPredict
> conda activate trajpredict
> ```

---

### 1. Generate Visual Demo

Runs inference on 6 curated scenes and renders dashboards showing past trajectory, ground truth, and 3 multimodal predictions.
```bash
python -m src.demo
```

Output images are saved to `outputs/demo/`. Open any `.png` file to view the dashboard.

> ⏱️ Expected runtime: under 30 seconds for all 6 scenes.

---

### 2. Evaluate Full Dataset Metrics

Runs inference across the entire validation split and reports mean ADE and FDE.
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

Tests the production API with arbitrary raw `(x, y)` coordinates and renders a live visualization.
```bash
python -m src.test_custom_input
```

> ℹ️ A plot window will open on your screen. Close it to exit the script.

---

## 🗂️ Repository Structure
```
TrajectoryPredict/
├── data/
│   ├── raw/                        ← place nuScenes JSON files here
│   └── processed/                  ← generated by preprocessing scripts
├── src/
│   ├── data/                       ← data pipeline
│   ├── model.py                    ← LSTM + Social Pooling architecture
│   ├── metrics.py                  ← ADE and FDE metric functions
│   ├── inference.py                ← single-sample inference
│   ├── evaluate_full_dataset.py    ← full val set evaluation
│   ├── demo.py                     ← dashboard visualization
│   └── test_custom_input.py        ← custom coordinate API test
├── outputs/
│   ├── checkpoints/                ← model weights (.pth files)
│   ├── predictions/                ← saved inference outputs (.npz files)
│   └── demo/                       ← generated dashboard images (.png files)
├── tests/                          ← unit tests
├── environment.yml                 ← local conda environment
├── requirements.txt                ← pip dependencies
└── README.md
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `conda: command not found` or `'conda' is not recognized` | Close and reopen terminal after installing Miniconda. Windows users try Anaconda Prompt from Start menu. |
| `(trajpredict)` not showing in prompt | Run `conda activate trajpredict` |
| Windows `conda activate` gives execution policy error | Run `Set-ExecutionPolicy RemoteSigned` in PowerShell as Administrator |
| `No module named 'src'` | Run `cd Desktop\TrajectoryPredict` (Windows) or `cd ~/Desktop/TrajectoryPredict` (macOS/Linux) |
| `FileNotFoundError: best_model_final.pth` | Complete Step 7 — place checkpoint in `outputs/checkpoints/` |
| `No module named 'torch'` | Run `conda activate trajpredict` then reinstall torch |
| PyTorch download keeps failing | Switch to mobile hotspot, retry with `--retries 10 --timeout 120` |
| Plot window does not open | Ensure you are running on a local machine, not a remote server |
| `CUDA available: False` | No GPU detected — project still runs on CPU |
| Wrong Python version errors | Recreate environment with `conda create -n trajpredict python=3.11 -y` |