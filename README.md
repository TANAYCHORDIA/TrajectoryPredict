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

**1. Clone the repository**
```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

**2. Create and activate the environment**
```bash
conda create -n trajpredict python=3.11
conda activate trajpredict
```

**3. Install dependencies**
```bash
# Install PyTorch first — handles CUDA compatibility automatically
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# Install remaining pipeline dependencies
pip install -r requirements.txt
```

> Ensure `best_model_final.pth` is in `outputs/checkpoints/` before running inference.

---

## 🚀 How to Run

**1. Generate Visual Demo**

Runs inference on 6 curated scenes and renders dashboards showing past trajectory, ground truth, and 3 multimodal predictions.
```bash
python -m src.demo
```
Output saved to `outputs/demo/`.

**2. Evaluate Full Dataset Metrics**

Independently verifies ADE and FDE across the full validation split with social features.
```bash
python -m src.evaluate_full_dataset
```

**3. Test Custom Inputs**

Tests the production API with arbitrary raw `(x, y)` coordinates and renders a live visualization.
```bash
python -m src.test_custom_input
```