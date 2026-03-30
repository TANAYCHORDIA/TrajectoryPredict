# 🧭 Autonomous Trajectory & Intent Prediction (L4)

## 📌 Project Overview

This project is a production-ready Behavioral AI pipeline designed to predict multimodal pedestrian and cyclist trajectories in complex urban environments. Built for autonomous vehicle systems, the model takes a 2-second history of an agent's coordinates and accurately predicts their most likely paths 3 seconds into the future.

Instead of relying on naive constant-velocity assumptions, this system successfully captures the uncertainty of human movement and avoids collisions by understanding the social dynamics of the surrounding environment.

---

## 🧠 Model Architecture

The system utilizes a **Multimodal LSTM + Social Pooling** architecture to generate context-aware predictions.

1. **Coordinate Normalization (Rotational Invariance):** Raw global `(x, y)` coordinates are translated to an agent-centric origin. The entire scene is then rotated so the primary agent's velocity vector aligns with the positive X-axis. This ensures the model learns movement patterns rather than memorizing global map locations.
2. **Social Context Pooling:** For every primary agent, the relative coordinates of all neighboring agents within a defined radius are extracted and encoded. This prevents predicted collisions and allows the model to understand group dynamics (e.g., people walking together or avoiding each other).
3. **Multimodal LSTM Backbone:** A recurrent neural network processes the temporal sequence and outputs a `[3, 6, 2]` tensor. This represents the **3 most likely predicted paths** (modes) over the next 6 time steps (3 seconds).
4. **Inverse Rotation API:** Predictions are mathematically un-rotated back into the global map frame for seamless integration into downstream autonomous vehicle simulators or visualization dashboards.

---

## 📊 Performance Metrics

The model was evaluated across 74 complex, interactive urban scenes in the nuScenes validation split. By incorporating Social Pooling to understand neighbor dynamics, the model achieves sub-meter accuracy globally.

| Model Iteration | Architecture | Social Context | Mean ADE (m) | Mean FDE (m) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Constant Velocity Extrapolation | ❌ | > 2.50 | > 4.00 |
| **Improved** | Vanilla LSTM (Unimodal) | ❌ | 1.85 | 3.42 |
| **Final Submission** | **LSTM + Social Pooling** | ✅ | **0.2308** | **0.4122** |

*Note: ADE (Average Displacement Error) and FDE (Final Displacement Error) are measured in meters.*

---

## 📂 Dataset Used

The model was trained and evaluated on the **nuScenes** dataset (Mini Split), focusing specifically on pedestrian and vehicle track features.

🔗 **[Download the Dataset on Kaggle]https://www.kaggle.com/datasets/tanaychordia/trajectorypredict**

*(If running locally, place the raw dataset files in the `data/raw/` directory before running the preprocessing scripts).* 

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/TANAYCHORDIA/TrajectoryPredict.git
cd TrajectoryPredict
```

### 2. Create and activate the environment

```bash
conda create -n trajpredict python=3.10
conda activate trajpredict
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Ensure the pre-trained weights file `best_social_model.pth` is located in `outputs/checkpoints/` before running inference.

---

## 🚀 How to Run the Code

### 1. Generate the Visual Demo (Dashboard Visualization)

This script runs inference on 5 distinct curated scenes (straight paths, sharp turns, intersections) and renders high-fidelity autonomous vehicle dashboards displaying the past trajectory, ground truth, and 3 multimodal predictions.

```bash
python -m src.demo
```

Outputs are automatically saved as PNG images in `outputs/demo/`.

### 2. Evaluate Full Dataset Metrics

To independently verify ADE and FDE metrics across the entire validation dataset in memory:

```bash
python -m src.evaluate_full_dataset
```

### 3. Test Custom Inputs (API Deliverable)

To test the model with a custom history of raw `(x, y)` coordinates, run the custom input tester. This will render a real-time visualization on your screen.

```bash
python -m src.test_custom_input
```


