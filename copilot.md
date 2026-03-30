I am the Data Engineer on a 7-day ML hackathon project called TrajectoryPredict.
My job is to build the complete data pipeline from raw nuScenes JSON files to a 
clean PyTorch DataLoader. Nothing else — no model code, no training loop.

─────────────────────────────────────────────────────────────────
DATASET STRUCTURE
─────────────────────────────────────────────────────────────────
The dataset is a folder of raw nuScenes JSON files. When unpacked it contains:
- attribute.json
- calibrated_sensor.json
- category.json
- ego_pose.json
- instance.json
- log.json
- map.json
- sample.json
- sample_annotation.json
- sample_data.json
- scene.json
- sensor.json
- visibility.json

The files we care about most are:
- scene.json          → top level, each scene is ~20 seconds of driving
- sample.json         → keyframes within a scene, at 2Hz
- sample_annotation.json → bounding boxes + agent positions per keyframe
- instance.json       → links annotations to a single agent across time
- category.json       → agent type labels (pedestrian, vehicle.bicycle, etc.)

DELIVERABLE 1 — Preprocessing Pipeline (Day 1 afternoon)
Write src/data/preprocess.py that:
- Loads data/processed/tracks_raw.csv (Assume this file already exists and contains: [scene_name, instance_token, timestamp, x, y, category] sorted chronologically).
- Groups data by scene_name and instance_token to process individual agent tracks.
- Maintains the native 2Hz sampling rate (DO NOT interpolate).
- Slides a window over each track:
  - Input window: 4 consecutive timesteps (2 seconds)
  - Output window: 6 consecutive timesteps (3 seconds)
  - Total sequence length needed: 10 timesteps. Skip any track shorter than this.
  - Stride = 1.
- Computes velocity features for the input window:
  - dx = x[t] - x[t-1]
  - dy = y[t] - y[t-1]
  - For the first frame of the input window, dx=0 and dy=0.
- Applies Agent-Centric Normalization (Translation AND Rotation) to the entire 10-timestep sequence based on the LAST observed input timestep (index 3):
  1. Translation: Subtract the (x,y) of index 3 from all points so index 3 becomes (0,0).
  2. Rotation: Calculate the heading angle using the velocity (dx, dy) at index 3. Construct a 2D rotation matrix to rotate the entire sequence so that this final velocity vector points straight along the positive X-axis.
- Applies scene-level train/val/test split:
  - CRITICAL: split by scene_name, never by frame or by agent.
  - Ratio: 70% train / 15% val / 15% test.
  - Use random seed 42.
- Saves the processed, chunked sequences as numpy arrays or tensors:
  - data/processed/train_inputs.npy (shape: [N, 4, 4] containing x, y, dx, dy)
  - data/processed/train_targets.npy (shape: [N, 6, 2] containing x, y)
  - Same for val and test.

DELIVERABLE 2 — PyTorch Dataset and DataLoader (Day 1 evening)
Write src/data/dataset.py that:
- Defines a TrajectoryDataset class inheriting from torch.utils.data.Dataset.
- Constructor loads the saved .npy files.
- __getitem__ returns:
  - input tensor: shape [4, 4] dtype float32
  - target tensor: shape [6, 2] dtype float32
- Write get_dataloaders(data_dir, batch_size=64) returning train_loader, val_loader, test_loader. (Train is shuffled, others are not).
- Write a sanity check under if __name__ == "__main__":
  - Prints one batch shape from each loader to confirm [batch, 4, 4] and [batch, 6, 2].

DELIVERABLE 3 — Neighbor Extraction for Social Pooling (Day 4)
Write src/data/social.py that:
- Defines get_neighbors(agent_positions, all_agent_positions, radius=2.0)
  - agent_positions: tensor [4, 2] — current agent
  - all_agent_positions: list of tensors [4, 2] — other agents
  - Returns a list of length 4 containing nearby agent positions at each timestep.
- Write tests/test_social.py with 3 basic unit tests (1m apart, 5m apart, empty scene).

─────────────────────────────────────────────────────────────────
HARD RULES
─────────────────────────────────────────────────────────────────
- Input tensors must be exactly [batch, 4, 4].
- Target tensors must be exactly [batch, 6, 2].
- Use math/numpy for the 2D rotation matrix. Do not skip the rotation step.
- Split strictly by scene_name.

─────────────────────────────────────────────────────────────────
FILE STRUCTURE TO CREATE
─────────────────────────────────────────────────────────────────
src/data/preprocess.py
src/data/dataset.py
src/data/social.py
src/data/__init__.py        ← empty file, just marks it as a package
tests/test_social.py
tests/test_dataset.py       ← sanity checks for tensor shapes and split integrity

─────────────────────────────────────────────────────────────────
DATASET ROOT PATH
─────────────────────────────────────────────────────────────────
Assume the input file data/processed/tracks_raw.csv is already generated 
and available. You do not need to parse the raw nuScenes JSONs.

─────────────────────────────────────────────────────────────────
DEFINITION OF DONE FOR MY ROLE
─────────────────────────────────────────────────────────────────
My pipeline is complete when:
1. python src/data/preprocess.py runs without error and produces train/val/test .npy arrays.
2. python src/data/dataset.py runs the sanity check and prints correct tensor shapes.
3. Input shape confirmed as exactly [batch, 4, 4] and target shape as exactly [batch, 6, 2].
4. No scene_name appears in more than one split (no leakage).
5. Unit tests in tests/test_social.py all pass.

Build these three files (preprocess.py, dataset.py, social.py) in order. 
Do not skip ahead to social.py before dataset.py is working and verified.