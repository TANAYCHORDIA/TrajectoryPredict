import os
import numpy as np

os.makedirs("data/processed", exist_ok=True)

OBS_LEN = 4
PRED_LEN = 6
MAX_NEIGHBORS = 4

def make_split(n_samples: int):
	inputs = np.random.randn(n_samples, OBS_LEN, 4).astype(np.float32)
	targets = np.random.randn(n_samples, PRED_LEN, 2).astype(np.float32)
	social = np.random.randn(n_samples, OBS_LEN, MAX_NEIGHBORS, 2).astype(np.float32)

	# Random neighbor validity mask in [0,1], then hard-binarize.
	mask = (np.random.rand(n_samples, OBS_LEN, MAX_NEIGHBORS) > 0.4).astype(np.float32)
	social = social * mask[..., None]
	return inputs, targets, social, mask

splits = {
	"train": 200,
	"val": 50,
	"test": 50,
}

for split, n in splits.items():
	split_inputs, split_targets, split_social, split_mask = make_split(n)
	np.save(f"data/processed/{split}_inputs.npy", split_inputs)
	np.save(f"data/processed/{split}_targets.npy", split_targets)
	np.save(f"data/processed/{split}_social.npy", split_social)
	np.save(f"data/processed/{split}_mask.npy", split_mask)

print("Fake data created successfully.")
print("Generated files:")
for split, n in splits.items():
	print(f"- {split}: {n} samples | inputs [N,{OBS_LEN},4], targets [N,{PRED_LEN},2], social [N,{OBS_LEN},{MAX_NEIGHBORS},2], mask [N,{OBS_LEN},{MAX_NEIGHBORS}]")