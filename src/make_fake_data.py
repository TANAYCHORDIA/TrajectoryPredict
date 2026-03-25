import os
import numpy as np

os.makedirs("data/processed", exist_ok=True)

# Fake training data
obs_train = np.random.randn(200, 8, 4).astype(np.float32)
fut_train = np.random.randn(200, 12, 2).astype(np.float32)

# Fake validation data
obs_val = np.random.randn(50, 8, 4).astype(np.float32)
fut_val = np.random.randn(50, 12, 2).astype(np.float32)

np.save("data/processed/obs_train.npy", obs_train)
np.save("data/processed/fut_train.npy", fut_train)
np.save("data/processed/obs_val.npy", obs_val)
np.save("data/processed/fut_val.npy", fut_val)

print("Fake data created successfully.")
print("obs_train shape:", obs_train.shape)
print("fut_train shape:", fut_train.shape)
print("obs_val shape:", obs_val.shape)
print("fut_val shape:", fut_val.shape)