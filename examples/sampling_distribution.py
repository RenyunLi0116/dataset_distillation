import numpy as np
import os
import shutil

def sample_indices_one(file_path, num_samples):
    values = np.loadtxt(file_path)
    normalized_values = values / np.sum(values) # normalize weights to sum to 1

    indices = np.random.choice(len(values), size=num_samples, p=normalized_values)
    indices = indices.tolist()

    return indices

idx = sample_indices_one('./0424_val_all/high_loss_values_val_stage_20000.txt', 30)

source_dir = './data/nerf_synthetic/lego/train/'
target_dir = './0426_sampled/'

os.makedirs(target_dir, exist_ok=True)
files = [file for file in os.listdir(source_dir) if file.endswith('.png')]

# find the indices of the samples in the path "data/nerf_synthetic/lego/train/" and copy them to "0426_sampled"
for i in idx:
    source_path = os.path.join(source_dir, files[i])
    target_path = os.path.join(target_dir, files[i])
    shutil.copy(source_path, target_path)
    print(f"Copied {files[i]} to {target_dir}")

