import numpy as np
import os
import shutil

import json


def sample_indices_one(file_path, num_samples):
    values = np.loadtxt(file_path)
    normalized_values = values / np.sum(values) # normalize weights to sum to 1

    indices = np.random.choice(len(values), size=num_samples, p=normalized_values)
    indices = indices.tolist()

    return indices


idx = sample_indices_one('./0424_val_all/high_loss_values_val_stage_20000.txt', 30)

# Path to the original JSON file
input_file_path = './data/nerf_synthetic/lego/transforms_train.json'
output_file_path = './data/nerf_synthetic/lego/transforms_train_sample30.json'

# Step 1: Read the JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Step 2: Extract elements using the list of indices
selected_frames = [data['frames'][i] for i in idx if i < len(data['frames'])]

# Step 3: Create a new JSON object
new_json = {
    "camera_angle_x": data["camera_angle_x"],  # Keep the camera angle if needed
    "frames": selected_frames
}

# Step 4: Write the new JSON object to a file
with open(output_file_path, 'w') as file:
    json.dump(new_json, file, indent=4)  # Use indent for pretty printing

print("New JSON file has been created with selected elements.")
