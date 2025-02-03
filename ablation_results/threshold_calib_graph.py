import os
import matplotlib.pyplot as plt
import numpy as np
from threshold_calib_dict import threshold_model_dictionary

# Create the folder for saving heatmaps
output_folder = "token_threshold_heatmap"
os.makedirs(output_folder, exist_ok=True)

# Function to sanitize keys for file names
def sanitize_key(key):
    return key.replace("/", "_").replace(" ", "_").replace(".", "_")

# Generate heatmaps
for key, value in threshold_model_dictionary.items():
    # Verify that the value is a tensor with the expected shape (31, 32)
    value = value.cpu().numpy()
    if isinstance(value, np.ndarray) and value.shape == (31, 32):
        sanitized_key = sanitize_key(key)
        file_path = os.path.join(output_folder, f"{sanitized_key}.pdf")
        
        # Create the heatmap
        plt.figure(figsize=(8, 8))  # Square plot
        plt.imshow(value, cmap="viridis", aspect="auto")
        plt.colorbar(label="Threshold Value")
        plt.title(key)  # Use the original key as the title
        plt.xlabel("Heads")
        plt.ylabel("Layers")
        plt.xticks(range(32), labels=range(1, 33))  # 1-based index for clarity
        plt.yticks(range(31), labels=range(1, 32))  # 1-based index for clarity
        plt.tight_layout()
        
        # Save the heatmap
        plt.savefig(file_path, format="pdf")
        plt.close()
    else:
        print(f"Skipping key {key} - value is not a tensor of shape (31, 32)")
