import os
import numpy as np
from PIL import Image

# ====================== CONFIG ======================
# CHANGE THESE PATHS ACCORDING TO YOUR DATASET STRUCTURE

ORIGINAL_LABEL_DIR = r"D:\Comp6011 report 1\PyTorch-ENet\datasets\CamVid"

OUTPUT_DIR = r"D:\Comp6011 report 1\PyTorch-ENet\datasets\CamVid\Converted"

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# RGB Color to 11-Class ID mapping
COLOR_TO_ID = {
    (128, 128, 128): 0,   # Sky
    (0, 128, 64): 1,      # Building
    (128, 0, 0): 1,
    (64, 192, 0): 1,
    (64, 0, 64): 1,
    (192, 0, 128): 1,
    (192, 192, 128): 2,   # Pole
    (0, 0, 64): 2,
    (128, 64, 128): 3,    # Road
    (128, 0, 192): 3,
    (192, 0, 64): 3,
    (0, 0, 192): 4,       # Pavement / Sidewalk
    (64, 192, 128): 4,
    (128, 128, 192): 4,
    (128, 128, 0): 5,     # Tree
    (192, 192, 0): 5,
    (192, 128, 128): 6,   # SignSymbol
    (0, 64, 64): 6,
    (64, 64, 128): 7,     # Fence
    (64, 0, 128): 8,      # Car
    (192, 0, 192): 8,
    (64, 64, 0): 9,       # Pedestrian
    (0, 128, 192): 10,    # Bicyclist
    # All other colors → Unlabeled (255)
}

def rgb_to_class_id(rgb_tuple):
    return COLOR_TO_ID.get(rgb_tuple, 255)

def convert_labels(input_folder, output_folder, split_name):
    """Convert colored labels to 11-class grayscale labels"""
    if not os.path.exists(input_folder):
        print(f"Folder not found: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nConverting {len(files)} labels for {split_name}...")

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        
        # Open image and convert to RGB
        img = Image.open(input_path).convert('RGB')
        rgb_array = np.array(img)
        
        # Create new label array [H, W]
        h, w = rgb_array.shape[:2]
        new_label = np.zeros((h, w), dtype=np.uint8)

        # Map each pixel
        for i in range(h):
            for j in range(w):
                rgb = tuple(rgb_array[i, j])
                new_label[i, j] = rgb_to_class_id(rgb)

        # Save as grayscale image (class indices)
        output_path = os.path.join(output_folder, filename)
        Image.fromarray(new_label).save(output_path)

    print(f"✓ {split_name} conversion completed ({len(files)} files).")


# ====================== RUN CONVERSION ======================

print("Starting 32-class → 11-class label conversion...\n")

# === UPDATE THESE PATHS TO MATCH YOUR ACTUAL FOLDERS ===
convert_labels(
    r"D:\Comp6011 report 1\PyTorch-ENet\datasets\CamVid\trainannot", 
    os.path.join(OUTPUT_DIR, "train"), 
    "Train"
)

convert_labels(
    r"D:\Comp6011 report 1\PyTorch-ENet\datasets\CamVid\valannot", 
    os.path.join(OUTPUT_DIR, "val"), 
    "Validation"
)

convert_labels(
    r"D:\Comp6011 report 1\PyTorch-ENet\datasets\CamVid\testannot", 
    os.path.join(OUTPUT_DIR, "test"), 
    "Test"
)

print("\n=== Conversion Finished ===")
print(f"11-class labels saved in: {OUTPUT_DIR}")
print("You can now use this folder for training/testing.")