import os
import random
import shutil

# Configuration
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'
OUTPUT_DIR = 'dataset'
SPLITS = {
    'train': 0.7,
    'val': 0.2,
    'test': 0.1
}
RANDOM_SEED = 42

# Prepare
random.seed(RANDOM_SEED)
images = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
random.shuffle(images)
total = len(images)

# Calculate split indices
train_end = int(SPLITS['train'] * total)
val_end = train_end + int(SPLITS['val'] * total)

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:]
}

# Create directories and copy files
for split_name, file_list in splits.items():
    img_out_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
    lbl_out_dir = os.path.join(OUTPUT_DIR, split_name, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for img_file in file_list:
        base = os.path.splitext(img_file)[0]
        label_file = f"{base}.txt"

        # Copy test
        shutil.copy(os.path.join(IMAGE_DIR, img_file), os.path.join(img_out_dir, img_file))
        # Copy corresponding label if exists
        src_lbl = os.path.join(LABEL_DIR, label_file)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, os.path.join(lbl_out_dir, label_file))

# Summary
print("Dataset split completed:")
for split_name in splits:
    print(f"  {split_name}: {len(splits[split_name])} test")

