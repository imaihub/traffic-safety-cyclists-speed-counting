import os
import shutil

# Source directory where frames are stored (including date folders)
source_base = "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid"

# Target directories
target_base = "/path/to/dataset"
train_dir = os.path.join(target_base, "train")
val_dir = os.path.join(target_base, "val")
test_dir = os.path.join(target_base, "test")

# Create target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Date to split mapping
date_to_split = {
    "train": ["20221216", "20221219", "20221222", "20221224", "20221227"],
    "val": ["20221215", "20221220", "20221226"],
    "test": ["20221217", "20221218", "20221221", "20221223", "20221225", "20221228"]
}

# Reverse mapping for easier lookup
split_lookup = {}
for split, dates in date_to_split.items():
    for date in dates:
        split_lookup[date] = split

# Walk through source directory
for root, dirs, files in os.walk(source_base):
    # Extract date from path (assuming structure: .../test/20221217/08/frames)
    path_parts = root.split('/')
    if len(path_parts) >= 2:
        date = path_parts[-3]  # Gets the date part (20221217)

        # Check if this date is in our mapping
        if date in split_lookup:
            target_dir = os.path.join(target_base, split_lookup[date])

            # Copy all files in this directory
            for file in files:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)

                # Copy file (use shutil.copy2 to preserve metadata)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} → {dst_path}")

print("All frames redistributed successfully!")