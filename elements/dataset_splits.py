import os
import shutil

# Base paths
base_dir = "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid"
destination_base = "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid"


#Dataset split for annotations V2, after meeting on the 13/05/2025 with MAYA
base_dir = "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid"

# Target redistribution
new_split = {
    "train": ["20221216", "20221219", "20221222", "20221224", "20221227"],
    "val": ["20221215", "20221220", "20221226"],
    "test": ["20221217", "20221218", "20221221", "20221223", "20221225", "20221228"]
}

# Iterate over all splits and move folders to the correct location
for target_split, dates in new_split.items():
    target_path = os.path.join(base_dir, target_split)
    os.makedirs(target_path, exist_ok=True)

    for date_folder in dates:
        # Search current location of the folder (in train, val, or test)
        found = False
        for current_split in ["train", "val", "test"]:
            current_path = os.path.join(base_dir, current_split, date_folder)
            if os.path.exists(current_path):
                shutil.move(current_path, os.path.join(target_path, date_folder))
                print(f"Moved {date_folder} from {current_split} → {target_split}")
                found = True
                break

        if not found:
            print(f"⚠️ Folder {date_folder} not found in any split!")

