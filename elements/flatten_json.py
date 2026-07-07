import json

# Load your current JSON file
with open('/path/to/dataset/annotations/validation_set.json', 'r') as f:
    data = json.load(f)


# Function to flatten the file paths
def flatten_image_paths(images):
    for img in images:
        # Split the path and reconstruct with just 'test/' or 'train/' or 'val/'
        parts = img['file_name'].split('/')

        # Keep only the split (test/train/val) and filename
        if parts[0] in ['test', 'train', 'val']:
            img['file_name'] = f"{parts[0]}/{parts[-1]}"
        else:
            # Handle cases where the path doesn't start with split
            img['file_name'] = f"test/{parts[-1]}"


# Update the image paths
flatten_image_paths(data['images'])

# Save the modified JSON
with open('/path/to/dataset/annotations/validation_set_f.json', 'w') as f:
    json.dump(data, f, indent=2)

print("JSON file paths have been flattened successfully!")