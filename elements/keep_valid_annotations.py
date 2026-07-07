# Define file paths
import json

train_annotation_file = "/path/to/dataset/annotations/training_set.json"
val_annotation_file = "/path/to/dataset/annotations/validation_set.json"
test_annotation_file = "/path/to/dataset/annotations/testing_set.json"

def count_and_save_annotated_images(annotation_file, output_txt_file):
    """Counts images with annotations and saves their filenames to a text file."""
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    # Get all image IDs and corresponding filenames
    image_ids = {image["id"]: image["file_name"] for image in coco_data["images"]}

    # Get image IDs that have annotations
    annotated_image_ids = {ann["image_id"] for ann in coco_data["annotations"]}

    # Get filenames of images with annotations
    annotated_image_filenames = [image_ids[image_id] for image_id in annotated_image_ids]

    # Write the filenames to a text file
    with open(output_txt_file, "w") as f:
        for filename in annotated_image_filenames:
            f.write(f"{filename}\n")

    # Find images with no annotations
    empty_images = set(image_ids.values()) - set(annotated_image_filenames)

    print(f"Total images: {len(image_ids)}")
    print(f"Images with annotations: {len(annotated_image_filenames)}")
    print(f"Images without annotations: {len(empty_images)}")

    return len(empty_images)


# Run for each dataset split and save filenames with annotations in text files
print("\nTraining Set:")
train_empty = count_and_save_annotated_images(train_annotation_file, "/path/to/dataset/final_ds_management/artifacts/train_images_with_annotations_f.txt")

print("\nValidation Set:")
val_empty = count_and_save_annotated_images(val_annotation_file, "/path/to/dataset/final_ds_management/artifacts/val_images_with_annotations_f.txt")

print("\nTest Set:")
test_empty = count_and_save_annotated_images(test_annotation_file, "/path/to/dataset/final_ds_management/artifacts/test_images_with_annotations_f.txt")

# Summary
print("\nSummary:")
print(f"Training images without annotations: {train_empty}")
print(f"Validation images without annotations: {val_empty}")
print(f"Test images without annotations: {test_empty}")


with open(train_annotation_file) as f:
    train_data = json.load(f)
with open(val_annotation_file) as f:
    val_data = json.load(f)
with open(test_annotation_file) as f:
    test_data = json.load(f)

print("Train first 2 images:", train_data["images"][:2])
print("Val first 2 images:", val_data["images"][:2])
print("Test first 2 images:", test_data["images"][:2])
