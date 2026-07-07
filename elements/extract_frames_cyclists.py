import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

def is_inside_polygon(x, y, polygon):
    """ Check if a point (x, y) is inside the polygon. """
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

# Function for detecting objects in a frame using your existing code
def detect_objects(frame, polygon_points):
    detections = []

    # Convert frame to PIL for processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    inputs = processor(images=image_pil, return_tensors="pt")

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions
    logits = outputs.logits[0]
    bboxes = outputs.pred_boxes[0]
    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    # Confidence threshold
    threshold = 0.8

    for bbox, score, label in zip(bboxes, scores, labels):
        if score > threshold and label == 1:  # Keep only "Person" detections
            x_center, y_center, width, height = bbox.tolist()

            # Convert normalized coordinates to absolute
            x_min = (x_center - width / 2) * image_pil.width
            y_min = (y_center - height / 2) * image_pil.height
            x_max = (x_center + width / 2) * image_pil.width
            y_max = (y_center + height / 2) * image_pil.height

            # Check if center of bbox is inside polygon
            if is_inside_polygon(x_center * image_pil.width, y_center * image_pil.height, polygon_points):
                detections.append([[x_min, y_min, x_max - x_min, y_max - y_min], float(score), int(label)])

    return detections

# Load DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

# Base directory containing the date folders
base_dir = "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid"

# List of valid video files
valid_video_files = [
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221215/08/pc5-1832-PSFV-01-N413-Zuid_20221215_083000271.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221215/08/pc5-1832-PSFV-01-N413-Zuid_20221215_084500219.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221216/08/pc5-1832-PSFV-01-N413-Zuid_20221216_083001283.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221216/08/pc5-1832-PSFV-01-N413-Zuid_20221216_084501313.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221217/08/pc5-1832-PSFV-01-N413-Zuid_20221217_083000119.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221217/08/pc5-1832-PSFV-01-N413-Zuid_20221217_084500103.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221218/08/pc5-1832-PSFV-01-N413-Zuid_20221218_083000915.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221218/08/pc5-1832-PSFV-01-N413-Zuid_20221218_084500947.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221219/08/pc5-1832-PSFV-01-N413-Zuid_20221219_083001903.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221219/08/pc5-1832-PSFV-01-N413-Zuid_20221219_084501903.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221220/08/pc5-1832-PSFV-01-N413-Zuid_20221220_083000209.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221220/08/pc5-1832-PSFV-01-N413-Zuid_20221220_084500206.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221221/08/pc5-1832-PSFV-01-N413-Zuid_20221221_083000767.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221221/08/pc5-1832-PSFV-01-N413-Zuid_20221221_084500739.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221222/08/pc5-1832-PSFV-01-N413-Zuid_20221222_083000042.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221222/08/pc5-1832-PSFV-01-N413-Zuid_20221222_084500038.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221223/08/pc5-1832-PSFV-01-N413-Zuid_20221223_083000655.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221223/08/pc5-1832-PSFV-01-N413-Zuid_20221223_084500816.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221224/08/pc5-1832-PSFV-01-N413-Zuid_20221224_083001769.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221224/08/pc5-1832-PSFV-01-N413-Zuid_20221224_084501729.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221225/08/pc5-1832-PSFV-01-N413-Zuid_20221225_083000156.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221225/08/pc5-1832-PSFV-01-N413-Zuid_20221225_084500115.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221226/08/pc5-1832-PSFV-01-N413-Zuid_20221226_083000753.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221226/08/pc5-1832-PSFV-01-N413-Zuid_20221226_084500716.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221227/08/pc5-1832-PSFV-01-N413-Zuid_20221227_083000789.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221227/08/pc5-1832-PSFV-01-N413-Zuid_20221227_084500765.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221228/08/pc5-1832-PSFV-01-N413-Zuid_20221228_083001637.mp4",
    "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/20221228/08/pc5-1832-PSFV-01-N413-Zuid_20221228_084500162.mp4"
]


# Define your polygon points
coords = "169,1079,1005,195,1120,221,1010,1079"
coords_list = list(map(int, coords.split(",")))
polygon_points = np.array(coords_list, np.int32).reshape(-1, 1, 2)  # OpenCV format


# Iterate through each video file
import os
import cv2

# Assuming valid_video_files and base_dir are defined elsewhere
base_dir = "/path/to/dataset/Utrecht/PSFV-01-N413-Zuid/Utrecht"  # Change this to your actual base directory

for video_path in valid_video_files:
    # Extract the date and time from the video path
    date_part = video_path.split("/")[-1].split("_")[1]  # Assuming date is in the filename
    year_month_day = date_part[:8]  # Extract YYYYMMDD

    # Create a directory for frames based on the date
    frames_dir = os.path.join(base_dir, year_month_day, "08", "cyclist_frames")
    os.makedirs(frames_dir, exist_ok=True)  # Ensure the directory exists

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize frame counter
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        # If no frame is returned, we've reached the end of the video
        if not ret:
            break

        # Process every 5th frame
        if frame_idx % 5 == 0:
            # Detect objects in the frame
            detections = detect_objects(frame, polygon_points)

            # If there are detections, save the frame
            if detections:
                # Extract the original video filename (without extension)
                video_filename = os.path.basename(video_path).rsplit(".", 1)[0]

                # Create the frame filename with the original video name and frame number
                frame_filename = f"{video_filename}_frame_{frame_idx:04d}.jpg"

                # Save the frame as an image
                frame_save_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_save_path, frame)

                print(f"Saved frame {frame_filename} with detections in {frames_dir}")

        # Increment the frame counter
        frame_idx += 1

    # Release the video capture object
    cap.release()

print("Frame extraction complete.")
