import argparse
import csv
import os
from typing import Dict, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from transformers import DetrImageProcessor

from elements.constants import (  # Import general constants
    CONFIDENCE_THRESHOLD, FPS  # Using FPS from constants for consistency
    # Other constants will be defined locally or based on provided snippet's values
)
# Import necessary components from the elements folder
from elements.models import Detr  # Assuming Detr class is defined here as a LightningModule
from elements.utils import (
    point_to_line_distance,
    # get_crossing_lines is not used directly, but its logic implies line definitions
    save_tracked_video, visualize_directions, load_annotated_images_list
)

# --- Define Constants specific to this GT speed generation logic ---
# These are derived from your provided snippet.
# TOTAL_DISTANCE_METERS is a fixed real-world distance between the "first" and "last" lines.
BLOCK_LENGTH = 0.5  # meters
INTER_BLOCK_SPACING = 0.5  # meters
NUM_BLOCKS = 14
NUM_GAPS = 13
TOTAL_DISTANCE_METERS = (NUM_BLOCKS * BLOCK_LENGTH) + (NUM_GAPS * INTER_BLOCK_SPACING)

# Line definitions for speed measurement (pixel coordinates from your snippet)
# These define the 'start' and 'end' points of the lines in the image plane.
FIRST_LINE = {
    "start": (832, 544),
    "end": (1085, 548)
}

LAST_LINE = {
    "start": (930, 387),
    "end": (1103, 387)
}

# General thresholds and debug flag (from your snippet)
LINE_DETECTION_THRESHOLD = 15  # Used by point_to_line_distance implicitly for close points
MIN_CROSSING_FRAMES = 5
MAX_CROSSING_FRAMES = 2000  # Large value to capture all valid crossings
DEBUG = True  # Set to True for verbose output
ENTRY_THRESHOLD = 15  # Pixel distance to line to consider an entry
EXIT_THRESHOLD = 15  # Pixel distance to line to consider an exit
MIN_SPEED = 3  # Min calculated speed in km/h to be considered valid
MAX_SPEED = 50  # Max calculated speed in km/h to be considered valid
MIN_PIXEL_DISTANCE = 50  # Minimum pixel distance traveled for a valid crossing

# CSV Headers for the generated Ground Truth Speeds file
GT_HEADERS = ["track_id", "speed_kph", "direction", "crossing_time_sec", "frames_taken", "entry_frame", "exit_frame"]


# --- Core Processing Function ---
def process_frame_for_gt_speed(
        frame_path: str,
        frame_id: int,
        polygon_points: np.ndarray,  # Polygon for ROI
        homography_matrix: np.ndarray,  # Homography matrix for pixel_to_realworld if needed
        detr_model: Detr,  # Loaded DETR model
        detr_processor: DetrImageProcessor,  # DETR image processor
        deepsort_tracker: DeepSort,  # DeepSort tracker instance
        track_info: Dict,  # Dictionary to maintain state for each track
        gt_csv_writer: Any,  # CSV writer object to write speeds
        all_vis_data_list: List  # List to store data for video visualization
):
    """
    Processes a single video frame: performs object detection (DETR), tracking (DeepSort),
    calculates cyclist speeds based on predefined lines and distance, and logs these as
    "ground truth" speeds to a CSV file. It also prepares data for video visualization.

    Args:
        frame_path (str): Full path to the current frame image.
        frame_id (int): Sequential index of the current frame.
        polygon_points (np.ndarray): Polygon defining the region of interest for detection.
        homography_matrix (np.ndarray): 3x3 homography matrix (for pixel_to_realworld).
        detr_model (Detr): Initialized DETR PyTorch Lightning model.
        detr_processor (DetrImageProcessor): Hugging Face DETR image processor.
        deepsort_tracker (DeepSort): Initialized DeepSort tracker.
        track_info (Dict): Dictionary storing persistent information for each track.
        gt_csv_writer (Any): CSV writer object opened in 'append' mode to write speed data.
        all_vis_data_list (List): List to accumulate frame visualization data for video saving.
    """
    # 1. Load and prepare image
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)  # RGB format (for matplotlib, visualization)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # BGR format (for OpenCV, DeepSort)

    # 2. DETR Object Detection
    # Prepare image for DETR model
    inputs = detr_processor(images=image, return_tensors="pt").to(detr_model.device)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = detr_model.model(**inputs)  # Pass inputs through the underlying Hugging Face model

    # Post-process DETR detections to get bounding boxes and scores
    results = detr_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]).to(detr_model.device),
        threshold=CONFIDENCE_THRESHOLD
    )[0]  # Get results for the first (and only) image in batch

    # 3. Format Detections for DeepSort
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 1:  # Only process 'cyclist' class (label ID 1)
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            center_x = x_min + width / 2
            center_y = y_min + height / 2

            # Filter detections to be within the defined polygon region of interest
            if cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0:
                # Extract the bounding box patch for feature embedding (DeepSort requirement)
                bbox_patch = image_bgr[int(y_min):int(y_max), int(x_min):int(x_max)]
                if bbox_patch.size == 0:  # Skip empty patches
                    continue
                # DeepSort expects detection in format: ([x, y, width, height], score, feature_data)
                detection = ([x_min, y_min, width, height], float(score), bbox_patch)
                detections.append(detection)

    # 4. Update DeepSort Tracker
    tracks = deepsort_tracker.update_tracks(detections, frame=image_bgr)

    # Prepare data for frame visualization (a copy to draw on)
    frame_vis_data = {"frame": image_np.copy(), "tracks": []}

    # Draw the configured measurement lines on the visualization frame
    cv2.line(frame_vis_data["frame"], FIRST_LINE["start"], FIRST_LINE["end"], (255, 0, 0), 3)  # Blue (BGR)
    cv2.line(frame_vis_data["frame"], LAST_LINE["start"], LAST_LINE["end"], (255, 0, 0), 3)  # Blue (BGR)

    # 5. Process Tracks for Speed Estimation and Logging
    for track in tracks:
        # Only consider confirmed tracks (stable tracks)
        if not track.is_confirmed():
            continue

        # Get bounding box and center point of the track
        ltrb = track.to_ltrb()  # (left, top, right, bottom)
        x_min, y_min, x_max, y_max = ltrb
        width = x_max - x_min
        height = y_max - y_min
        center = (x_min + width / 2, y_min + height / 2)  # Center of bbox for line checks
        tracking_point = center  # Point used for tracking across lines
        confidence = getattr(track, 'det_conf', 1.0)  # Get detection confidence

        # Initialize track-specific information if it's a new track
        if track.track_id not in track_info:
            track_info[track.track_id] = {
                'positions': [],
                'entry_frame': None,
                'exit_frame': None,
                'entry_line': None,  # Which line was crossed first ('FIRST' or 'LAST')
                'speed': None,
                'direction': None,  # 'northbound' or 'southbound'
                'in_measurement': False,  # Flag if currently in the speed measurement zone
                'last_position': None
            }

        # Store position history for movement direction analysis
        track_info[track.track_id]['positions'].append(tracking_point)
        if len(track_info[track.track_id]['positions']) > 50:  # Keep a limited history
            track_info[track.track_id]['positions'].pop(0)

        # Check for entry into the measurement zone
        if not track_info[track.track_id]['in_measurement']:
            first_line_dist = point_to_line_distance(tracking_point, FIRST_LINE)
            last_line_dist = point_to_line_distance(tracking_point, LAST_LINE)

            if DEBUG:
                tqdm.write(
                    f"  [Frame {frame_id}] ID {track.track_id} - Dist to FIRST: {first_line_dist:.1f}, LAST: {last_line_dist:.1f}")

            if first_line_dist < ENTRY_THRESHOLD:
                track_info[track.track_id].update({
                    'entry_line': 'FIRST',
                    'entry_frame': frame_id,
                    'entry_position': tracking_point,
                    'in_measurement': True,
                    'last_position': tracking_point
                })
                if DEBUG:
                    tqdm.write(f"  [Frame {frame_id}] ENTRY: ID {track.track_id} at FIRST line (frame {frame_id})")
            elif last_line_dist < ENTRY_THRESHOLD:
                track_info[track.track_id].update({
                    'entry_line': 'LAST',
                    'entry_frame': frame_id,
                    'entry_position': tracking_point,
                    'in_measurement': True,
                    'last_position': tracking_point
                })
                if DEBUG:
                    tqdm.write(f"  [Frame {frame_id}] ENTRY: ID {track.track_id} at LAST line (frame {frame_id})")

        # If track is in measurement, check for exit
        elif track_info[track.track_id]['in_measurement'] and not track_info[track.track_id]['exit_frame']:
            exit_detected = False
            direction_str = None
            target_line = None

            # Determine the target exit line and expected direction based on entry line
            if track_info[track.track_id]['entry_line'] == 'FIRST':
                target_line = LAST_LINE
                # 'southbound' is typically moving towards higher Y values in pixel coordinates (down)
                # 'northbound' is typically moving towards lower Y values (up)
                direction_str = 'southbound'
            else:  # Entry from LAST_LINE
                target_line = FIRST_LINE
                direction_str = 'northbound'

            current_dist = point_to_line_distance(tracking_point, target_line)
            prev_pos = track_info[track.track_id]['last_position']
            prev_dist = point_to_line_distance(prev_pos, target_line) if prev_pos else float('inf')

            # Exit condition: current position is very close to target line OR moving towards it and very close
            if (current_dist < EXIT_THRESHOLD) or (current_dist < 2 * EXIT_THRESHOLD and current_dist < prev_dist):
                exit_detected = True

                # Additional validation: ensure minimum pixel distance was traveled
                entry_pos = track_info[track.track_id]['entry_position']
                distance_traveled_pixels = np.linalg.norm(np.array(tracking_point) - np.array(entry_pos))

                if distance_traveled_pixels < MIN_PIXEL_DISTANCE:
                    if DEBUG:
                        tqdm.write(
                            f"  [Frame {frame_id}] ID {track.track_id}: Insufficient pixel distance traveled ({distance_traveled_pixels:.1f}px). Exit aborted.")
                    exit_detected = False

            if exit_detected:
                frames_taken = frame_id - track_info[track.track_id]['entry_frame']

                # Validate the duration of the crossing
                if MIN_CROSSING_FRAMES <= frames_taken <= MAX_CROSSING_FRAMES:
                    time_sec = frames_taken / FPS

                    # Calculate speed using the fixed TOTAL_DISTANCE_METERS
                    speed_kph = (TOTAL_DISTANCE_METERS / time_sec) * 3.6

                    # Validate speed within reasonable bounds
                    if MIN_SPEED <= speed_kph <= MAX_SPEED:
                        track_info[track.track_id].update({
                            'exit_frame': frame_id,
                            'speed': speed_kph,
                            'direction': direction_str,
                            'crossing_time': time_sec
                        })
                        # Write the "ground truth" speed entry to CSV
                        gt_csv_writer.writerow([
                            track.track_id,
                            f"{speed_kph:.1f}",  # Format for CSV output
                            direction_str,
                            f"{time_sec:.2f}",
                            frames_taken,
                            track_info[track.track_id]['entry_frame'],
                            frame_id  # exit_frame
                        ])
                        if DEBUG:
                            tqdm.write(
                                f"  [Frame {frame_id}] SPEED GT: ID {track.track_id} - {speed_kph:.1f} km/h ({direction_str}) "
                                f"({time_sec:.2f}s, {frames_taken} frames)")
                    else:
                        if DEBUG:
                            tqdm.write(
                                f"  [Frame {frame_id}] INVALID SPEED GT: {speed_kph:.1f} km/h (discarded due to range).")
                else:
                    if DEBUG:
                        tqdm.write(
                            f"  [Frame {frame_id}] INVALID CROSSING TIME GT: {frames_taken} frames (discarded due to range).")

        # Update last known position for movement vector calculation in next frame
        track_info[track.track_id]['last_position'] = tracking_point

        # Add track info to frame_vis_data for video output
        frame_vis_data["tracks"].append({
            "track_id": track.track_id,
            "bbox": [x_min, y_min, x_max, y_max],  # Store as ltrb for consistency in save_tracked_video
            "confidence": confidence
        })

    # Add the current frame's visualization data to the overall list
    all_vis_data_list.append(frame_vis_data)


def main():
    """
    Main function to run the process of generating ground truth speed values
    by applying a DETR + DeepSort pipeline on annotated frames.
    """
    parser = argparse.ArgumentParser(
        description="Generate Ground Truth Speed Values from Tracking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments for input data and model
    parser.add_argument('--frames-dir', required=True,
                        help='Directory containing frame images')
    parser.add_argument('--annotations-list', required=True,
                        help='Path to text file with list of annotated images to process')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to DETR model checkpoint file')

    # Output arguments
    parser.add_argument('--gt-output-csv', default="results/gt_speeds_generated_fullds.csv",
                        help='Path to save the generated ground truth speed CSV file')
    parser.add_argument('--output-video', default="results/gt_speed_visualization_video.mp4",
                        help='Path to save output video with visualizations of tracking and speed calculation.')

    # Configuration for tracking and speed calculation
    parser.add_argument('--polygon-coords', default="937,390,1100,388,1003,1080,469,1079",
                        help='Comma-separated polygon coordinates (x1,y1,x2,y2,...) defining the region of interest for detection and tracking.')
    parser.add_argument('--embedder', default="mobilenet",
                        help='DeepSort embedder type (e.g., "torchreid", "mobilenet", "clip_ViT-B/32").')
    parser.add_argument('--embedder-weights', default=None,
                        help='Path to pre-trained weights for the DeepSort embedder (e.g., for CLIP models).')
    parser.add_argument('--fps', type=int, default=FPS,  # Use FPS from elements.constants
                        help='Frames per second for output video and speed calculations.')
    parser.add_argument('--progress-interval', type=int, default=50,
                        help='Interval (in frames) to print progress updates to console.')

    args = parser.parse_args()

    # --- 1. Validate and Prepare Paths ---
    print("\n--- Validating paths and setting up directories ---")
    for path_arg in [args.frames_dir, args.annotations_list, args.checkpoint]:
        if not os.path.exists(path_arg):
            raise FileNotFoundError(f"Required path not found: {path_arg}")

    for file_path_arg in [args.gt_output_csv, args.output_video]:
        output_dir = os.path.dirname(file_path_arg)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

    # Parse polygon coordinates for ROI filtering
    try:
        coords_list = list(map(int, args.polygon_coords.split(",")))
        polygon_points = np.array(coords_list, np.int32).reshape(-1, 1, 2)
    except ValueError as e:
        raise ValueError(f"Invalid polygon coordinates format: {e}. Expected comma-separated integers.")

    # --- 2. Initialize Models and Tracker ---
    print("\n--- Initializing DETR model and DeepSort tracker ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading DETR model from checkpoint: {args.checkpoint}")
    detr_model = Detr.load_from_checkpoint(args.checkpoint)
    detr_model = detr_model.to(device)
    detr_model.eval()  # Set model to evaluation mode
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    print("DETR model and processor loaded successfully.")

    print(f"Initializing DeepSort tracker with embedder: {args.embedder}")
    tracker_params = {
        "max_age": 6,
        "n_init": 2,
        "max_cosine_distance": 0.3,
        "nn_budget": None,
        "embedder": args.embedder,
        "half": True,  # Use half-precision for embedder if supported
        "bgr": True,  # DeepSort embedder expects BGR images
        "embedder_gpu": True  # Run embedder on GPU
    }
    if args.embedder_weights:
        tracker_params["embedder_wts"] = args.embedder_weights
        print(f"Using embedder weights from: {args.embedder_weights}")
        # Special check for CLIP, as it requires the 'clip' library
        if "clip" in args.embedder.lower():
            try:
                import clip
            except ImportError:
                print(
                    "Warning: CLIP embedder specified, but 'clip' library not found. Please install it (`pip install clip`).")

    deepsort_tracker = DeepSort(**tracker_params)
    print("DeepSort tracker initialized successfully.")

    # --- 3. Prepare for Data Processing ---
    print("\n--- Preparing frame list for processing ---")
    annotated_frames = load_annotated_images_list(args.annotations_list)
    # Construct full paths to frames and sort them for sequential processing
    frame_paths = [os.path.join(args.frames_dir, os.path.basename(f)) for f in annotated_frames]
    frame_paths.sort()

    print(f'Total Frames to Process: {len(frame_paths)}')
    if not frame_paths:
        print("No frames found to process. Exiting.")
        return

    print(f'First Frame: {os.path.basename(frame_paths[0])}')
    print(f'Last Frame: {os.path.basename(frame_paths[-1])}')

    # Initialize dictionaries/lists to hold tracking state and visualization data
    track_info = {}  # Stores detailed info for each track's crossing
    all_vis_data = []  # Stores frame data for generating the output video

    # Open the output CSV file for writing generated GT speeds
    # The file will be opened in write mode ('w') to ensure it's fresh for each run
    with open(args.gt_output_csv, mode='w', newline='') as gt_csv_f:
        gt_csv_writer = csv.writer(gt_csv_f)
        gt_csv_writer.writerow(GT_HEADERS)  # Write the CSV header row

        # --- 4. Process Each Frame ---
        print("\n--- Starting frame processing and GT speed generation ---")
        # Wrap the frame processing loop with tqdm for a command-line progress bar
        for frame_id, frame_path in enumerate(tqdm(frame_paths, desc="Generating GT Speeds")):
            process_frame_for_gt_speed(
                frame_path=frame_path,
                frame_id=frame_id,
                polygon_points=polygon_points,
                homography_matrix=np.eye(3),
                # Using identity matrix as pixel_to_realworld is not used for speed calculation here.
                detr_model=detr_model,
                detr_processor=detr_processor,
                deepsort_tracker=deepsort_tracker,
                track_info=track_info,
                gt_csv_writer=gt_csv_writer,
                all_vis_data_list=all_vis_data
            )

            # Get current counts of completed (and valid) crossings for console feedback
            current_north = sum(
                1 for t in track_info.values() if t.get('direction') == 'northbound' and t.get('speed') is not None)
            current_south = sum(
                1 for t in track_info.values() if t.get('direction') == 'southbound' and t.get('speed') is not None)

            # Update the latest frame in `all_vis_data` with directional counts overlay
            if all_vis_data and all_vis_data[-1].get("frame") is not None:
                # FIX: Pass the .start and .end components of FIRST_LINE and LAST_LINE
                all_vis_data[-1]["frame"] = visualize_directions(
                    all_vis_data[-1]["frame"],
                    (FIRST_LINE["start"], FIRST_LINE["end"]),  # Corrected format
                    (LAST_LINE["start"], LAST_LINE["end"]),  # Corrected format
                    current_north,
                    current_south
                )

            # Print periodic progress updates using tqdm.write to avoid interfering with the progress bar
            if (frame_id + 1) % args.progress_interval == 0 or frame_id == len(frame_paths) - 1:
                tqdm.write(f"Processed frame {frame_id + 1}/{len(frame_paths)}")
                tqdm.write(
                    f"Current generated GT speed counts - Northbound: {current_north}, Southbound: {current_south}")

    print(f"\nGenerated GT speed values saved to: {args.gt_output_csv}")

    # --- 5. Final Video Generation ---
    if args.output_video:
        print(f"\n--- Generating output video to: {args.output_video} ---")
        save_tracked_video(all_vis_data, args.output_video, fps=args.fps)
        print(f"Output video saved: {args.output_video}")
    else:
        print("Warning: Output video path not provided. Video will not be saved.")

    print("\n--- GT Speed Generation Complete ---")


if __name__ == "__main__":
    main()

