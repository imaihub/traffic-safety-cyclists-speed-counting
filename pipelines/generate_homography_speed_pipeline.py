import argparse
import csv
import os
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from transformers import DetrImageProcessor

from elements.constants import (  # Import general constants
    CONFIDENCE_THRESHOLD, FPS  # Using FPS from constants for consistency
)
# Import necessary components from the elements folder
from elements.models import Detr  # Assuming Detr class is defined here as a LightningModule
from elements.utils import (  # Using point_to_line_distance, save_tracked_video, visualize_directions, load_annotated_images_list
    point_to_line_distance, save_tracked_video, visualize_directions, load_annotated_images_list
)

# --- Constants for Speed Calculation Logic (Homography-based) ---
# These are the pixel coordinates that correspond to known real-world points for homography.
# You need to define these based on your specific camera setup and scene.
# Example: pts_src_ref are points in the image, pts_dst_ref are their real-world counterparts.
PTS_SRC_REF = np.array([
    [937.30, 390.20],  # Top-left image pixel
    [1100.30, 388.70],  # Top-right image pixel
    [1084.60, 506.20],  # Bottom-right image pixel
    [862.40, 504.00]  # Bottom-left image pixel
], dtype=np.float32)

PTS_DST_REF_METERS = np.array([
    [0, 0],  # Top-left (0m, 0m)
    [4, 0],  # Top-right (4m, 0m) - assuming 4 meters width
    [4, 10.5],  # Bottom-right (4m, 10.5m) - assuming 10.5 meters length
    [0, 10.5]  # Bottom-left (0m, 10.5m)
], dtype=np.float32)

# Scale factor for homography (e.g., 100 means 1 unit in homography output = 1 cm)
# When calculating real-world points, divide by this scale.
HOMOGRAPHY_SCALE = 100

# Line definitions for speed measurement in pixel coordinates (from your snippet)
FIRST_LINE_PIXELS = {
    "start": (832, 544),
    "end": (1085, 548)
}

LAST_LINE_PIXELS = {
    "start": (930, 387),
    "end": (1103, 387)
}

# General thresholds for tracking and speed
LINE_DETECTION_THRESHOLD = 15
MIN_CROSSING_FRAMES = 5
MAX_CROSSING_FRAMES = 2000
DEBUG = True
ENTRY_THRESHOLD = 15
EXIT_THRESHOLD = 15
MIN_SPEED = 3  # km/h
MAX_SPEED = 50  # km/h
MIN_PIXEL_DISTANCE = 50  # Minimum pixel distance for valid crossing

# CSV Headers for the generated speed predictions
PRED_SPEED_HEADERS = ["track_id", "speed_kph", "direction", "crossing_time_sec", "frames_taken", "entry_frame",
                      "exit_frame"]


# --- Helper Functions ---
def pixel_to_realworld(point_px: Tuple[float, float], H: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert pixel coordinates to real-world coordinates using a homography matrix.

    Args:
        point_px (tuple): (x, y) pixel coordinates.
        H (np.ndarray): 3x3 homography matrix.
        scale (float): Scale factor used when computing H (e.g., if H transforms to cm, divide by 100 for meters).

    Returns:
        np.ndarray: (x_real, y_real) in meters.
    """
    # Reshape point to (1, 1, 2) for cv2.perspectiveTransform
    point_px_reshaped = np.array([[point_px]], dtype=np.float32)
    point_real = cv2.perspectiveTransform(point_px_reshaped, H)[0][0] / scale
    return point_real


def calculate_real_distance(point1_px: Tuple[float, float], point2_px: Tuple[float, float], H: np.ndarray,
                            scale: float) -> float:
    """
    Calculate the real-world distance between two points given in pixel coordinates,
    using a homography matrix.

    Args:
        point1_px (tuple): (x, y) pixel coordinates of the first point.
        point2_px (tuple): (x, y) pixel coordinates of the second point.
        H (np.ndarray): 3x3 homography matrix.
        scale (float): Scale factor used in homography (e.g., 100 if H transforms to cm).

    Returns:
        float: Real-world Euclidean distance in meters.
    """
    world1 = pixel_to_realworld(point1_px, H, scale)
    world2 = pixel_to_realworld(point2_px, H, scale)
    distance = np.linalg.norm(world2 - world1)
    return distance


# --- Core Processing Function ---
def process_frame_for_homography_speed(
        frame_path: str,
        frame_id: int,
        polygon_points: np.ndarray,  # Polygon for ROI
        homography_matrix: np.ndarray,  # The 3x3 homography matrix
        homography_scale: float,  # The scale factor for homography (e.g., 100 for cm to meter)
        detr_model: Detr,  # Loaded DETR model
        detr_processor: DetrImageProcessor,  # DETR image processor
        deepsort_tracker: DeepSort,  # DeepSort tracker instance
        track_info: Dict,  # Dictionary to maintain state for each track
        pred_speed_csv_writer: Any,  # CSV writer object to write predicted speeds
        all_vis_data_list: List  # List to store data for video visualization
):
    """
    Processes a single video frame: performs object detection (DETR), tracking (DeepSort),
    calculates cyclist speeds using homography for real-world distance, and logs these.
    It also prepares data for video visualization.

    Args:
        frame_path (str): Full path to the current frame image.
        frame_id (int): Sequential index of the current frame.
        polygon_points (np.ndarray): Polygon defining the region of interest for detection.
        homography_matrix (np.ndarray): The 3x3 homography matrix.
        homography_scale (float): The scale factor used with the homography matrix.
        detr_model (Detr): Initialized DETR PyTorch Lightning model.
        detr_processor (DetrImageProcessor): Hugging Face DETR image processor.
        deepsort_tracker (DeepSort): Initialized DeepSort tracker.
        track_info (Dict): Dictionary storing persistent information for each track.
        pred_speed_csv_writer (Any): CSV writer object opened in 'append' mode to write speed data.
        all_vis_data_list (List): List to accumulate frame visualization data for video saving.
    """
    # 1. Load and prepare image
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)  # RGB format (for matplotlib, visualization)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # BGR format (for OpenCV, DeepSort)

    # 2. DETR Object Detection
    inputs = detr_processor(images=image, return_tensors="pt").to(detr_model.device)
    with torch.no_grad():
        outputs = detr_model.model(**inputs)

    # Post-process DETR detections
    results = detr_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]).to(detr_model.device),
        threshold=CONFIDENCE_THRESHOLD
    )[0]

    # 3. Format Detections for DeepSort
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 1:  # Only 'cyclist' class
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            center_x = x_min + width / 2
            center_y = y_min + height / 2

            if cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0:
                bbox_patch = image_bgr[int(y_min):int(y_max), int(x_min):int(x_max)]
                if bbox_patch.size == 0:
                    continue
                detection = ([x_min, y_min, width, height], float(score), bbox_patch)
                detections.append(detection)

    # 4. Update DeepSort Tracker
    tracks = deepsort_tracker.update_tracks(detections, frame=image_bgr)

    # Prepare data for frame visualization (a copy to draw on)
    frame_vis_data = {"frame": image_np.copy(), "tracks": []}

    # Draw the configured measurement lines on the visualization frame
    cv2.line(frame_vis_data["frame"], FIRST_LINE_PIXELS["start"], FIRST_LINE_PIXELS["end"], (255, 0, 0), 3)  # Blue
    cv2.line(frame_vis_data["frame"], LAST_LINE_PIXELS["start"], LAST_LINE_PIXELS["end"], (255, 0, 0), 3)  # Blue

    # 5. Process Tracks for Speed Estimation and Logging
    for track in tracks:
        if not track.is_confirmed():
            continue

        ltrb = track.to_ltrb()
        x_min, y_min, x_max, y_max = ltrb
        width = x_max - x_min
        height = y_max - y_min
        center = (x_min + width / 2, y_min + height / 2)
        tracking_point = center
        confidence = getattr(track, 'det_conf', 1.0)

        if track.track_id not in track_info:
            track_info[track.track_id] = {
                'positions': [],
                'entry_frame': None,
                'exit_frame': None,
                'entry_line': None,
                'speed': None,
                'direction': None,
                'in_measurement': False,
                'last_position': None
            }

        track_info[track.track_id]['positions'].append(tracking_point)
        if len(track_info[track.track_id]['positions']) > 50:
            track_info[track.track_id]['positions'].pop(0)

        # Check for entry into the measurement zone
        if not track_info[track.track_id]['in_measurement']:
            first_line_dist = point_to_line_distance(tracking_point, FIRST_LINE_PIXELS)
            last_line_dist = point_to_line_distance(tracking_point, LAST_LINE_PIXELS)

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

            if track_info[track.track_id]['entry_line'] == 'FIRST':
                target_line = LAST_LINE_PIXELS
                direction_str = 'southbound'
            else:
                target_line = FIRST_LINE_PIXELS
                direction_str = 'northbound'

            current_dist = point_to_line_distance(tracking_point, target_line)
            prev_pos = track_info[track.track_id]['last_position']
            prev_dist = point_to_line_distance(prev_pos, target_line) if prev_pos else float('inf')

            if (current_dist < EXIT_THRESHOLD) or (current_dist < 2 * EXIT_THRESHOLD and current_dist < prev_dist):
                exit_detected = True

                entry_pos = track_info[track.track_id]['entry_position']
                distance_traveled_pixels = np.linalg.norm(np.array(tracking_point) - np.array(entry_pos))

                if distance_traveled_pixels < MIN_PIXEL_DISTANCE:
                    if DEBUG:
                        tqdm.write(
                            f"  [Frame {frame_id}] ID {track.track_id}: Insufficient pixel distance traveled ({distance_traveled_pixels:.1f}px). Exit aborted.")
                    exit_detected = False

            if exit_detected:
                frames_taken = frame_id - track_info[track.track_id]['entry_frame']

                if MIN_CROSSING_FRAMES <= frames_taken <= MAX_CROSSING_FRAMES:
                    time_sec = frames_taken / FPS

                    # Calculate real-world distance using homography
                    distance_traveled_real = calculate_real_distance(
                        track_info[track.track_id]['entry_position'],
                        tracking_point,
                        homography_matrix,
                        homography_scale
                    )

                    speed_kph = (distance_traveled_real / time_sec) * 3.6

                    if MIN_SPEED <= speed_kph <= MAX_SPEED:
                        track_info[track.track_id].update({
                            'exit_frame': frame_id,
                            'speed': speed_kph,
                            'direction': direction_str,
                            'crossing_time': time_sec
                        })
                        # Write the predicted speed entry to CSV
                        pred_speed_csv_writer.writerow([
                            track.track_id,
                            f"{speed_kph:.1f}",
                            direction_str,
                            f"{time_sec:.2f}",
                            frames_taken,
                            track_info[track.track_id]['entry_frame'],
                            frame_id
                        ])
                        if DEBUG:
                            tqdm.write(
                                f"  [Frame {frame_id}] SPEED PRED: ID {track.track_id} - {speed_kph:.1f} km/h ({direction_str}) "
                                f"({time_sec:.2f}s, {frames_taken} frames) Real distance: {distance_traveled_real:.2f}m")
                    else:
                        if DEBUG:
                            tqdm.write(
                                f"  [Frame {frame_id}] INVALID SPEED PRED: {speed_kph:.1f} km/h (discarded due to range).")
                else:
                    if DEBUG:
                        tqdm.write(
                            f"  [Frame {frame_id}] INVALID CROSSING TIME PRED: {frames_taken} frames (discarded due to range).")

        # Update last known position
        track_info[track.track_id]['last_position'] = tracking_point

        # Add track info to frame_vis_data for video output
        frame_vis_data["tracks"].append({
            "track_id": track.track_id,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": confidence
        })

    # Add the current frame's visualization data to the overall list
    all_vis_data_list.append(frame_vis_data)


def main():
    """
    Main function to run the process of generating speed predictions
    by applying a DETR + DeepSort pipeline and homography for distance calculation.
    """
    parser = argparse.ArgumentParser(
        description="Generate Homography-based Speed Predictions from Tracking Pipeline",
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
    parser.add_argument('--pred-output-csv', default="results/predicted_homography_speeds.csv",
                        help='Path to save the generated predicted speed CSV file')
    parser.add_argument('--output-video', default="results/homography_speed_visualization_video.mp4",
                        help='Path to save output video with visualizations of tracking and speed calculation.')

    # Homography reference points (pixel coords)
    parser.add_argument('--homography-src-ref', type=str,
                        default="937.30,390.20,1100.30,388.70,1084.60,506.20,862.40,504.00",
                        help='Comma-separated pixel coordinates for homography source reference points (x1,y1,x2,y2,x3,y3,x4,y4).')
    # Homography reference points (real-world coords in meters)
    parser.add_argument('--homography-dst-ref', type=str,
                        default="0,0,4,0,4,10.5,0,10.5",
                        help='Comma-separated real-world coordinates (in meters) for homography destination reference points (x1,y1,x2,y2,x3,y3,x4,y4).')

    # Polygon for ROI
    parser.add_argument('--polygon-coords', default="937,390,1100,388,1003,1080,469,1079",
                        help='Comma-separated polygon coordinates (x1,y1,x2,y2,...) defining the region of interest for detection and tracking.')

    # DeepSort embedder configuration
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

    for file_path_arg in [args.pred_output_csv, args.output_video]:
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

    # Parse homography reference points
    try:
        src_ref_coords = list(map(float, args.homography_src_ref.split(",")))
        pts_src_ref = np.array(src_ref_coords, dtype=np.float32).reshape(-1, 2)

        dst_ref_coords = list(map(float, args.homography_dst_ref.split(",")))
        # Scale destination points for homography calculation (cm for better numerical stability)
        pts_dst_ref = np.array(dst_ref_coords, dtype=np.float32).reshape(-1, 2) * HOMOGRAPHY_SCALE

        if pts_src_ref.shape[0] != 4 or pts_dst_ref.shape[0] != 4:
            raise ValueError("Homography reference points must consist of exactly 4 pairs (x,y).")

    except ValueError as e:
        raise ValueError(
            f"Invalid homography reference points format: {e}. Expected comma-separated floats for 4 points.")

    # Calculate homography matrix
    if pts_src_ref.shape[0] >= 4 and pts_dst_ref.shape[0] >= 4:
        H, _ = cv2.findHomography(pts_src_ref, pts_dst_ref, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Could not compute homography matrix. Check reference points.")
        print("\nHomography matrix calculated successfully.")
        if DEBUG:
            print(f"Homography Matrix H:\n{H}")
    else:
        raise ValueError("Need at least 4 corresponding points to compute homography.")

    # --- 2. Initialize Models and Tracker ---
    print("\n--- Initializing DETR model and DeepSort tracker ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading DETR model from checkpoint: {args.checkpoint}")
    detr_model = Detr.load_from_checkpoint(args.checkpoint)
    detr_model = detr_model.to(device)
    detr_model.eval()
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    print("DETR model and processor loaded successfully.")

    print(f"Initializing DeepSort tracker with embedder: {args.embedder}")
    tracker_params = {
        "max_age": 6, "n_init": 2, "max_cosine_distance": 0.3, "nn_budget": None,
        "embedder": args.embedder, "half": True, "bgr": True, "embedder_gpu": True
    }
    if args.embedder_weights:
        tracker_params["embedder_wts"] = args.embedder_weights
        print(f"Using embedder weights from: {args.embedder_weights}")
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
    frame_paths = [os.path.join(args.frames_dir, os.path.basename(f)) for f in annotated_frames]
    frame_paths.sort()

    print(f'Total Frames to Process: {len(frame_paths)}')
    if not frame_paths:
        print("No frames found to process. Exiting.")
        return

    print(f'First Frame: {os.path.basename(frame_paths[0])}')
    print(f'Last Frame: {os.path.basename(frame_paths[-1])}')

    track_info = {}
    all_vis_data = []

    # Open the output CSV file for writing predicted speeds
    with open(args.pred_output_csv, mode='w', newline='') as pred_csv_f:
        pred_csv_writer = csv.writer(pred_csv_f)
        pred_csv_writer.writerow(PRED_SPEED_HEADERS)

        # --- 4. Process Each Frame ---
        print("\n--- Starting frame processing and homography-based speed prediction ---")
        for frame_id, frame_path in enumerate(tqdm(frame_paths, desc="Generating Speed Predictions")):
            process_frame_for_homography_speed(
                frame_path=frame_path,
                frame_id=frame_id,
                polygon_points=polygon_points,
                homography_matrix=H,  # Pass the computed homography matrix
                homography_scale=HOMOGRAPHY_SCALE,  # Pass the scale
                detr_model=detr_model,
                detr_processor=detr_processor,
                deepsort_tracker=deepsort_tracker,
                track_info=track_info,
                pred_speed_csv_writer=pred_csv_writer,
                all_vis_data_list=all_vis_data
            )

            # Get current counts of completed (and valid) crossings for console feedback
            current_north = sum(
                1 for t in track_info.values() if t.get('direction') == 'northbound' and t.get('speed') is not None)
            current_south = sum(
                1 for t in track_info.values() if t.get('direction') == 'southbound' and t.get('speed') is not None)

            # Update the latest frame in `all_vis_data` with directional counts overlay
            if all_vis_data and all_vis_data[-1].get("frame") is not None:
                all_vis_data[-1]["frame"] = visualize_directions(
                    all_vis_data[-1]["frame"],
                    (FIRST_LINE_PIXELS["start"], FIRST_LINE_PIXELS["end"]),
                    (LAST_LINE_PIXELS["start"], LAST_LINE_PIXELS["end"]),
                    current_north,
                    current_south
                )

            if (frame_id + 1) % args.progress_interval == 0 or frame_id == len(frame_paths) - 1:
                tqdm.write(f"Processed frame {frame_id + 1}/{len(frame_paths)}")
                tqdm.write(f"Current predicted speed counts - Northbound: {current_north}, Southbound: {current_south}")

    print(f"\nGenerated predicted speed values saved to: {args.pred_output_csv}")

    # --- 5. Final Video Generation ---
    if args.output_video:
        print(f"\n--- Generating output video to: {args.output_video} ---")
        save_tracked_video(all_vis_data, args.output_video, fps=args.fps)
        print(f"Output video saved: {args.output_video}")
    else:
        print("Warning: Output video path not provided. Video will not be saved.")

    print("\n--- Homography-based Speed Prediction Complete ---")


if __name__ == "__main__":
    main()

