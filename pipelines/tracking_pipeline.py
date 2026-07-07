import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm  # Import tqdm for progress bars
from transformers import DetrImageProcessor

from elements.constants import (
    CONFIDENCE_THRESHOLD, FPS, MIN_CROSSING_FRAMES,
    MAX_CROSSING_FRAMES, ENTRY_THRESHOLD, EXIT_THRESHOLD, MIN_SPEED, MAX_SPEED,
    MIN_PIXEL_DISTANCE, HOMOGRAPHY_SCALE, PRED_CSV_HEADERS, DEBUG, FIRST_LINE, LAST_LINE
)
# Import necessary components from the elements folder
from elements.models import Detr
from elements.utils import (
    pixel_to_realworld, point_to_line_distance, get_crossing_lines,
    save_tracked_video, visualize_directions, load_annotated_images_list
)


# Note: PRED_CSV_FILE and OUTPUT_VIDEO are removed from direct import here
# as they will be passed to __init__ instead.

class TrackingPipeline:
    """
    Orchestrates the entire object detection, tracking, counting, and speed estimation pipeline.
    It loads models, initializes trackers, and processes video frames.
    """

    def __init__(self, checkpoint_path, polygon_coords, homography_matrix, homography_distance,
                 embedder_type: str = "torchreid", embedder_weights_path: str = None,
                 annotations_list_path: str = None, output_file_path: str = None,
                 output_video_path: str = None, pred_csv_file_path: str = None):  # Added new arguments
        """
        Initializes the tracking pipeline with models, tracker, and configuration.

        Args:
            checkpoint_path (str): Path to the DETR model checkpoint.
            polygon_coords (numpy.ndarray): The polygon defining the area of interest in pixel coordinates.
            homography_matrix (numpy.ndarray): The 3x3 homography matrix for pixel-to-real-world conversion.
            homography_distance (float): The pre-calculated real-world distance between entry/exit lines.
            embedder_type (str): Type of embedder for DeepSort (e.g., "torchreid", "mobilenet", "clip_ViT-B/32").
            embedder_weights_path (str, optional): Path to pre-trained weights for the embedder.
                                                  Required for certain embedders like CLIP.
            annotations_list_path (str, optional): Path to text file with list of annotated images to process.
            output_file_path (str, optional): Path for tracking results output file (MOT format).
            output_video_path (str, optional): Path for the output video file with visualizations.
            pred_csv_file_path (str, optional): Path for the predicted speeds CSV file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store paths provided by run_inference.py
        self.annotations_list_path = annotations_list_path
        self.output_file_path = output_file_path
        self.output_video_path = output_video_path  # Stored
        self.pred_csv_file_path = pred_csv_file_path  # Stored

        # 1. Load DETR model
        print(f"Loading DETR model from checkpoint: {checkpoint_path}")
        self.model = Detr.load_from_checkpoint(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        print("DETR model loaded.")

        # 2. Initialize DeepSort tracker with dynamic embedder
        print(f"Initializing DeepSort tracker with embedder: {embedder_type}")
        tracker_params = {
            "max_age": 6,
            "n_init": 2,
            "max_cosine_distance": 0.3,
            "nn_budget": None,
            "embedder": embedder_type,
            "half": True,
            "bgr": True,
            "embedder_gpu": True
        }

        # Add embedder weights path if provided and applicable
        if embedder_weights_path:
            tracker_params["embedder_wts"] = embedder_weights_path
            print(f"Using embedder weights from: {embedder_weights_path}")
            # Special handling for CLIP: DeepSort's embedder might need clip library imported.
            if "clip" in embedder_type.lower():
                try:
                    import clip  # Attempt to import clip if using CLIP embedder
                    print("CLIP library imported successfully for embedder.")
                except ImportError:
                    print(
                        "Warning: CLIP embedder specified, but 'clip' library not found. Please install it (`pip install clip`).")

        self.tracker = DeepSort(**tracker_params)
        print("DeepSort tracker initialized.")
        if hasattr(self.tracker.embedder, 'model'):
            print(f"DeepSort embedder model type: {type(self.tracker.embedder.model)}")

        self.polygon_points = polygon_coords
        self.homography_matrix = homography_matrix
        self.homography_distance = homography_distance

        # Define boundary lines for counting using the provided polygon
        crossing_lines = get_crossing_lines(self.polygon_points)
        self.top_line = crossing_lines['top_line']
        self.bottom_line = crossing_lines['bottom_line']
        print(f"Auto-detected top line for counting: {self.top_line}")
        print(f"Auto-detected bottom line for counting: {self.bottom_line}")

        # Initialize tracking variables for the pipeline run
        self.track_info = {}  # Stores detailed info for each track (positions, entry/exit frames, speed)
        self.completed_tracks = set()  # Stores IDs of tracks that have completed a crossing
        self.active_tracks = {}  # Maps DeepSort track_id to a simplified 'id_counter' for output
        self.id_counter = 1  # Simple counter for assigning unique IDs to new tracks

        # Initialize predictions CSV file with headers, using the path from __init__
        if self.pred_csv_file_path:
            with open(self.pred_csv_file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(PRED_CSV_HEADERS)
            print(f"Initialized prediction CSV: {self.pred_csv_file_path}")
        else:
            print("Warning: pred_csv_file_path not provided. Speed predictions will not be saved to CSV.")

    def process_frame(self, frame_path: str, frame_id: int):
        """
        Processes a single video frame: performs detection, tracking,
        and updates counting and speed estimation.

        Args:
            frame_path (str): Full path to the current frame image.
            frame_id (int): Sequential index of the current frame.

        Returns:
            tuple:
                - mot_predictions (list): List of detected and tracked objects in MOT format.
                - frame_vis_data (dict): Data for visualizing the frame (image_np, tracks).
        """
        # Removed the DEBUG print for 'Processing frame {frame_id}' here
        # as tqdm will show overall progress and we'll use tqdm.write for granular info.

        # 1. Load and prepare image
        image = Image.open(frame_path).convert("RGB")
        image_np = np.array(image)  # RGB format
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # BGR for OpenCV operations and DeepSort embedder
        frame_name = 'test/' + os.path.basename(frame_path)  # Consistent naming for MOT output

        # 2. DETR Detection
        # Keeping DEBUG prints inside process_frame for more detailed per-frame info if DEBUG is True
        if DEBUG:
            tqdm.write(f"  [Frame {frame_id}] Performing DETR detection...")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.model(**inputs)

        # 3. Post-process DETR detections
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]).to(self.device),
            threshold=CONFIDENCE_THRESHOLD
        )[0]
        if DEBUG:
            tqdm.write(f"  [Frame {frame_id}] Detected {len(results['boxes'])} objects.")

        # 4. Format detections for DeepSort
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 1:  # Only process 'cyclist' class (label ID 1)
                x_min, y_min, x_max, y_max = box.tolist()
                width = x_max - x_min
                height = y_max - y_min
                center_x = x_min + width / 2
                center_y = y_min + height / 2

                # Only include detections whose center is within the defined polygon area
                if cv2.pointPolygonTest(self.polygon_points, (center_x, center_y), False) >= 0:
                    bbox_patch = image_bgr[int(y_min):int(y_max), int(x_min):int(x_max)]
                    if bbox_patch.size == 0:  # Skip empty patches
                        continue

                    # DeepSort expects [x, y, width, height] for bbox
                    detection = ([x_min, y_min, width, height], float(score), bbox_patch)
                    detections.append(detection)
        if DEBUG:
            tqdm.write(f"  [Frame {frame_id}] {len(detections)} detections within polygon for DeepSort.")

        # 5. Update DeepSort tracker
        if DEBUG:
            tqdm.write(f"  [Frame {frame_id}] Updating DeepSort tracker...")
        tracks = self.tracker.update_tracks(detections, frame=image_bgr)  # Pass BGR frame to embedder
        if DEBUG:
            tqdm.write(f"  [Frame {frame_id}] DeepSort returned {len(tracks)} tracks.")

        # 6. Process tracking results for counting and speed estimation
        mot_predictions = []
        frame_vis_data = {"frame": image_np.copy(), "tracks": []}  # Use a copy to draw on for visualization

        # Draw the measurement lines on the frame
        cv2.line(frame_vis_data["frame"], FIRST_LINE["start"], FIRST_LINE["end"], (255, 0, 0), 3)  # Blue
        cv2.line(frame_vis_data["frame"], LAST_LINE["start"], LAST_LINE["end"], (255, 0, 0), 3)  # Blue

        for track in tracks:
            # Only consider confirmed tracks (seen enough frames to be stable)
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()  # Bounding box in (left, top, right, bottom) format
            x_min, y_min, x_max, y_max = ltrb
            width = x_max - x_min
            height = y_max - y_min
            center = (x_min + width / 2, y_min + height / 2)  # Center of the bounding box
            tracking_point = center  # Using center for line crossing checks
            confidence = getattr(track, 'det_conf', 1.0)  # Get detection confidence, default to 1.0 if not available

            # Debug visualization for tracking points
            if DEBUG:
                cv2.circle(frame_vis_data["frame"], (int(tracking_point[0]), int(tracking_point[1])), 5, (0, 255, 255),
                           -1)
                cv2.putText(frame_vis_data["frame"], f"ID:{track.track_id}",
                            (int(tracking_point[0]), int(tracking_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Initialize track_info for new tracks
            if track.track_id not in self.track_info:
                self.track_info[track.track_id] = {
                    'positions': [],  # History of tracking points
                    'entry_frame': None,
                    'exit_frame': None,
                    'entry_line': None,  # Which line was crossed first ('FIRST' or 'LAST')
                    'speed': None,
                    'direction': None,  # 'northbound' or 'southbound'
                    'in_measurement': False,  # Flag if currently in the speed measurement zone
                    'last_position': None  # Last known position for movement vector calculation
                }

            # Store position history for movement direction analysis
            self.track_info[track.track_id]['positions'].append(tracking_point)
            if len(self.track_info[track.track_id]['positions']) > 50:  # Keep history limited to recent frames
                self.track_info[track.track_id]['positions'].pop(0)

            # --- Speed Measurement Logic ---
            if not self.track_info[track.track_id]['in_measurement']:
                # Check for entry into the measurement zone
                first_line_dist = point_to_line_distance(tracking_point, FIRST_LINE)
                last_line_dist = point_to_line_distance(tracking_point, LAST_LINE)

                if DEBUG:
                    tqdm.write(
                        f"  [Frame {frame_id}] ID {track.track_id} - Dist to FIRST: {first_line_dist:.1f}, LAST: {last_line_dist:.1f}")

                if first_line_dist < ENTRY_THRESHOLD:
                    self.track_info[track.track_id].update({
                        'entry_line': 'FIRST',
                        'entry_frame': frame_id,
                        'entry_position': tracking_point,
                        'in_measurement': True,
                        'last_position': tracking_point
                    })
                    if DEBUG:
                        tqdm.write(f"  [Frame {frame_id}] ENTRY: ID {track.track_id} at FIRST line (frame {frame_id})")
                elif last_line_dist < ENTRY_THRESHOLD:
                    self.track_info[track.track_id].update({
                        'entry_line': 'LAST',
                        'entry_frame': frame_id,
                        'entry_position': tracking_point,
                        'in_measurement': True,
                        'last_position': tracking_point
                    })
                    if DEBUG:
                        tqdm.write(f"  [Frame {frame_id}] ENTRY: ID {track.track_id} at LAST line (frame {frame_id})")

            # If track is in measurement, check for exit
            elif self.track_info[track.track_id]['in_measurement'] and not self.track_info[track.track_id][
                'exit_frame']:
                exit_detected = False
                direction = None
                target_line = None

                # Determine the target exit line and expected direction
                if self.track_info[track.track_id]['entry_line'] == 'FIRST':
                    target_line = LAST_LINE
                    direction = 'southbound'  # Moving from higher Y (bottom) to lower Y (top)
                else:  # Entry from LAST_LINE
                    target_line = FIRST_LINE
                    direction = 'northbound'  # Moving from lower Y (top) to higher Y (bottom)

                # Check current distance to target line
                current_dist = point_to_line_distance(tracking_point, target_line)
                prev_pos = self.track_info[track.track_id]['last_position']
                prev_dist = point_to_line_distance(prev_pos, target_line) if prev_pos else float('inf')

                # Exit condition: crossing the line OR moving toward the line and very close
                if (current_dist < EXIT_THRESHOLD) or (current_dist < 2 * EXIT_THRESHOLD and current_dist < prev_dist):
                    exit_detected = True

                    # Verify minimum distance traveled in pixels to avoid false positives (e.g., oscillating at entry line)
                    entry_pos = self.track_info[track.track_id]['entry_position']
                    distance_traveled_pixels = np.linalg.norm(np.array(tracking_point) - np.array(entry_pos))

                    if distance_traveled_pixels < MIN_PIXEL_DISTANCE:
                        if DEBUG:
                            tqdm.write(
                                f"  [Frame {frame_id}] ID {track.track_id}: Insufficient pixel distance traveled ({distance_traveled_pixels:.1f}px)")
                        exit_detected = False

                if exit_detected:
                    frames_taken = frame_id - self.track_info[track.track_id]['entry_frame']

                    # Validate the duration of the crossing
                    if MIN_CROSSING_FRAMES <= frames_taken <= MAX_CROSSING_FRAMES:
                        time_sec = frames_taken / FPS

                        # Calculate real-world distance traveled using homography
                        entry_pos_real = pixel_to_realworld(self.track_info[track.track_id]['entry_position'],
                                                            self.homography_matrix, HOMOGRAPHY_SCALE)
                        exit_pos_real = pixel_to_realworld(tracking_point, self.homography_matrix, HOMOGRAPHY_SCALE)
                        distance_traveled_real = np.linalg.norm(exit_pos_real - entry_pos_real)

                        # Calculate speed in km/h
                        speed_kph = (distance_traveled_real / time_sec) * 3.6

                        # Validate speed within reasonable bounds
                        if MIN_SPEED <= speed_kph <= MAX_SPEED:
                            self.track_info[track.track_id].update({
                                'exit_frame': frame_id,
                                'speed': speed_kph,
                                'direction': direction,
                                'crossing_time': time_sec
                            })
                            self.completed_tracks.add(track.track_id)  # Mark as completed

                            # Write to CSV, using self.pred_csv_file_path
                            if self.pred_csv_file_path:
                                with open(self.pred_csv_file_path, mode='a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        track.track_id,
                                        f"{speed_kph:.1f}",  # Format speed for CSV
                                        direction,
                                        f"{time_sec:.2f}",  # Format time
                                        frames_taken,
                                        self.track_info[track.track_id]['entry_frame'],
                                        frame_id  # exit_frame
                                    ])
                            else:
                                if DEBUG:
                                    tqdm.write(
                                        f"  [Frame {frame_id}] Warning: pred_csv_file_path not set, skipping CSV write for ID {track.track_id}")

                            if DEBUG:
                                tqdm.write(
                                    f"  [Frame {frame_id}] SPEED: ID {track.track_id} - {speed_kph:.1f} km/h ({direction}) "
                                    f"Real-world distance: {distance_traveled_real:.2f}m")
                                if self.homography_distance:
                                    tqdm.write(
                                        f"               Fixed homography distance: {self.homography_distance:.2f}m")

            # Update last known position for movement vector calculation in next frame
            self.track_info[track.track_id]['last_position'] = tracking_point

            # Visual feedback for tracks currently in measurement zone
            if self.track_info[track.track_id]['in_measurement'] and not self.track_info[track.track_id]['exit_frame']:
                # Calculate progress based on entry line and current Y position
                total_y_dist = abs(FIRST_LINE["start"][1] - LAST_LINE["start"][1])
                if total_y_dist > 0:
                    if self.track_info[track.track_id]['entry_line'] == 'FIRST':
                        # Moving from FIRST (higher Y) to LAST (lower Y)
                        progress = min(1.0, max(0.0, (tracking_point[1] - LAST_LINE["start"][1]) / total_y_dist))
                        direction_indicator = "↓"  # Visually indicates movement down (southbound)
                    else:  # Entry from LAST_LINE, moving to FIRST_LINE
                        # Moving from LAST (lower Y) to FIRST (higher Y)
                        progress = min(1.0, max(0.0, (FIRST_LINE["start"][1] - tracking_point[1]) / total_y_dist))
                        direction_indicator = "↑"  # Visually indicates movement up (northbound)
                    cv2.putText(frame_vis_data["frame"], f"{progress * 100:.0f}% {direction_indicator}",
                                (int(center[0]), int(center[1]) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Update active tracks and MOT predictions
            inside_polygon = cv2.pointPolygonTest(self.polygon_points, center, False) >= 0
            if inside_polygon:
                if track.track_id not in self.active_tracks:
                    self.active_tracks[track.track_id] = {
                        'id': self.id_counter,
                        'counted': False  # This 'counted' flag seems redundant with self.completed_tracks
                    }
                    self.id_counter += 1

                mot_predictions.append([
                    frame_name,
                    str(self.active_tracks[track.track_id]['id']),  # Use the simplified numeric ID
                    f"{x_min:.2f}",  # Format coordinates for output consistency
                    f"{y_min:.2f}",
                    f"{width:.2f}",
                    f"{height:.2f}",
                    f"{confidence:.2f}",
                    1,  # category_id (cyclist)
                    1  # visibility
                ])

                frame_vis_data["tracks"].append({
                    "track_id": self.active_tracks[track.track_id]['id'],
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": confidence
                })

        # Draw the tracking polygon on the frame
        cv2.polylines(frame_vis_data["frame"], [self.polygon_points], True, (255, 0, 0), 2)  # Red polygon

        return mot_predictions, frame_vis_data

    def run(self, frames_dir, output_video_path, progress_interval):
        """
        Runs the full tracking and counting pipeline over a sequence of frames.

        Args:
            frames_dir (str): Directory containing frame images.
            output_video_path (str): Path for the output tracked video file.
            progress_interval (int): Interval to print progress updates.
        """
        print(f"Starting tracking pipeline for frames in: {frames_dir}")

        # Use the annotations_list_path stored in the instance
        if not self.annotations_list_path:
            raise ValueError("annotations_list_path must be provided to TrackingPipeline during initialization.")

        annotated_frames = load_annotated_images_list(self.annotations_list_path)

        frame_paths = [
            os.path.join(frames_dir, os.path.basename(f))
            for f in annotated_frames
            if os.path.basename(f) in os.listdir(frames_dir)  # Ensure the file actually exists
        ]
        frame_paths.sort()  # Ensure frames are processed in order

        print(f'Total Frames Found: {len(frame_paths)}')
        if not frame_paths:
            print("No frames found to process. Exiting.")
            return

        print(f'First Frame: {os.path.basename(frame_paths[0])}')
        print(f'Last Frame: {os.path.basename(frame_paths[-1])}')

        all_mot_predictions = []
        all_vis_data = []

        # Wrap the frame processing loop with tqdm for a progress bar
        for frame_id, frame_path in enumerate(tqdm(frame_paths, desc="Processing Frames")):
            mot_preds, frame_data = self.process_frame(frame_path, frame_id)
            all_mot_predictions.extend(mot_preds)
            all_vis_data.append(frame_data)

            # Get current counts for display
            current_north = sum(1 for t in self.track_info.values() if t.get('direction') == 'northbound')
            current_south = sum(1 for t in self.track_info.values() if t.get('direction') == 'southbound')

            # Update visualization with current directional counts
            frame_data["frame"] = visualize_directions(
                frame_data["frame"],
                self.top_line,
                self.bottom_line,
                current_north,
                current_south
            )

            # Print progress (using tqdm.write to avoid interfering with the progress bar)
            if frame_id % progress_interval == 0 or frame_id == len(frame_paths) - 1:
                tqdm.write(f"Processed frame {frame_id + 1}/{len(frame_paths)}")
                tqdm.write(f"Current counts - Northbound: {current_north}, Southbound: {current_south}")

        # Final summary
        final_northbound = sum(1 for t in self.track_info.values() if t.get('direction') == 'northbound')
        final_southbound = sum(1 for t in self.track_info.values() if t.get('direction') == 'southbound')
        total_counted = final_northbound + final_southbound

        print("\nFinal Directional Crossing Counts:")
        print(f"Northbound cyclists: {final_northbound}")
        print(f"Southbound cyclists: {final_southbound}")
        print(f"Total counted: {total_counted}")

        # Save output video, using self.output_video_path
        if self.output_video_path:
            save_tracked_video(all_vis_data, self.output_video_path, fps=FPS)
        else:
            print("Warning: output_video_path not provided. Output video will not be saved.")

        # Save MOT predictions to a text file, using self.output_file_path
        if not self.output_file_path:
            print("Warning: output_file_path not provided. Tracking predictions will not be saved.")
        else:
            with open(self.output_file_path, "w") as f:
                for pred in all_mot_predictions:
                    f.write(",".join(map(str, pred)) + "\n")
            print(f"\nTracking predictions saved to: {self.output_file_path}")
