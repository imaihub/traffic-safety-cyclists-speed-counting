import argparse
import os

import cv2  # For homography calculations
import numpy as np

from elements.constants import (
    FPS, PRED_CSV_FILE, OUTPUT_VIDEO, OUTPUT_FILE,  # Still imported for argparse defaults
    FRAMES_DIR, ANNOTATIONS_TEST, ANNOTATIONS_LIST_FILE, CHECKPOINT_PATH,
    pts_dst_ref, pts_src_ref, HOMOGRAPHY_SCALE
)
from elements.utils import pixel_to_realworld  # Import here for homography calculation print
# Import components from pipelines and elements
from pipelines.tracking_pipeline import TrackingPipeline


def parse_arguments():
    """
    Parses command-line arguments for the cyclist tracking and counting script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Cyclist Tracking, Counting, and Speed Estimation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required paths (with default values for convenience, but can be overridden)
    parser.add_argument('--frames-dir', default=FRAMES_DIR,
                        help='Directory containing frame images')
    parser.add_argument('--annotations-test', default=ANNOTATIONS_TEST,
                        help='Path to test annotations JSON file (currently not used by main logic)')
    parser.add_argument('--annotations-list', default=ANNOTATIONS_LIST_FILE,
                        help='Path to text file with list of annotated images to process')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH,
                        help='Path to DETR model checkpoint file')

    # Optional output paths (with defaults)
    parser.add_argument('--output-video', default=OUTPUT_VIDEO, # Default from constants
                        help='Path for output video file with visualizations')
    parser.add_argument('--output-file', default=OUTPUT_FILE,   # Default from constants
                        help='Path for tracking results output file (MOT format)')
    parser.add_argument('--pred-csv', default=PRED_CSV_FILE,    # Default from constants
                        help='Path for predicted speeds CSV file')

    # DeepSort embedder configuration
    parser.add_argument('--embedder', default="torchreid",
                        help='DeepSort embedder type (e.g., "torchreid", "mobilenet", "clip_ViT-B/32"). '
                             'Corresponds to the `embedder` argument in DeepSort constructor.')
    parser.add_argument('--embedder-weights', default=None,
                        help='Path to pre-trained weights for the embedder (e.g., for CLIP models).')


    # Processing parameters
    parser.add_argument('--coords', default="937,390,1100,388,1003,1080,469,1079",
                        help='Comma-separated polygon coordinates (x1,y1,x2,y2,...) '
                             'defining the region of interest for detection and tracking. '
                             'Default corresponds to pts_track_pixels from constants.')
    parser.add_argument('--fps', type=int, default=FPS,
                        help='Frames per second for output video and speed calculations')
    parser.add_argument('--progress-interval', type=int, default=50,
                        help='Print progress update every N frames')

    return parser.parse_args()


def validate_paths(args):
    """
    Validates that all required input paths exist and creates output directories if necessary.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        FileNotFoundError: If any required input file/directory does not exist.
        ValueError: If polygon coordinates are invalid.
    """
    required_inputs = [
        ('frames_dir', args.frames_dir),
        ('annotations_list', args.annotations_list),
        ('checkpoint', args.checkpoint),
    ]

    # If embedder weights are specified, validate that path too
    if args.embedder_weights:
        required_inputs.append(('embedder_weights', args.embedder_weights))

    for name, path in required_inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Required {name} path does not exist: {path}")

    # Check and create output directories
    # Ensure parent directories for output files exist
    output_files = [
        ('output_video', args.output_video),
        ('output_file', args.output_file),
        ('pred_csv', args.pred_csv),
    ]

    for name, path in output_files:
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            print(f"Created directory for {name}: {dirname}")


def main():
    """
    Main function to run the cyclist tracking and counting application.
    It parses arguments, validates paths, calculates homography, and
    initializes/runs the TrackingPipeline.
    """
    args = parse_arguments()

    try:
        validate_paths(args)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        exit(1)

    # Parse polygon coordinates from arguments
    try:
        coords_list = list(map(int, args.coords.split(",")))
        # Reshape to (N, 1, 2) for OpenCV functions like pointPolygonTest
        polygon_points = np.array(coords_list, np.int32).reshape(-1, 1, 2)
    except ValueError as e:
        print(f"Error: Invalid coordinates format for --coords. Expected comma-separated integers. Error: {e}")
        exit(1)

    # --- Homography Calculation (moved from constants for runtime calculation) ---
    # Calculate Homography Matrix
    # We use pts_src_ref and pts_dst_ref from elements.constants to calculate H
    # The real-world coordinates (pts_dst_ref) are scaled by HOMOGRAPHY_SCALE
    # during H calculation, so pixel_to_realworld needs to divide by HOMOGRAPHY_SCALE later.
    try:
        H = cv2.getPerspectiveTransform(pts_src_ref, pts_dst_ref * HOMOGRAPHY_SCALE)
        print("\nHomography matrix calculated.")

        # Calculate the real-world distance between the reference lines using homography.
        # This can serve as a fixed distance for speed calculation if desired.
        first_start_pt_real = pixel_to_realworld(pts_src_ref[0], H, HOMOGRAPHY_SCALE)
        first_end_pt_real = pixel_to_realworld(pts_src_ref[1], H, HOMOGRAPHY_SCALE)
        last_start_pt_real = pixel_to_realworld(pts_src_ref[2], H, HOMOGRAPHY_SCALE)
        last_end_pt_real = pixel_to_realworld(pts_src_ref[3], H, HOMOGRAPHY_SCALE)

        # Using midpoints for distance calculation
        first_mid_real = (first_start_pt_real + first_end_pt_real) / 2
        last_mid_real = (last_start_pt_real + last_end_pt_real) / 2
        fixed_homography_distance = np.linalg.norm(last_mid_real - first_mid_real)
        print(f"Pre-calculated distance between reference lines (using homography): {fixed_homography_distance:.2f} meters")

    except Exception as e:
        print(f"Error calculating homography or distances: {e}")
        print("Please ensure pts_src_ref and pts_dst_ref in elements/constants.py are correctly defined.")
        exit(1)
    # --- End Homography Calculation ---

    # Initialize the tracking pipeline, passing ALL relevant paths
    pipeline = TrackingPipeline(
        checkpoint_path=args.checkpoint,
        polygon_coords=polygon_points,
        homography_matrix=H,
        homography_distance=fixed_homography_distance,
        embedder_type=args.embedder,
        embedder_weights_path=args.embedder_weights,
        annotations_list_path=args.annotations_list, # Passed here
        output_file_path=args.output_file,          # Passed here
        output_video_path=args.output_video,        # Passed here
        pred_csv_file_path=args.pred_csv            # Passed here
    )

    # Run the pipeline. output_video_path is passed again just for function signature but it's now internal to pipeline
    pipeline.run(
        frames_dir=args.frames_dir,
        output_video_path=args.output_video, # This argument is somewhat redundant now but harmless
        progress_interval=args.progress_interval
    )

    print(f"\n--- Processing Complete ---")
    print(f"Results saved:")
    print(f"- Output video: {args.output_video}")
    print(f"- Tracking predictions (MOT format): {args.output_file}")
    print(f"- Predicted speeds CSV: {args.pred_csv}")


if __name__ == "__main__":
    main()

