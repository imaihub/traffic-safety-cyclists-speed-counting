from typing import Dict, List, Tuple, Optional

import numpy as np
import os
from collections import defaultdict
import motmetrics as mm
from pathlib import Path
import argparse
import pandas as pd  # Import pandas for saving to CSV

# DEFAULT FILES - These paths are relative to the project root (BikeSafeAI/)
# The script expects these to be the output of your run_inference.py
PREDICTION_FILES = {
    "mobilenet": "results/predictions_DETR_deepsort_mobilenet_working.txt",
    "clip": "results/predictions_DETR_deepsort_base_clip_working.txt",
    "osnet": "results/predictions_DETR_deepsort_osnet_working.txt",
}


def load_detections_from_list(file_path: str) -> Dict[str, List[Tuple[int, List[float]]]]:
    """
    Load detections from a tracking results file (e.g., in MOT format).
    Expected format per line: frame_name,track_id,x,y,w,h,confidence,category_id,visibility
    The frame_name can be a full path or just the filename; only the filename is extracted.

    Args:
        file_path (str): Path to the tracking results file.

    Returns:
        defaultdict: A dictionary where keys are frame filenames and values are lists
                     of (track_id, [x, y, w, h]) tuples.
    """
    detections = defaultdict(list)
    total_detections = 0
    print(f"Loading detections from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Detection file does not exist: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:  # Ensure at least frame, id, x, y, w, h are present
                print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
                continue

            # Extract just the filename from the potentially full path in parts[0]
            frame_name = Path(parts[0]).name
            try:
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError as e:
                print(f"Warning: Skipping line with non-numeric values in {file_path}: {line.strip()} - {e}")
                continue

            bbox = [x, y, w, h]
            detections[frame_name].append((track_id, bbox))
            total_detections += 1
    print(f"Successfully loaded {total_detections} detections from {file_path} across {len(detections)} frames.")
    return detections


def bbox_iou(bb1: List[float], bb2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are in [x, y, w, h] format.
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    x1_bb1, y1_bb1, w1_bb1, h1_bb1 = bb1
    x2_bb1, y2_bb1 = x1_bb1 + w1_bb1, y1_bb1 + h1_bb1

    x1_bb2, y1_bb2, w2_bb2, h2_bb2 = bb2
    x2_bb2, y2_bb2 = x1_bb2 + w2_bb2, y1_bb2 + h2_bb2

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    bb1_area = w1_bb1 * h1_bb1
    bb2_area = w2_bb2 * h2_bb2

    # Compute the area of union
    union_area = bb1_area + bb2_area - intersection_area

    # Avoid division by zero
    iou = intersection_area / (union_area + 1e-6)  # Add a small epsilon
    return iou


def evaluate_mot(gt_file: str, pred_file: str, iou_threshold: float = 0.5,
                 max_frames: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluate tracking performance using MOT metrics.

    Args:
        gt_file (str): Path to ground truth tracking file.
        pred_file (str): Path to predicted tracking results file.
        iou_threshold (float): IoU threshold for matching detections.
        max_frames (int, optional): Maximum number of frames to evaluate. If None, all frames are evaluated.

    Returns:
        pandas.DataFrame: A summary of the calculated MOT metrics.
    """
    gt = load_detections_from_list(gt_file)
    pred = load_detections_from_list(pred_file)

    all_frames = sorted(set(gt.keys()) | set(pred.keys()))
    if max_frames:
        print(f"Limiting evaluation to the first {max_frames} frames.")
        all_frames = all_frames[:max_frames]

    print('\nFrame Statistics:')
    print(f'Ground Truth Frames with detections: {len(gt)}')
    print(f'Predicted Frames with detections: {len(pred)}')
    print(f'Evaluating on: {len(all_frames)} common and unique frames.')

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_idx, frame_name in enumerate(all_frames):
        gt_objs = gt.get(frame_name, [])
        pred_objs = pred.get(frame_name, [])

        gt_ids = [obj[0] for obj in gt_objs]
        pred_ids = [obj[0] for obj in pred_objs]
        gt_boxes = [obj[1] for obj in gt_objs]
        pred_boxes = [obj[1] for obj in pred_objs]

        # Initialize distance matrix with NaNs
        distance_matrix = np.full((len(gt_boxes), len(pred_boxes)), np.nan, dtype=np.float32)

        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou = bbox_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    distance_matrix[i, j] = 1 - iou  # Cost is 1 - IoU

        # Handle cases where one list is empty, motmetrics expects specific behavior
        # (This is implicitly handled by the default_accumulator if gt_ids/pred_ids are empty,
        # but explicit updates with empty lists and NaN matrices can also work depending on mm version)
        # acc.update handles empty lists gracefully for mm >= 1.2.0

        acc.update(gt_ids, pred_ids, distance_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='Tracking')

    print("\nEvaluation Results:")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    return summary


def parse_arguments():
    """Parse command line arguments for MOT evaluation."""
    parser = argparse.ArgumentParser(
        description='Multiple Object Tracking (MOT) Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--gt', required=True,
                        default='BikeSafeAI/dataset/artifacts/groundtruth_test_fullds.txt',  # Default GT path
                        help='Path to ground truth tracking file')

    parser.add_argument('--pred',
                        help='Path to predicted tracking results file (overridden if --pred-choice is used). '
                             'This should be relative to BikeSafeAI/ or an absolute path.')
    parser.add_argument('--pred-choice', choices=PREDICTION_FILES.keys(),
                        help='Use a predefined prediction file name (mobilenet, clip, osnet) '
                             'instead of manually specifying --pred. Paths are relative to project root.')

    # Optional arguments
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for detection matching')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to evaluate (e.g., 50 for initial testing)')
    parser.add_argument('--output', default=None,
                        help='Path to save evaluation results (CSV format). '
                             'e.g., results/evaluation_summary.csv')

    return parser.parse_args()


def validate_args(args):
    """
    Validate input arguments.
    Note: Paths are handled more robustly by `load_detections_from_list` now.
    """
    if args.pred_choice:
        # If pred_choice is used, `args.pred` will be updated with a relative path.
        # We need to make it absolute relative to the current working directory.
        full_pred_path = os.path.join(os.getcwd(), PREDICTION_FILES[args.pred_choice])
        if not os.path.exists(full_pred_path):
            raise FileNotFoundError(f"Prediction file for choice '{args.pred_choice}' not found at: {full_pred_path}")
        args.pred = full_pred_path  # Update args.pred with the absolute path for evaluation

    # Check for ground truth existence
    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"Ground truth file not found: {args.gt}")

    # If a custom pred path is given (not via pred-choice)
    if not args.pred:  # This means neither --pred nor --pred-choice was used to set it
        raise ValueError("Either --pred or --pred-choice must be specified.")
    elif not os.path.exists(args.pred):  # If --pred was given, check its existence
        raise FileNotFoundError(f"Prediction file not found: {args.pred}")

    if not (0 < args.iou_threshold <= 1):  # IoU threshold must be strictly > 0
        raise ValueError("IoU threshold must be between 0 (exclusive) and 1 (inclusive)")


def main():
    args = parse_arguments()

    try:
        # If --pred-choice is used, update args.pred with the correct path
        if args.pred_choice:
            args.pred = PREDICTION_FILES[args.pred_choice]  # This sets a relative path from PREDICTION_FILES

        # Now validate paths, including making args.pred absolute if it came from pred_choice
        validate_args(args)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)

    print(f"\nEvaluating tracking results:")
    print(f"Ground truth: {args.gt}")
    print(f"Predictions: {args.pred}")
    print(f"IoU threshold: {args.iou_threshold}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")

    summary = evaluate_mot(
        gt_file=args.gt,
        pred_file=args.pred,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames
    )

    if args.output:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        summary.to_csv(args.output)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
