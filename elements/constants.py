import numpy as np

# --- Paths and Checkpoints ---
# Default directories and file paths for input/output.
# These can be overridden by command-line arguments in scripts/run_inference.py
FRAMES_DIR = '/path/to/dataset/test'
ANNOTATIONS_TEST = '/path/to/dataset/annotations/testing_set_f.json'
ANNOTATIONS_LIST_FILE = '/path/to/dataset/final_ds_management/artifacts/test_images_with_annotations_f.txt'
CHECKPOINT_PATH = '/path/to/dataset/final_ds_management/artifacts/detr_experiment/version_1/checkpoints/epoch=30-step=18879.ckpt'
PRED_CSV_FILE = "/path/to/dataset/DETR_deepsort_pretrained_metrics/predicted_base_osnet_speeds.csv"
OUTPUT_VIDEO = "/path/to/dataset/results/tracking_base_detr_osnet_fullds.mp4"
OUTPUT_FILE = "/path/to/dataset/DETR_deepsort_pretrained_metrics/predictions_DETR_deepsort_FT_working.txt"

# --- Model and Tracking Parameters ---
CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence for object detection
FPS = 5                    # Frames per second for video processing and speed calculation

# --- Real-world Measurement Configuration ---
# These parameters are used for speed estimation based on homography.
BLOCK_LENGTH = 0.5          # Length of a single block in meters
INTER_BLOCK_SPACING = 0.5   # Spacing between blocks in meters
NUM_BLOCKS = 14
NUM_GAPS = 13
TOTAL_DISTANCE = (NUM_BLOCKS * BLOCK_LENGTH) + (NUM_GAPS * INTER_BLOCK_SPACING) # Total distance for speed calculation

# --- Speed Calculation and Tracking Constants ---
LINE_DETECTION_THRESHOLD = 15  # Pixel threshold for detecting line crossings
MIN_CROSSING_FRAMES = 5        # Minimum frames required to consider a valid crossing
MAX_CROSSING_FRAMES = 2000     # Maximum frames for a valid crossing (prevents stalled tracks)
ENTRY_THRESHOLD = 15           # Consistent threshold for initial entry detection
EXIT_THRESHOLD = 15            # Consistent threshold for exit detection
MIN_SPEED = 3                  # Minimum valid speed in km/h
MAX_SPEED = 50                 # Maximum valid speed in km/h
MIN_PIXEL_DISTANCE = 50        # Minimum pixel distance traveled for a valid crossing

# --- Debugging Flags ---
DEBUG = True # Enable or disable debug print statements and visualizations

# --- Homography and Region Definitions ---
# Reference points in destination (real-world) coordinates for homography calculation.
# These points define a rectangular region in the real world.
# Scaled by 'scale' for better numerical stability during homography calculation.
pts_dst_ref = np.array([
    [0, 0],      # Top-left (0m, 0m)
    [4, 0],      # Top-right (4m, 0m)
    [4, 10.5],   # Bottom-right (4m, 10.5m)
    [0, 10.5]    # Bottom-left (0m, 10.5m)
], dtype=np.float32)

# Corresponding source (image pixel) coordinates for homography calculation.
pts_src_ref = np.array([
    [937.30, 390.20],  # Top-left
    [1100.30, 388.70], # Top-right
    [1084.60, 506.20], # Bottom-right
    [862.40, 504.00]   # Bottom-left
], dtype=np.float32)

# Points defining the full cycle track in pixel coordinates.
# This polygon defines the region of interest for detection.
pts_track_pixels = np.array([
    [937.40, 390.40],    # Top-left
    [1100.50, 388.80],   # Top-right
    [1003.95, 1080.00],  # Bottom-right
    [469.90, 1079.50]    # Bottom-left
], dtype=np.float32)

# Define the lines used for crossing detection in pixel coordinates.
# These lines represent the entry/exit points of the measurement zone.
FIRST_LINE = {
    "start": (min(832, 1085), 544), # Automatically determines min/max for consistent line definition
    "end": (max(832, 1085), 548)
}

LAST_LINE = {
    "start": (min(930, 1103), 387),
    "end": (max(930, 1103), 387)
}

# Scale factor for homography (1 pixel = 1 cm, so multiply real-world by 100 for pixels)
HOMOGRAPHY_SCALE = 100

# Placeholder for homography matrix (calculated once at startup)
HOMOGRAPHY_MATRIX = None

# Placeholder for calculated real-world distance between lines (derived from homography)
HOMOGRAPHY_DISTANCE = None

# --- Cyclist Records and Logging ---
# Dictionary to store tracking information for each cyclist.
# Format: {track_id: {"entry_time": float, "exit_time": float, ...}}
CYCLIST_RECORDS = {}

# Log to store recent speed measurements for display.
SPEED_LOG = []
MAX_LOG_ENTRIES = 100 # Maximum number of recent speed measurements to display

# Initialize an empty dictionary to store track information
TRACK_INFO = {}

# Headers for the predictions CSV file.
PRED_CSV_HEADERS = ["track_id", "speed_kph", "direction", "crossing_time_sec", "frames_taken", "entry_frame", "exit_frame"]

