import math

import cv2
import numpy as np


def get_foot_position(bbox):
    """
    Calculates the bottom-center coordinate (foot position) of a bounding box.
    Useful for tracking objects and determining their ground contact point.

    Args:
        bbox (tuple/list): Bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
        tuple: (x_center, y_bottom) of the bounding box.
    """
    return (bbox[0] + bbox[2]) / 2, bbox[3] # (x_center, y_bottom)

def is_in_block(x, y, block_config):
    """
    Checks if a given (x, y) point is within the boundaries of a defined block.

    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        block_config (dict): Dictionary with 'x_range' and 'y_range' defining the block.
                             e.g., {"x_range": (x_min, x_max), "y_range": (y_min, y_max)}

    Returns:
        bool: True if the point is inside the block, False otherwise.
    """
    return (block_config["x_range"][0] <= x <= block_config["x_range"][1] and
            block_config["y_range"][0] <= y <= block_config["y_range"][1])

def is_inside_polygon(bbox, polygon_points):
    """
    Checks if the center of a bounding box is inside a given polygon.

    Args:
        bbox (list/tuple): Bounding box coordinates (x_min, y_min, width, height).
                           Note: This expects (x_min, y_min, width, height) format,
                           not (x_min, y_min, x_max, y_max).
        polygon_points (numpy.ndarray): An array of polygon vertices, e.g., np.array([[x1,y1],[x2,y2],...]).

    Returns:
        bool: True if the bbox center is inside or on the polygon boundary, False otherwise.
    """
    x_min, y_min, width, height = bbox
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    # cv2.pointPolygonTest returns a positive value if the point is inside,
    # 0 if on the edge, and a negative value if outside.
    return cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0

def get_relative_position(point, polygon):
    """
    Determines if a point is near the top or bottom boundary of a polygon.
    Used to infer initial entry side for directional counting.

    Args:
        point (tuple): (x, y) coordinates of the point.
        polygon (numpy.ndarray): An array of polygon vertices, e.g., np.array([[x1,y1],[x2,y2],...]).

    Returns:
        str: "top", "bottom", or "inside" indicating the relative position.
    """
    y = point[1]
    # Find min and max y-coordinates of the polygon
    poly_ymin = polygon[:, :, 1].min()
    poly_ymax = polygon[:, :, 1].max()

    if y < poly_ymin + 50:  # Near top boundary (arbitrary threshold 50 pixels)
        return "top"
    elif y > poly_ymax - 50: # Near bottom boundary
        return "bottom"
    return "inside"

def get_crossing_lines(polygon):
    """
    Extracts the approximate top and bottom horizontal lines from a given polygon.
    Assumes the polygon roughly defines a rectangular region and uses the
    highest and lowest y-coordinates to define these lines.

    Args:
        polygon (numpy.ndarray): An array of polygon vertices in the format
                                 np.array([[[x1,y1]],[[x2,y2]],...]) or similar.

    Returns:
        dict: A dictionary containing 'top_line' and 'bottom_line' as tuples
              of (point1, point2) representing the line segments.
    """
    points = polygon.reshape(-1, 2) # Flatten to (N, 2)
    # Sort points by y-coordinate to find top and bottom points
    sorted_by_y = sorted(points, key=lambda p: p[1])

    # The two points with the smallest y-values form the top line
    top_points = sorted_by_y[:2]
    # The two points with the largest y-values form the bottom line
    bottom_points = sorted_by_y[-2:]

    # Ensure points for lines are ordered for consistent drawing (e.g., by x-coordinate)
    top_line = (tuple(top_points[0].astype(int)), tuple(top_points[1].astype(int)))
    bottom_line = (tuple(bottom_points[0].astype(int)), tuple(bottom_points[1].astype(int)))

    return {
        'top_line': top_line,
        'bottom_line': bottom_line
    }

def point_near_line(point, line, threshold=10):
    """
    Checks if a point is close to a line segment within a given threshold.
    This function specifically checks if the projection of the point lies
    within the line segment and its perpendicular distance is within the threshold.

    Args:
        point (tuple): (x, y) coordinates of the point.
        line (tuple): ((x1, y1), (x2, y2)) representing the start and end of the line segment.
        threshold (float): Maximum distance for a point to be considered "near" the line.

    Returns:
        bool: True if the point is near the line segment, False otherwise.
    """
    x, y = point
    (x1, y1), (x2, y2) = line

    # Calculate squared length of the line segment
    dx, dy = x2 - x1, y2 - y1
    line_len_sq = dx*dx + dy*dy

    if line_len_sq == 0: # Line is a point
        return math.sqrt((x - x1)**2 + (y - y1)**2) < threshold

    # Calculate parameter t for projection of point onto the line
    # t = ((x - x1) * dx + (y - y1) * dy) / line_len_sq
    # This t represents where the projection falls along the line segment (0 to 1)

    # Calculate squared distance from point to the infinite line
    dist_sq = ((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y1 * x1)**2 / line_len_sq

    # If the point's projection falls outside the segment,
    # check distance to endpoints instead
    # if t < 0:
    #     dist_sq = (x - x1)**2 + (y - y1)**2
    # elif t > 1:
    #     dist_sq = (x - x2)**2 + (y - y2)**2

    return dist_sq < threshold**2 # Compare squared distances to avoid sqrt

def point_to_line_distance(point, line):
    """
    Calculates the perpendicular distance from a point to an infinite line defined by two points.

    Args:
        point (tuple): (x, y) coordinates of the point.
        line (dict): A dictionary with 'start' and 'end' keys, each a tuple (x, y).

    Returns:
        float: The perpendicular distance from the point to the line.
               Returns 0 if the line is a single point.
    """
    x, y = point
    (x1, y1), (x2, y2) = line["start"], line["end"]

    # Calculate components of the line vector
    dx = x2 - x1
    dy = y2 - y1

    # Length of the line segment squared
    line_length_sq = dx*dx + dy*dy

    if line_length_sq == 0: # Handle case where line is a single point
        return math.sqrt((x - x1)**2 + (y - y1)**2) # Distance to the point itself

    # Calculate the cross product for the distance (Area of parallelogram / base length)
    # Formula for distance from point (x0, y0) to line Ax + By + C = 0 is |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    # Line equation from (x1, y1) to (x2, y2) is (y2-y1)X - (x2-x1)Y + (x2y1-x1y2) = 0
    numerator = abs(dy * x - dx * y + x2 * y1 - y2 * x1)
    denominator = math.sqrt(line_length_sq)

    return numerator / denominator

def pixel_to_realworld(point_px, homography_matrix, scale):
    """
    Converts pixel coordinates to real-world coordinates (in meters) using a homography matrix.

    Args:
        point_px (tuple): (x, y) pixel coordinates.
        homography_matrix (numpy.ndarray): The 3x3 homography matrix.
        scale (float): The scale factor used when computing the homography
                       (e.g., 100 if real-world coordinates were scaled by 100).

    Returns:
        numpy.ndarray: (x, y) real-world coordinates in meters.
    """
    point_px = np.array([[point_px]], dtype=np.float32) # Reshape for cv2.perspectiveTransform
    point_real = cv2.perspectiveTransform(point_px, homography_matrix)[0][0] / scale
    return point_real # Returns (x,y) in meters

def load_annotated_images_list(file_path):
    """
    Loads a list of image paths (or names) from a text file, where each line
    in the file corresponds to an image that has annotations.

    Args:
        file_path (str): The path to the text file containing the list of image names.

    Returns:
        list: A list of strings, each representing an image path/name.
    """
    with open(file_path, 'r') as f:
        # Read each line, strip whitespace, and filter out empty lines.
        return [line.strip() for line in f.readlines() if line.strip()]

def save_tracked_video(vis_data_list, output_path, fps=20):
    """
    Saves a list of processed frames (with tracking visualizations) as a video file.

    Args:
        vis_data_list (list): A list of dictionaries, where each dictionary
                              contains a "frame" (numpy.ndarray) and optionally "tracks" data.
        output_path (str): The full path including filename for the output video (e.g., "output.mp4").
        fps (int): Frames per second for the output video.
    """
    # Check for valid input data.
    if not vis_data_list or not isinstance(vis_data_list, list):
        print("No valid frame data available for video saving.")
        return

    try:
        # Get video dimensions from the first valid frame.
        first_frame = None
        for frame_data in vis_data_list:
            if "frame" in frame_data and isinstance(frame_data["frame"], np.ndarray):
                first_frame = frame_data["frame"]
                break

        if first_frame is None:
            print("No valid frame found in vis_data_list to determine video dimensions.")
            return

        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 files
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for path {output_path}")
            return

        # Iterate through visualized frames and write them to the video.
        print(f"Saving video to {output_path}...")
        for frame_data in vis_data_list:
            if "frame" not in frame_data or not isinstance(frame_data["frame"], np.ndarray):
                continue

            # Make a copy to avoid modifying the original frame data in vis_data_list
            frame = frame_data["frame"].copy()

            # Ensure the frame is in BGR format for OpenCV's VideoWriter
            if frame.shape[2] == 3 and frame.dtype == np.uint8 and np.any(frame[:,:,0] != frame[:,:,2]): # Simple check for RGB
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_to_write = frame

            video_writer.write(frame_to_write)

        video_writer.release() # Release the video writer object
        print(f"Video successfully saved to {output_path}")

    except Exception as e:
        print(f"Error generating video: {str(e)}")
        if 'video_writer' in locals() and video_writer.isOpened():
            video_writer.release()

def visualize_frame(image_np, predictions, frame_id):
    """
    Visualizes tracking results on a single frame using Matplotlib.
    This is primarily for debugging and showing individual frames.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array (RGB).
        predictions (list): A list of dictionaries, each containing 'bbox', 'track_id', and 'confidence'.
        frame_id (int): The current frame number.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    ax = plt.gca()
    plt.title(f"Frame {frame_id} - Tracked Cyclists")

    for pred in predictions:
        box = pred["bbox"]
        track_id = pred["track_id"]
        confidence = pred.get("confidence", 1.0) # Use .get with default for robustness

        # Draw bounding box
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],  # width
            box[3] - box[1],  # height
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Draw label
        label = f"ID:{track_id} ({confidence:.2f})-(Cyclist)"
        plt.text(
            box[0], box[1] - 10,
            label,
            color="white",
            fontsize=5,
            bbox=dict(facecolor="red", alpha=0.7)
        )

    plt.axis("off")
    plt.show()

def visualize_directions(image, top_line, bottom_line, northbound_count, southbound_count):
    """
    Visualizes crossing direction counts and boundary lines on the frame.

    Args:
        image (numpy.ndarray): The input image frame (will be modified).
        top_line (tuple): ((x1, y1), (x2, y2)) for the top boundary line.
        bottom_line (tuple): ((x1, y1), (x2, y2)) for the bottom boundary line.
        northbound_count (int): Current count of northbound cyclists.
        southbound_count (int): Current count of southbound cyclists.

    Returns:
        numpy.ndarray: The modified image with visualizations.
    """
    # Draw boundary lines (in BGR format: (0, 255, 255) is yellow)
    cv2.line(image, top_line[0], top_line[1], (0, 255, 255), 2)
    cv2.line(image, bottom_line[0], bottom_line[1], (0, 255, 255), 2)

    # Get image dimensions
    img_h, img_w = image.shape[:2]

    # Position settings for the display box (top-left for consistent placement)
    box_x = 10
    box_width = 300
    box_height = 90
    box_y = 10 # Place near the top-left

    text_x = box_x + 10
    line_spacing = 35

    # Draw white rectangle for background
    cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)

    # Draw black text for counts
    cv2.putText(image, f"South-bound: {southbound_count}", (text_x, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, f"North-bound: {northbound_count}", (text_x, box_y + 30 + line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return image

def draw_blocks(frame, first_block_config, last_block_config):
    """
    Draws the 'first' and 'last' measurement blocks on a frame.
    (Note: These blocks were part of the original speed estimation logic
    and might not be strictly necessary if using line crossings for entry/exit).

    Args:
        frame (numpy.ndarray): The image frame to draw on (will be modified).
        first_block_config (dict): Configuration for the first block.
        last_block_config (dict): Configuration for the last block.

    Returns:
        numpy.ndarray: The modified frame with blocks drawn.
    """
    # Draw first block (green)
    cv2.rectangle(frame,
                  (first_block_config["x_range"][0], first_block_config["y_range"][0]),
                  (first_block_config["x_range"][1], first_block_config["y_range"][1]),
                  (0, 255, 0), 2)

    # Draw last block (red)
    cv2.rectangle(frame,
                  (last_block_config["x_range"][0], last_block_config["y_range"][0]),
                  (last_block_config["x_range"][1], last_block_config["y_range"][1]),
                  (0, 0, 255), 2)
    return frame

