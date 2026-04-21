# Note: If your camera angle differs from mine, you will need to adjust the red dot coordinates (as shown in photo 4). 
# You can use the 'placing_dots.py' script to find the correct (x, y) values. 
# Just copy and paste that snippet into this code to locate your points. 
# Once you update lines 37-40, you can delete the temporary helper code.
import cv2
import numpy as np

# Initialize video capture
video_capture = cv2.VideoCapture("LaneVideo.mp4")
is_reading_success, raw_frame = video_capture.read()

def empty_callback(x):
    """Empty callback function for trackbars."""
    pass

# Setup debug window and trackbars for HSV masking
cv2.namedWindow("HSV_Trackbars")

cv2.createTrackbar("Hue Lower", "HSV_Trackbars", 0, 255, empty_callback)
cv2.createTrackbar("Saturation Lower", "HSV_Trackbars", 0, 255, empty_callback)
cv2.createTrackbar("Value Lower", "HSV_Trackbars", 200, 255, empty_callback)
cv2.createTrackbar("Hue Upper", "HSV_Trackbars", 255, 255, empty_callback)
cv2.createTrackbar("Saturation Upper", "HSV_Trackbars", 50, 255, empty_callback)
cv2.createTrackbar("Value Upper", "HSV_Trackbars", 255, 255, empty_callback)

# Store previous lane positions to handle missing detection frames
previous_left_x_coords = []
previous_right_x_coords = []

while is_reading_success:
    is_reading_success, raw_frame = video_capture.read()
    if not is_reading_success:
        break
        
    resized_frame = cv2.resize(raw_frame, (640, 480))

    ## 1. Perspective Transformation (Bird's Eye View)
    # Define Region of Interest (ROI) points
    top_left = (222, 387)
    bottom_left = (70, 472)
    top_right = (400, 380)
    bottom_right = (538, 472)

    # Draw points on the original frame for debugging
    cv2.circle(resized_frame, top_left, 5, (0, 0, 255), -1)
    cv2.circle(resized_frame, bottom_left, 5, (0, 0, 255), -1)
    cv2.circle(resized_frame, top_right, 5, (0, 0, 255), -1)
    cv2.circle(resized_frame, bottom_right, 5, (0, 0, 255), -1)

    # Apply warp perspective
    source_points = np.float32([top_left, bottom_left, top_right, bottom_right]) 
    destination_points = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points) 
    birds_eye_frame = cv2.warpPerspective(resized_frame, perspective_matrix, (640, 480))

    ## 2. Object Detection & Color Masking
    hsv_frame = cv2.cvtColor(birds_eye_frame, cv2.COLOR_BGR2HSV)
    
    # Get current trackbar positions
    hue_lower = cv2.getTrackbarPos("Hue Lower", "HSV_Trackbars")
    sat_lower = cv2.getTrackbarPos("Saturation Lower", "HSV_Trackbars")
    val_lower = cv2.getTrackbarPos("Value Lower", "HSV_Trackbars")
    hue_upper = cv2.getTrackbarPos("Hue Upper", "HSV_Trackbars")
    sat_upper = cv2.getTrackbarPos("Saturation Upper", "HSV_Trackbars")
    val_upper = cv2.getTrackbarPos("Value Upper", "HSV_Trackbars")
    
    lower_bounds = np.array([hue_lower, sat_lower, val_lower])
    upper_bounds = np.array([hue_upper, sat_upper, val_upper])
    lane_mask = cv2.inRange(hsv_frame, lower_bounds, upper_bounds)

    ## 3. Histogram Analysis to Find Lane Bases
    histogram = np.sum(lane_mask[lane_mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_lane_base = np.argmax(histogram[:midpoint])
    right_lane_base = np.argmax(histogram[midpoint:]) + midpoint

    ## 4. Sliding Window Algorithm
    current_y = 472
    current_left_x_coords = []
    current_right_x_coords = []

    # Copy mask for sliding window visualization
    sliding_window_vis = lane_mask.copy()

    while current_y > 0:
        # Left lane tracking
        left_window_roi = lane_mask[current_y-40:current_y, left_lane_base-50:left_lane_base+50]
        contours, _ = cv2.findContours(left_window_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                current_left_x_coords.append(left_lane_base - 50 + center_x)
                left_lane_base = left_lane_base - 50 + center_x
        
        # Right lane tracking
        right_window_roi = lane_mask[current_y-40:current_y, right_lane_base-50:right_lane_base+50]
        contours, _ = cv2.findContours(right_window_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                current_right_x_coords.append(right_lane_base - 50 + center_x)
                right_lane_base = right_lane_base - 50 + center_x
        
        # Draw sliding windows for visualization
        cv2.rectangle(sliding_window_vis, (left_lane_base-50, current_y), (left_lane_base+50, current_y-40), (255, 255, 255), 2)
        cv2.rectangle(sliding_window_vis, (right_lane_base-50, current_y), (right_lane_base+50, current_y-40), (255, 255, 255), 2)
        current_y -= 40
        
    ## 5. Validate and Map Coordinates
    # Fallback to previous coordinates if detection fails in this frame
    if len(current_left_x_coords) == 0:
        current_left_x_coords = previous_left_x_coords
    else:
        previous_left_x_coords = current_left_x_coords
        
    if len(current_right_x_coords) == 0:
        current_right_x_coords = previous_right_x_coords
    else:
        previous_right_x_coords = current_right_x_coords

    # Ensure arrays have matching dimensions for polygon drawing
    min_array_length = min(len(current_left_x_coords), len(current_right_x_coords))

    # Define the boundaries of the detected lane
    lane_top_left = (current_left_x_coords[0], 472)
    lane_bottom_left = (current_left_x_coords[min_array_length-1], 0)
    lane_top_right = (current_right_x_coords[0], 472)
    lane_bottom_right = (current_right_x_coords[min_array_length-1], 0)
    
    polygon_points = np.array([lane_top_left, lane_bottom_left, lane_bottom_right, lane_top_right], dtype=np.int32)
    polygon_points = polygon_points.reshape((-1, 1, 2))

    # Draw the detected lane area
    lane_overlay = birds_eye_frame.copy()
    cv2.fillPoly(lane_overlay, [polygon_points], (0, 255, 0))

    overlay_opacity = 0.2
    cv2.addWeighted(lane_overlay, overlay_opacity, birds_eye_frame, 1 - overlay_opacity, 0, birds_eye_frame)

    ## 6. Inverse Transformation (Project back to original view)
    inverse_perspective_matrix = cv2.getPerspectiveTransform(destination_points, source_points)
    original_perspective_lane = cv2.warpPerspective(birds_eye_frame, inverse_perspective_matrix, (640, 480))

    final_result = cv2.addWeighted(resized_frame, 1, original_perspective_lane, 0.5, 0)

    ## 7. Display Output Streams
    cv2.imshow("1 - Original Frame", resized_frame)
    cv2.imshow("2 - Bird's Eye View", birds_eye_frame)
    cv2.imshow("3 - Color Masking", lane_mask)
    cv2.imshow("4 - Sliding Windows", sliding_window_vis)
    cv2.imshow("5 - Final Result", final_result)

    if cv2.waitKey(10) == 27: # Press 'ESC' to exit
        break

# Clean up resources
video_capture.release()
cv2.destroyAllWindows()
