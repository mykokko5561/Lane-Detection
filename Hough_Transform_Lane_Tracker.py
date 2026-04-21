import cv2
import numpy as np
from typing import Tuple, List, Optional

# ============================================================================
# MAIN PIPELINE: HOUGH TRANSFORM LANE TRACKER
# ============================================================================

class HoughLaneTracker:
    """
    A lightweight, line-based lane tracking system using Canny Edge Detection
    and Hough Line Transform.
    """
    def __init__(self):
        # State memory to prevent flickering or crashing if lines are lost in a frame
        self.prev_left_fit: Optional[Tuple[float, float]] = None
        self.prev_right_fit: Optional[Tuple[float, float]] = None

    def apply_canny(self, frame: np.ndarray) -> np.ndarray:
        """Applies grayscale conversion, Gaussian blur, and Canny edge detection."""
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        canny_edges = cv2.Canny(blurred_image, 50, 150)
        return canny_edges

    def apply_region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        """Isolates the triangular region of the image where lanes are expected."""
        height = edges.shape[0]
        # Defined as a triangle matching the original logic
        polygon_points = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon_points, 255)
        
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges

    def get_line_coordinates(self, image: np.ndarray, line_parameters: Tuple[float, float]) -> np.ndarray:
        """Calculates endpoint coordinates (x1, y1, x2, y2) for a line given its slope and intercept."""
        slope, intercept = line_parameters
        
        y1 = image.shape[0]           # Bottom of the image
        y2 = int(y1 * (3 / 5))        # Extending slightly above the middle of the image
        
        # Calculate x values based on y = mx + b -> x = (y - b) / m
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image: np.ndarray, hough_lines: Optional[np.ndarray]) -> np.ndarray:
        """Separates left/right lines, averages them, and returns final coordinates."""
        left_fits = []
        right_fits = []

        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line.reshape(4)
                # Fit a 1st-degree polynomial (y = mx + b) to get slope [0] and intercept [1]
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]

                # Left lane lines have negative slope, right lane lines have positive slope
                if slope < 0:
                    left_fits.append((slope, intercept))
                else:
                    right_fits.append((slope, intercept))

        # Calculate averages (will be NaN/None if list is empty)
        left_fit_average = np.average(left_fits, axis=0) if left_fits else None
        right_fit_average = np.average(right_fits, axis=0) if right_fits else None

        # Print for debugging (Matches original code behavior)
        print(f"Left Fit Average:  {left_fit_average}")
        print(f"Right Fit Average: {right_fit_average}")

        # Safely attempt to generate coordinates, fallback to previous frame if lost
        try:
            left_line = self.get_line_coordinates(image, left_fit_average)
            self.prev_left_fit = left_fit_average
        except Exception:
            # Fallback to the last known good line if current calculation fails
            if self.prev_left_fit is not None:
                left_line = self.get_line_coordinates(image, self.prev_left_fit)
            else:
                left_line = np.array([0, 0, 0, 0])

        try:
            right_line = self.get_line_coordinates(image, right_fit_average)
            self.prev_right_fit = right_fit_average
        except Exception:
            # Fallback to the last known good line if current calculation fails
            if self.prev_right_fit is not None:
                right_line = self.get_line_coordinates(image, self.prev_right_fit)
            else:
                right_line = np.array([0, 0, 0, 0])

        return np.array([left_line, right_line])

    def draw_lines(self, image: np.ndarray, lines: np.ndarray) -> np.ndarray:
        """Draws the computed lines onto a blank mask."""
        line_mask = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                if len(line) == 4 and not np.all(line == 0): # Prevent drawing [0,0,0,0] fallback lines
                    x1, y1, x2, y2 = line
                    cv2.line(line_mask, (x1, y1), (x2, y2), (255, 0, 100), 12)
        return line_mask

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Executes the complete lane detection pipeline on a single frame."""
        canny_image = self.apply_canny(frame)
        cropped_image = self.apply_region_of_interest(canny_image)
        
        # Detect line segments using Probabilistic Hough Transform
        hough_lines = cv2.HoughLinesP(
            cropped_image, 
            rho=2, 
            theta=np.pi/180, 
            threshold=100, 
            lines=np.array([]), 
            minLineLength=40, 
            maxLineGap=5
        )
        
        averaged_lines = self.average_slope_intercept(frame, hough_lines)
        line_overlay = self.draw_lines(frame, averaged_lines)
        
        # Blend the original frame with the line overlay
        blended_result = cv2.addWeighted(frame, 0.9, line_overlay, 1, 1)
        
        return blended_result, canny_image, cropped_image


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    video_path = "LaneVideo.mp4"
    cap = cv2.VideoCapture(video_path)
    
    tracker = HoughLaneTracker()

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
        
        # Run the pipeline
        final_result, canny_edges, roi_view = tracker.process_frame(current_frame)
        
        # Display the output streams
        cv2.imshow("1 - Canny Edge Detection", canny_edges)
        cv2.imshow("2 - Region of Interest (ROI)", roi_view)
        cv2.imshow("3 - Final Hough Lane Tracking", final_result)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            print("[INFO] Process terminated by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
