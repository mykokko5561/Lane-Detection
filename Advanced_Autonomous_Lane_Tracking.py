# Not suitable for sloping roads.
import cv2
import numpy as np
from typing import Tuple, List, Optional

# ============================================================================
# UTILITY CLASSES: CONTROL & FILTERING
# ============================================================================

class PIDController:
    """Proportional-Integral-Derivative Controller for steering smoothing."""
    def __init__(self, kp: float = 0.003, ki: float = 0.00001, kd: float = 0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        self.previous_error = 0.0

    def compute(self, error: float) -> float:
        self.integral_error += error
        derivative_error = error - self.previous_error
        self.previous_error = error
        return (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)


class KalmanFilter1D:
    """1D Kalman Filter to smooth noisy lane center measurements."""
    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 1e-1):
        self.state_estimate = 0.0
        self.error_covariance = 1.0
        self.q = process_variance
        self.r = measurement_variance
        self.is_initialized = False

    def update(self, measurement: Optional[float]) -> float:
        if measurement is None:
            return self.state_estimate
        
        if not self.is_initialized:
            self.state_estimate = measurement
            self.is_initialized = True
            return measurement

        # Prediction & Update
        self.error_covariance += self.q
        kalman_gain = self.error_covariance / (self.error_covariance + self.r)
        self.state_estimate += kalman_gain * (measurement - self.state_estimate)
        self.error_covariance *= (1 - kalman_gain)
        
        return self.state_estimate


# ============================================================================
# MAIN PIPELINE: AUTONOMOUS LANE TRACKER
# ============================================================================

class AutonomousLaneTracker:
    def __init__(self):
        self.pid_controller = PIDController()
        self.kalman_filter = KalmanFilter1D()
        
        # State memory for fail-safe
        self.last_left_poly = None
        self.last_right_poly = None
        self.last_steering_angle = 0.0

    def apply_color_and_edge_threshold(self, frame: np.ndarray) -> np.ndarray:
        """Applies CLAHE, HLS/HSV color masking, and Sobel edge detection."""
        lab_color_space = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_color_space)

        clahe_optimizer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel_optimized = clahe_optimizer.apply(l_channel)

        hls_color_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        _, lightness_channel, _ = cv2.split(hls_color_space)
        white_lane_mask = cv2.inRange(lightness_channel, 200, 255)

        hsv_color_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_lower_bound = np.array([15, 80, 80])
        yellow_upper_bound = np.array([35, 255, 255])
        yellow_lane_mask = cv2.inRange(hsv_color_space, yellow_lower_bound, yellow_upper_bound)

        sobel_x = cv2.Sobel(l_channel_optimized, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = np.absolute(sobel_x)
        scaled_sobel = np.uint8(255 * abs_sobel_x / (np.max(abs_sobel_x) + 1e-6))
        _, edge_mask = cv2.threshold(scaled_sobel, 30, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(white_lane_mask, yellow_lane_mask), edge_mask)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, morph_kernel)

        return cleaned_mask

    def get_perspective_matrices(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates transformation matrices for Bird's Eye View."""
        source_points = np.float32([
            [width * 0.43, height * 0.62],
            [width * 0.57, height * 0.62],
            [width * 0.95, height],
            [width * 0.05, height]
        ])

        destination_points = np.float32([
            [width * 0.25, 0],
            [width * 0.75, 0],
            [width * 0.75, height],
            [width * 0.25, height]
        ])

        transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        inverse_matrix = cv2.getPerspectiveTransform(destination_points, source_points)
        return transform_matrix, inverse_matrix

    def execute_sliding_window(self, binary_warped: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects lane pixels using the sliding window algorithm."""
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        smoothed_histogram = np.convolve(histogram, np.ones(31)/31, mode='same')

        midpoint = len(smoothed_histogram) // 2
        left_base_x = np.argmax(smoothed_histogram[:midpoint])
        right_base_x = np.argmax(smoothed_histogram[midpoint:]) + midpoint

        non_zero_pixels = binary_warped.nonzero()
        non_zero_y = np.array(non_zero_pixels[0])
        non_zero_x = np.array(non_zero_pixels[1])

        num_windows = 9
        window_height = binary_warped.shape[0] // num_windows
        search_margin = 80
        min_pixels_recenter = 40

        current_left_x = left_base_x
        current_right_x = right_base_x

        left_lane_indices = []
        right_lane_indices = []

        for window in range(num_windows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low, win_xleft_high = current_left_x - search_margin, current_left_x + search_margin
            win_xright_low, win_xright_high = current_right_x - search_margin, current_right_x + search_margin

            good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & 
                              (non_zero_x >= win_xleft_low) & (non_zero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & 
                               (non_zero_x >= win_xright_low) & (non_zero_x < win_xright_high)).nonzero()[0]

            if len(good_left_inds) > min_pixels_recenter:
                current_left_x = int(np.mean(non_zero_x[good_left_inds]))
            if len(good_right_inds) > min_pixels_recenter:
                current_right_x = int(np.mean(non_zero_x[good_right_inds]))

            left_lane_indices.append((non_zero_x[good_left_inds], non_zero_y[good_left_inds]))
            right_lane_indices.append((non_zero_x[good_right_inds], non_zero_y[good_right_inds]))

        if not left_lane_indices or not right_lane_indices:
            return None, None, None, None

        left_x_coords = np.concatenate([np.array(p[0]) for p in left_lane_indices])
        left_y_coords = np.concatenate([np.array(p[1]) for p in left_lane_indices])
        right_x_coords = np.concatenate([np.array(p[0]) for p in right_lane_indices])
        right_y_coords = np.concatenate([np.array(p[1]) for p in right_lane_indices])

        return left_x_coords, left_y_coords, right_x_coords, right_y_coords

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main pipeline to process a single video frame."""
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]

        binary_mask = self.apply_color_and_edge_threshold(frame)
        transform_matrix, inverse_matrix = self.get_perspective_matrices(width, height)
        birds_eye_view = cv2.warpPerspective(binary_mask, transform_matrix, (width, height))

        left_x, left_y, right_x, right_y = self.execute_sliding_window(birds_eye_view)

        output_frame = frame.copy()

        # Detection & Polynomial Fitting
        if left_x is not None and len(left_x) > 300:
            left_poly = np.polyfit(left_y, left_x, 2)
            right_poly = np.polyfit(right_y, right_x, 2)
            self.last_left_poly = left_poly
            self.last_right_poly = right_poly
        else:
            left_poly = self.last_left_poly
            right_poly = self.last_right_poly

        # Control & Visualization
        if left_poly is not None and right_poly is not None:
            plot_y = np.linspace(0, height - 1, height)
            left_fit_x = left_poly[0]*plot_y**2 + left_poly[1]*plot_y + left_poly[2]
            right_fit_x = right_poly[0]*plot_y**2 + right_poly[1]*plot_y + right_poly[2]

            lookahead_index = int(len(plot_y) * 0.85)
            raw_lane_center = (left_fit_x[lookahead_index] + right_fit_x[lookahead_index]) / 2
            smoothed_lane_center = self.kalman_filter.update(raw_lane_center)

            car_center = width / 2
            cross_track_error = car_center - smoothed_lane_center

            steering_angle = self.pid_controller.compute(cross_track_error)
            target_speed = max(0.3, 1 - abs(steering_angle))
            self.last_steering_angle = steering_angle

            # Draw the lane polygon
            pts_left = np.transpose(np.vstack([left_fit_x, plot_y]))
            pts_right = np.transpose(np.vstack([right_fit_x, plot_y]))
            pts_right = pts_right[::-1]  
            polygon_points = np.int32(np.vstack([pts_left, pts_right]))

            lane_overlay = np.zeros_like(frame)
            cv2.fillPoly(lane_overlay, [polygon_points], (0, 255, 0))

            unwarped_overlay = cv2.warpPerspective(lane_overlay, inverse_matrix, (width, height))
            output_frame = cv2.addWeighted(frame, 1, unwarped_overlay, 0.4, 0)

            # Telemetry Display
            cv2.putText(output_frame, f"Steering: {steering_angle:.4f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(output_frame, f"Speed: {target_speed:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        else:
            # Recovery Mode
            cv2.putText(output_frame, "RECOVERY MODE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Developer Debug Views (Optional, can be commented out for production)
        cv2.imshow("Debug: Binary Mask", binary_mask)
        cv2.imshow("Debug: Bird's Eye View", birds_eye_view)

        return output_frame


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    video_path = "LaneVideo.mp4"
    cap = cv2.VideoCapture(video_path)
    tracker = AutonomousLaneTracker()

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        final_result = tracker.process_frame(current_frame)
        cv2.imshow("Autonomous Lane Tracking System", final_result)

        if cv2.waitKey(20) == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
