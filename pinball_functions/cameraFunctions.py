import cv2
import numpy as np
import os

class BallTracker:
    def __init__(self, camera_index=0, rewarp=False, recolor=False, resquare=False):
        """Initialize the BallTracker object."""

        self.ending = False
        self.cap = cv2.VideoCapture(camera_index)
        self.points = []
        self.warp_matrix = None
        self.selected_point = None
        self.background_color = (57, 178, 222)
        self.width = 640
        self.height = 1020
        self.square_size = 100  # Size of the plunger area
        self.x_offset = 900  # X offset for the window
        self.y_offset = 195  # Y offset for the window
        self.square_top_left = (160, 80)  # Top-left corner of the square
        self.ball_in_square = False  # Boolean to track if the ball is inside the square
        self.max_ball_size = 45  # Maximum size of the ball in pixels (w and h)
        self.ball_x = 0
        self.ball_y = 0
        
        # Default color range (hue, saturation, value)
        if os.path.exists('externalFiles/color_range.npy'):
            with open('externalFiles/color_range.npy', 'rb') as f:
                self.lower_hue, self.upper_hue, self.lower_saturation, self.upper_saturation, self.lower_value, self.upper_value = np.load(f)
        else:
            print("color_range.npy not found. Please run with rewarp=True to create it.")
            raise SystemExit("color_range.npy not found")

        # Rewarp if needed
        if rewarp:
            self.select_corners()
        
        # Otherwise, load the warp matrix from a file
        else:
            if os.path.exists('externalFiles/warp_matrix.npy'):
                with open('externalFiles/warp_matrix.npy', 'rb') as f:
                    self.warp_matrix = np.load(f)
            else:
                print("warp_matrix.npy not found. Please run with rewarp=True to create it.")
                raise SystemExit("warp_matrix.npy not found")

        # Recolor if needed
        if recolor:
            self.adjust_red_range()
        
        # Resquare if needed
        if resquare:
            self.select_square()

    def select_corners(self):
        """Select the corners of the pinball field to create a warp matrix."""

        cv2.namedWindow("Select Corners")
        frame = self.get_frame()
        clone = frame.copy()
        
        # Function to draw points on the frame
        def draw_points():
            temp = clone.copy()
            if len(self.points) > 0:
                for i, point in enumerate(self.points):
                    cv2.circle(temp, point, 5, (0, 255, 0), -1)
                if len(self.points) == 4:
                    cv2.polylines(temp, [np.array(self.points)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Select Corners", temp)
        
        # Function to handle mouse events
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, (px, py) in enumerate(self.points):
                    if abs(px - x) < 10 and abs(py - y) < 10:
                        self.selected_point = i
                        return
                if len(self.points) < 4:
                    self.points.append((x, y))
                draw_points()
            elif event == cv2.EVENT_LBUTTONUP:
                self.selected_point = None
            elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
                self.points[self.selected_point] = (x, y)
                draw_points()
        
        # Display the frame and set the mouse callback
        cv2.imshow("Select Corners", frame)
        cv2.setMouseCallback("Select Corners", click_event)
        cv2.waitKey(0)
        cv2.destroyWindow("Select Corners")
        
        # If we have 4 points, sort them and create the warp matrixs
        if len(self.points) == 4:
            self.points = sorted(self.points, key=lambda p: (p[1], p[0]))
            top_two = sorted(self.points[:2], key=lambda p: p[0])
            bottom_two = sorted(self.points[2:], key=lambda p: p[0])
            ordered_points = np.array([bottom_two[1], bottom_two[0], top_two[0], top_two[1]], dtype=np.float32)
            
            # Define the destination points for the perspective transform
            # These points correspond to the corners of the output image
            destination_points = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], dtype=np.float32)
            self.warp_matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
            
            # Save the warp matrix to a file
            with open('externalFiles/warp_matrix.npy', 'wb') as f:
                np.save(f, self.warp_matrix)
   
    def get_frame(self):
        """Get a frame from the camera and apply rotation and padding."""

        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.copyMakeBorder(frame, 75, 75, 75, 75, cv2.BORDER_CONSTANT, value=self.background_color)
        return frame

    def warp_frame(self, frame):
        """
        Warp the frame using the warp matrix.
        
        Args:
            frame: The input frame to warp.
        """

        if self.warp_matrix is not None:
            return cv2.warpPerspective(frame, self.warp_matrix, (self.width, self.height))
        return frame

    def adjust_red_range(self):
        """Adjust the red color range for the ball detection."""

        def nothing(x):
            pass
        
        # Create trackbars for the lower and upper HSV values
        cv2.namedWindow('Adjust Color Range')
        cv2.createTrackbar('Low H', 'Adjust Color Range', self.lower_hue, 180, nothing)
        cv2.createTrackbar('High H', 'Adjust Color Range', self.upper_hue, 180, nothing)
        cv2.createTrackbar('Low S', 'Adjust Color Range', self.lower_saturation, 255, nothing)
        cv2.createTrackbar('High S', 'Adjust Color Range', self.upper_saturation, 255, nothing)
        cv2.createTrackbar('Low V', 'Adjust Color Range', self.lower_value, 255, nothing)
        cv2.createTrackbar('High V', 'Adjust Color Range', self.upper_value, 255, nothing)
        
        cv2.namedWindow('Square Mask')  # Create a new window for the square mask

        while True:
            frame = self.get_frame() # Get a frame from the camera
            frame_warped = self.warp_frame(frame)  # Apply warping here
            hsv = cv2.cvtColor(frame_warped, cv2.COLOR_BGR2HSV)
            
            # Get trackbar positions for H, S, V ranges
            l_h = cv2.getTrackbarPos('Low H', 'Adjust Color Range')
            h_h = cv2.getTrackbarPos('High H', 'Adjust Color Range')
            l_s = cv2.getTrackbarPos('Low S', 'Adjust Color Range')
            h_s = cv2.getTrackbarPos('High S', 'Adjust Color Range')
            l_v = cv2.getTrackbarPos('Low V', 'Adjust Color Range')
            h_v = cv2.getTrackbarPos('High V', 'Adjust Color Range')
            
            # Update the color range based on the sliders
            lower_hue = l_h
            upper_hue = h_h
            
            # Hue range from 0 to High H and from Low H to 180
            mask1 = cv2.inRange(hsv, np.array([0, l_s, l_v]), np.array([upper_hue, h_s, h_v]))
            mask2 = cv2.inRange(hsv, np.array([lower_hue, l_s, l_v]), np.array([180, h_s, h_v]))
            
            mask = mask1 + mask2
            
            # Show selected pixels using bitwise_and
            selected_pixels = cv2.bitwise_and(frame_warped, frame_warped, mask=mask)
            
            # Concatenate the original warped image with the selected pixels side by side
            combined = np.hstack((frame_warped, selected_pixels))
            
            cv2.imshow('Adjust Color Range', combined)
            
            # Extract square from the original (non-warped) image
            square_top_left = self.square_top_left
            square_bottom_right = (square_top_left[0] + self.square_size, square_top_left[1] + self.square_size)
            square_region = frame[square_top_left[1]:square_bottom_right[1], square_top_left[0]:square_bottom_right[0]]
            
            # Convert square region to HSV
            hsv_square = cv2.cvtColor(square_region, cv2.COLOR_BGR2HSV)
            
            # Create mask for the square region
            mask1_square = cv2.inRange(hsv_square, np.array([0, l_s, l_v]), np.array([upper_hue, h_s, h_v]))
            mask2_square = cv2.inRange(hsv_square, np.array([lower_hue, l_s, l_v]), np.array([180, h_s, h_v]))
            mask_square = mask1_square + mask2_square
            
            # Show selected pixels in the square region using bitwise_and
            selected_pixels_square = cv2.bitwise_and(square_region, square_region, mask=mask_square)
            selected_pixels_square_resized = cv2.resize(selected_pixels_square, (self.square_size * 4, self.square_size * 4))
            
            cv2.imshow('Square Mask', selected_pixels_square_resized)  # Display the square mask
            
            # Break the loop if 'c' is pressed
            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.lower_hue, self.upper_hue = lower_hue, upper_hue
                self.lower_saturation, self.upper_saturation = l_s, h_s
                self.lower_value, self.upper_value = l_v, h_v

                # Save the color range to a file
                with open('externalFiles/color_range.npy', 'wb') as f:
                    np.save(f, np.array([lower_hue, upper_hue, l_s, h_s, l_v, h_v]))
                break
        cv2.destroyWindow('Adjust Color Range')
        cv2.destroyWindow('Square Mask')  # Destroy the square mask window

    def select_square(self):
        """Select square from the original image"""

        cv2.namedWindow('Select Square')
        frame = self.get_frame()
        
        def draw_square():
            # Draw the square based on the top-left corner and size
            top_left = self.square_top_left
            bottom_right = (top_left[0] + self.square_size, top_left[1] + self.square_size)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.square_top_left = (x, y)
                draw_square()

        cv2.setMouseCallback('Select Square', click_event)

        while True:
            frame = self.get_frame()
            draw_square()
            cv2.imshow('Select Square', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        cv2.destroyWindow('Select Square')

    def filter_contours_by_size(self, contours, max_size):
        """Filter contours based on their size."""

        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= max_size and h <= max_size:
                filtered_contours.append(contour)
        return filtered_contours

    def track_object(self):
        """
        Track the ball in the pinball field.
        This updates the ball_x, ball_y, and ball_in_square attributes.
        """

        # Create named windows and set properties once
        cv2.namedWindow('Game Over')
        cv2.setWindowProperty('Game Over', cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow('Game Over', self.x_offset + self.width, self.y_offset + self.height - self.square_size * 4)
        
        cv2.namedWindow('Warped Image')
        cv2.setWindowProperty('Warped Image', cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow('Warped Image', self.x_offset, self.y_offset)

        while not self.ending:
            frame = self.get_frame()
            frame_warped = self.warp_frame(frame)  # Apply warping here
            
            hsv_warped = cv2.cvtColor(frame_warped, cv2.COLOR_BGR2HSV)
            
            # Create mask for the warped frame
            mask_warped = cv2.inRange(hsv_warped, np.array([0, self.lower_saturation, self.lower_value]), np.array([self.upper_hue, self.upper_saturation, self.upper_value]))
            mask_warped += cv2.inRange(hsv_warped, np.array([self.lower_hue, self.lower_saturation, self.lower_value]), np.array([180, self.upper_saturation, self.upper_value]))

            # Extract square from the original (non-warped) image
            square_top_left = self.square_top_left
            square_bottom_right = (square_top_left[0] + self.square_size, square_top_left[1] + self.square_size)
            square_region = frame[square_top_left[1]:square_bottom_right[1], square_top_left[0]:square_bottom_right[0]]
            
            # Double the size of the square region
            doubled_square_region = cv2.resize(square_region, (self.square_size * 4, self.square_size * 4))

            # Convert square region to HSV
            hsv_square = cv2.cvtColor(doubled_square_region, cv2.COLOR_BGR2HSV)
            
            # Create mask for the square region
            mask_square = cv2.inRange(hsv_square, np.array([0, self.lower_saturation, self.lower_value]), np.array([self.upper_hue, self.upper_saturation, self.upper_value]))
            mask_square += cv2.inRange(hsv_square, np.array([self.lower_hue, self.lower_saturation, self.lower_value]), np.array([180, self.upper_saturation, self.upper_value]))

            # Find contours in the warped image mask
            contours_warped, _ = cv2.findContours(mask_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_square, _ = cv2.findContours(mask_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Track largest ball in warped image
            largest_contour_warped = max(contours_warped, key=cv2.contourArea) if contours_warped else None
            largest_contour_square = max(contours_square, key=cv2.contourArea) if contours_square else None
            
            # Initialize comparison and selection
            largest_contour = None
            if largest_contour_warped is not None and (largest_contour_square is None or cv2.contourArea(largest_contour_warped) > cv2.contourArea(largest_contour_square)):
                largest_contour = largest_contour_warped
                self.ball_in_square = False  # Ball is not in the square
                frame_to_draw = frame_warped  # Ball is in the warped image
            elif largest_contour_square is not None:
                largest_contour = largest_contour_square
                self.ball_in_square = True  # Ball is inside the square
                frame_to_draw = doubled_square_region  # Ball is in the doubled square image

            # Check if we have a valid largest_contour before drawing it
            if largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                self.ball_x = x + 0.5 * w
                self.ball_y = y + 0.5 * h
                
                cv2.rectangle(frame_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_to_draw, str((w, h)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the images in separate windows
            cv2.imshow('Warped Image', frame_warped)
            cv2.imshow('Game Over', doubled_square_region)  # Display the doubled square region
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("Ball Tracker Ended.")


# Example Usage
if __name__ == "__main__":
    tracker = BallTracker(rewarp=True, recolor=True, resquare=True)
    tracker.track_object()
