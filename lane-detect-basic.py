import cv2
import numpy as np
import mss
import mss.tools
import time


def detect_turn():
    # Set the monitor coordinates for the region of interest (ROI)
    monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
    
    # Set the threshold values for lane detection
    slope_threshold = 1.0  # Adjust this threshold for more extreme turns
    
    # Initialize turn flags
    turn_left = False
    turn_right = False

    with mss.mss() as sct:
        while True:
            # Capture the screen frame
            frame = np.array(sct.grab(monitor))
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply a Gaussian blur to the grayscale image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # Perform edge detection using the Canny algorithm
            edges = cv2.Canny(blur, 50, 150)
            
            # Define a region of interest mask
            mask = np.zeros_like(edges)
            polygon = np.array([[(0, monitor["height"]),
                                 (monitor["width"], monitor["height"]),
                                 (monitor["width"], 2 * monitor["height"] // 3),
                                 (0, 2 * monitor["height"] // 3)]], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            
            # Apply the mask to the edges image
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Detect lines using the Hough transform
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
            
            # Reset turn flags
            turn_left = False
            turn_right = False
            
            # Check if any lines were detected
            if lines is not None:
                # Loop through all the detected lines
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate the slope of the line
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Check if the slope indicates a left or right turn
                    if slope > slope_threshold:
                        turn_right = True
                    elif slope < -slope_threshold:
                        turn_left = True
            
            # Determine the detected turn
            if turn_left and not turn_right:
                print("Turn detected: LEFT")
            elif turn_right and not turn_left:
                print("Turn detected: RIGHT")
            else:
                print("No turn detected")
            
            # Display the frame with the ROI and detected lines
            cv2.imshow('Lane Detection', frame)
            
            # Wait for 0.5 second and check for key press
            if cv2.waitKey(500) == ord('q'):
                break
    
    # Close all windows
    cv2.destroyAllWindows()

# Call the function to start lane detection
detect_turn()