import cv2
import numpy as np
import time

def detect_turn():
    # Set the region of interest (ROI) coordinates
    roi_top = 300
    roi_bottom = 500
    roi_left = 0
    roi_right = 800
    
    # Set the threshold values for lane detection
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    
    # Open the video capture (change the index if necessary)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame to the defined ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Convert the ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Perform edge detection using the Canny algorithm
        edges = cv2.Canny(blur, 50, 150)
        
        # Define a region of interest mask
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, roi_bottom - roi_top),
                             (roi_right - roi_left, roi_bottom - roi_top),
                             (roi_right - roi_left, roi_bottom),
                             (0, roi_bottom)]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        
        # Apply the mask to the edges image
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines using the Hough transform
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
        
        # Check if any lines were detected
        if lines is not None:
            # Loop through all the detected lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate the slope of the line
                slope = (y2 - y1) / (x2 - x1)
                
                # Check if the slope indicates a left or right turn
                if slope > 0.5:
                    print("Turn detected: RIGHT")
                elif slope < -0.5:
                    print("Turn detected: LEFT")
        
        # If no lines were detected, assume no turn
        print("No turn detected")
        
        # Display the frame with the ROI and detected lines
        cv2.imshow('Lane Detection', frame)
        
        # Wait for 1 second and check for key press
        if cv2.waitKey(1000) == ord('q'):
            break
    
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start lane detection
detect_turn()