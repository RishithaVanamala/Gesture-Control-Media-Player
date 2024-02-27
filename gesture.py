import cv2
import numpy as np

# Function to calculate the angle between three points
def angle(pt1, pt2, pt3):
    a = np.linalg.norm(pt2 - pt3)
    b = np.linalg.norm(pt1 - pt3)
    c = np.linalg.norm(pt1 - pt2)
    return np.arccos((a**2 + c**2 - b**2) / (2 * a * c))

# Function to detect gestures based on the approximated contour
def detect_gestures(approximated_contour, frame):
    hull = cv2.convexHull(approximated_contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(approximated_contour, hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(approximated_contour[s][0])
                end = tuple(approximated_contour[e][0])
                far = tuple(approximated_contour[f][0])
                a = np.array(start)
                b = np.array(far)
                c = np.array(end)
                angle_value = angle(a, b, c)
                # Count fingers based on the angles of the defects
                if angle_value < np.pi / 2:  # angle less than 90 degrees, treated as a finger
                    cnt += 1
            if cnt > 0:
                return f"{cnt+1} fingers"
            else:
                return "1 finger"
    return "No gesture"

# Function to detect a thumbs up gesture
def is_thumbs_up(defects, approximated_contour):
    # Assuming the thumb is the highest point in the contour
    # This will depend on the orientation of the hand
    if defects is not None and len(defects) == 1:  # Typically, a thumbs up will have one defect
        s, e, f, d = defects[0, 0]
        start = tuple(approximated_contour[s][0])
        end = tuple(approximated_contour[e][0])
        far = tuple(approximated_contour[f][0])
        
        if start[1] < far[1]:  # If the start is above the defect point
            return True
    return False

# Function to detect an OK sign
def is_ok_sign(defects, approximated_contour):
    # Assuming the 'O' created by thumb and index finger is a notable trait of an OK sign
    # This will depend on the orientation of the hand
    if defects is not None and len(defects) >= 2:  # An OK sign will typically have two defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approximated_contour[s][0])
            end = tuple(approximated_contour[e][0])
            far = tuple(approximated_contour[f][0])
            # Check the distance between start and end points to infer a closed loop (circle)
            if np.linalg.norm(np.array(start) - np.array(end)) < 40:  # Threshold for the distance to decide on a loop
                return True
    return False

# Set up the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Preprocessing the frame for better contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    gesture = detect_gestures(approximated_contour, frame)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will correspond to the hand
        max_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour
        epsilon = 0.0005 * cv2.arcLength(max_contour, True)
        approximated_contour = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # Draw the largest contour and its approximation
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 3)
        cv2.drawContours(frame, [approximated_contour], -1, (255, 0, 0), 3)

        # Detect gestures
        hull = cv2.convexHull(approximated_contour, returnPoints=False)
        defects = cv2.convexityDefects(approximated_contour, hull)
        
        # Here we should first check for special gestures like "Thumbs Up" or "OK sign"
        # before calling the general gesture detection which counts the fingers.
        if is_thumbs_up(defects, approximated_contour):
            gesture = "Thumbs Up"
        elif is_ok_sign(defects, approximated_contour):
            gesture = "OK Sign"
        else:
            # If no special gestures detected, then check for general hand gestures
            gesture = detect_gestures(approximated_contour, frame)
        
        # Display the gesture on the screen
        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()