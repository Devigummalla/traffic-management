import cv2
from tracker2 import *
import numpy as np

# Create Tracker Object
tracker = EuclideanDistTracker()

# Load the video
cap = cv2.VideoCapture("Resources/sample.mp4")

f = 25  # Frame rate
w = int(1000 / (f - 1))

# Background subtraction for object detection
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Kernels for morphological operations
kernalOp = np.ones((3, 3), np.uint8)
kernalCl = np.ones((11, 11), np.uint8)
kernal_e = np.ones((5, 5), np.uint8)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match the video dimensions
    frame = cv2.resize(frame, (848, 478))

    # Extract the region of interest (ROI)
    roi = frame[50:478, 0:848]

    # Background subtraction and masking
    fgmask = fgbg.apply(roi)
    _, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)

    # Contour detection
    contours, _ = cv2.findContours(e_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # Track objects
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        speed = tracker.get_speed(id)

        if speed > 0:
            tracker.capture(roi, x, y, h, w, id)

        # Display speed label and bounding box
        color = (0, 255, 0) if speed < limit else (0, 0, 255)
        label = f"{id} {speed} km/h"

        cv2.putText(roi, label, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 3)

    cv2.imshow("ROI", roi)

    if cv2.waitKey(w - 10) == 27:
        tracker.end()
        break

cap.release()
cv2.destroyAllWindows()
