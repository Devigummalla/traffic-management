import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from twilio.rest import Client

# Twilio Configuration
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""
ALERT_PHONE_NUMBER = ""

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Parameters
MIN_COLLISION_FRAMES = 10     # Increased frames for sustained collision
IOU_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 50        # Min pixel distance for collision
ACCIDENT_DELAY_FRAMES = 30     # Buffer to delay accident detection (~1 sec at 30 FPS)
collision_frames = {}
trackers = {}  # Dictionary to store individual trackers

# Flag to track SMS alert and stop execution
accident_detected_once = False

# SMS Alert Function
def send_sms_alert():
    """Send SMS alert via Twilio"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Accident detected! Immediate attention required.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(f"✅ SMS Sent: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU)"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def centroid(box):
    """Get centroid of bounding box"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def detect_objects(frame):
    """Detect vehicles with YOLOv8"""
    results = model(frame)[0]
    boxes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))
    
    return boxes

def initialize_trackers(frame, boxes):
    """Initialize individual CSRT trackers"""
    global trackers
    trackers = {}  # Reset trackers

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        trackers[i] = {
            "tracker": tracker,
            "frames_since_collision": 0  # Track frames before accident detection
        }

def update_trackers(frame):
    """Update all individual trackers"""
    updated_boxes = []
    to_remove = []

    for i, data in trackers.items():
        tracker = data["tracker"]
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = map(int, box)
            updated_boxes.append((x, y, x + w, y + h))
        else:
            to_remove.append(i)

    # Remove failed trackers
    for idx in to_remove:
        del trackers[idx]

    return updated_boxes

def detect_accidents(frame, boxes, prev_boxes, frame_id):
    """Detect accidents with delayed detection buffer"""
    global accident_detected_once

    accident_detected = False

    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(prev_boxes):
            iou = calculate_iou(box1, box2)
            
            centroid1 = centroid(box1)
            centroid2 = centroid(box2)

            dist = distance.euclidean(centroid1, centroid2)

            if iou > IOU_THRESHOLD and dist < DISTANCE_THRESHOLD:
                collision_key = (i, j)

                # Initialize collision frames if not already present
                if collision_key not in collision_frames:
                    collision_frames[collision_key] = 0

                collision_frames[collision_key] += 1

                # Check if the collision persists for buffer frames
                if collision_frames[collision_key] >= ACCIDENT_DELAY_FRAMES:
                    accident_detected = True
                    cv2.putText(frame, "ACCIDENT DETECTED", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)

                    # ✅ Send SMS once and stop execution
                    if not accident_detected_once:
                        print("🚨 Accident detected! Sending SMS and stopping execution.")
                        send_sms_alert()
                        accident_detected_once = True
                        return frame, True  # Stop execution

            else:
                # Decrease frame count if no collision detected
                if (i, j) in collision_frames:
                    collision_frames[(i, j)] = max(0, collision_frames[(i, j)] - 1)

    return frame, accident_detected

def process_video(video_path, output_path):
    global accident_detected_once

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                          (int(cap.get(3)), int(cap.get(4))))

    prev_boxes = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Detect and initialize trackers every 10 frames
        if frame_id % 10 == 0:
            boxes = detect_objects(frame)
            initialize_trackers(frame, boxes)
        else:
            # Update the individual trackers
            boxes = update_trackers(frame)

        frame, accident_detected = detect_accidents(frame, boxes, prev_boxes, frame_id)

        if accident_detected:
            print("✅ Stopping execution after accident detection.")
            break  # Stop execution after first accident detection

        prev_boxes = boxes
        out.write(frame)

        cv2.imshow("Accident Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ Processing complete.")

# Paths
video_path = "acc.mp4"   # Input video file path
output_path = "accident_output.mp4"  # Output video path

# Run the video processing
process_video(video_path, output_path)
