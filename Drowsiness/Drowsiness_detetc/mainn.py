import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import imutils
import time
import threading
from playsound import playsound

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye and mouth landmarks for Mediapipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 82, 87, 13, 311, 308, 317, 14]  # Landmarks for mouth detection

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.75  # Yawning detection threshold
MOUTH_AR_CONSEC_FRAMES = 15
BLINK_THRESH = 5

COUNTER = 0
MOUTH_COUNTER = 0
TOTAL_BLINKS = 0

ALARM_PATH = r"C:\Users\JHANSI GUMMALLA\Music\objdetect\Drowsiness_detetc\alarm.mp3"

# Audio alert function with exception handling
def sound_alarm(path):
    try:
        playsound(path)
    except Exception as e:
        print(f"Error playing sound: {e}")

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    mar = (A + B) / (2.0 * C)
    return mar

# Start video capture
cap = cv2.VideoCapture(0)

# Flags to prevent alarm spamming
drowsy_alarm_triggered = False
yawn_alarm_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye coordinates
            left_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                         int(face_landmarks.landmark[i].y * frame.shape[0])) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                          int(face_landmarks.landmark[i].y * frame.shape[0])) for i in RIGHT_EYE]
            
            # Get mouth coordinates
            mouth = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                      int(face_landmarks.landmark[i].y * frame.shape[0])) for i in MOUTH]

            # Calculate EAR and MAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Draw eye and mouth landmarks
            for (x, y) in left_eye + right_eye + mouth:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Blink detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= BLINK_THRESH:
                    TOTAL_BLINKS += 1
                COUNTER = 0

            # Drowsiness detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not drowsy_alarm_triggered:
                    drowsy_alarm_triggered = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Play alarm sound in a separate thread
                    threading.Thread(target=sound_alarm, args=(ALARM_PATH,)).start()

            else:
                COUNTER = 0
                drowsy_alarm_triggered = False

            # Yawning detection
            if mar > MOUTH_AR_THRESH:
                MOUTH_COUNTER += 1
                if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES and not yawn_alarm_triggered:
                    yawn_alarm_triggered = True
                    cv2.putText(frame, "YAWN ALERT!", (150, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Play alarm sound in a separate thread
                    threading.Thread(target=sound_alarm, args=(ALARM_PATH,)).start()
            else:
                MOUTH_COUNTER = 0
                yawn_alarm_triggered = False

            # Display blink count
            cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Display output
    cv2.imshow("Drowsiness Detection System", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
