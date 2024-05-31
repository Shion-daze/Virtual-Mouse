import cv2
import numpy as np
import mediapipe as mp
from mysort import Sort

# Initialize MediaPipe and SORT
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
tracker = Sort(max_age=10, min_hits=5, iou_threshold=0.3)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def landmarks_to_bbox(landmarks):
    x_min = min([lm.x for lm in landmarks]) * 1920
    y_min = min([lm.y for lm in landmarks]) * 1080
    x_max = max([lm.x for lm in landmarks]) * 1920
    y_max = max([lm.y for lm in landmarks]) * 1080
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def is_fist(landmarks, wrist, threshold=0.1):
    distances = [
        np.linalg.norm(np.array([landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP.value].x, landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP.value].y]) - wrist),
        np.linalg.norm(np.array([landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value].x, landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value].y]) - wrist),
        np.linalg.norm(np.array([landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP.value].x, landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP.value].y]) - wrist),
        np.linalg.norm(np.array([landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP.value].x, landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP.value].y]) - wrist),
        np.linalg.norm(np.array([landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP.value].x, landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP.value].y]) - wrist)
    ]

    # Check if all distances are below the threshold, indicating a fist
    return all(distance < threshold for distance in distances)

current_controller_id = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb_frame)

    sort_input = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bbox = landmarks_to_bbox(hand_landmarks.landmark)
            sort_input.append(bbox)

    tracked_hands = tracker.update(np.array(sort_input)) if sort_input else tracker.update(np.empty((0, 5)))

    for tracked_hand in tracked_hands:
        track_id, bbox = int(tracked_hand[4]), tracked_hand[:4]
        bbox = [int(coord) for coord in bbox]

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Find the matching hand landmarks
        matching_hand_landmarks = None
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            hand_bbox = landmarks_to_bbox(landmarks)
            if bbox[0] <= hand_bbox[0] <= bbox[2] and bbox[1] <= hand_bbox[1] <= bbox[3]:
                matching_hand_landmarks = landmarks
                break

        if matching_hand_landmarks:
            wrist_pos = np.array([matching_hand_landmarks[mp.solutions.hands.HandLandmark.WRIST.value].x, matching_hand_landmarks[mp.solutions.hands.HandLandmark.WRIST.value].y])
            if is_fist(matching_hand_landmarks, wrist_pos):
                cv2.putText(frame, "Fist Detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if current_controller_id is None or current_controller_id != track_id:
                    current_controller_id = track_id
                    print(f"Control taken by hand with ID: {current_controller_id}")

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
