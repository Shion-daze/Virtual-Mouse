import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080
SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1920
SENSITIVITY = 1.0
x_scale_factor = SCREEN_WIDTH / CAMERA_WIDTH * SENSITIVITY
y_scale_factor = SCREEN_HEIGHT / CAMERA_HEIGHT * SENSITIVITY
SCROLL_SENSITIVITY = 10000
SCROLL_THRESHOLD = 0.05
y_index_previous = 0
y_middle_previous = 0
click_threshold = 0.02

mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

y_hand_previous = 0
current_mode = "cursor"
control_enabled = False  # Initially, no user has control

def is_index_finger_up(landmark):
    return (landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)

def is_hand_open(landmark):
    return all([
        landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ])

def is_peace_sign(landmark):
    return (landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmark[mp_hands.HandLandmark.PINKY_PIP].y)

def is_thumb_up(landmark):
    # Thumb up gesture for passing control
    return landmark[mp_hands.HandLandmark.THUMB_TIP].y < landmark[mp_hands.HandLandmark.THUMB_IP].y

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)
    hands = results.multi_hand_landmarks

    if hands:
        hand = hands[0]
        landmark = hand.landmark

        if is_thumb_up(landmark):
            control_enabled = not control_enabled  # Toggle control on thumb up
            continue

        if control_enabled:
            if is_index_finger_up(landmark):
                current_mode = "cursor"
            elif is_hand_open(landmark):
                current_mode = "scroll"
            elif is_peace_sign(landmark):
                current_mode = "click"

            if current_mode == "cursor":
                index_landmark = landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_landmark.x * CAMERA_WIDTH * x_scale_factor)
                index_y = int(index_landmark.y * CAMERA_HEIGHT * y_scale_factor)
                pyautogui.moveTo(index_x, index_y)

            elif current_mode == "click":
                y_index_current = landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                y_middle_current = landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                if y_index_previous - y_index_current > click_threshold:
                    pyautogui.click(button='left')
                elif y_middle_previous - y_middle_current > click_threshold:
                    pyautogui.click(button='right')
                y_index_previous = y_index_current
                y_middle_previous = y_middle_current

            elif current_mode == "scroll":
                y_hand_current = landmark[mp_hands.HandLandmark.WRIST].y
                scroll_speed = (y_hand_previous - y_hand_current) * SCROLL_SENSITIVITY
                if abs(scroll_speed) > SCROLL_THRESHOLD:
                    pyautogui.scroll(int(scroll_speed))
                y_hand_previous = y_hand_current

    # Display whether control is enabled
    control_status = "Enabled" if control_enabled else "Disabled"
    cv2.putText(frame, f'Control: {control_status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Mode: {current_mode}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Virtual Controller', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
