import cv2
import mediapipe as mp
import pyautogui
import time

# 解像度の設定
MONITOR_WIDTH, MONITOR_HEIGHT = 2560, 1920
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080

# 感度調整
SENSITIVITY = 1.5
x_scale_factor = MONITOR_WIDTH / CAMERA_WIDTH * SENSITIVITY
y_scale_factor = MONITOR_HEIGHT / CAMERA_HEIGHT * SENSITIVITY

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for tracking click state
y_index_previous = 0
y_middle_previous = 0
click_threshold = 0.012
ignore_click_until = 0  # クリックを無視するタイムスタンプ

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)
    hands = results.multi_hand_landmarks

    if hands:
        hand = hands[0]
        landmarks = hand.landmark

        index_is_open = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_is_open = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

        if index_is_open and middle_is_open:
            y_index_current = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            y_middle_current = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

            if time.time() > ignore_click_until:
                if y_index_previous - y_index_current > click_threshold:
                    pyautogui.click(button='left')
                elif y_middle_previous - y_middle_current > click_threshold:
                    pyautogui.click(button='right')

            y_index_previous = y_index_current
            y_middle_previous = y_middle_current

            cv2.putText(frame, 'CLICK MODE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if index_is_open:
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_tip.x * CAMERA_WIDTH * x_scale_factor)
                index_y = int(index_tip.y * CAMERA_HEIGHT * y_scale_factor)

                pyautogui.moveTo(index_x, index_y)

                cv2.putText(frame, 'CURSOR MODE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ignore_click_until = time.time() + 0.5

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
