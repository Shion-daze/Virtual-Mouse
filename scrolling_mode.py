import cv2
import mediapipe as mp
import pyautogui
import math

# カメラの設定
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080

# スクロールの感度と閾値
SENSITIVITY = 10000
SCROLL_THRESHOLD = 0.05

# Mediapipe handsモデルの初期化
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 手が開いているかのチェック
def is_hand_open(landmark):
    return all([
        landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

y_hand_previous = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)
    hands = results.multi_hand_landmarks

    if hands:
        hand = hands[0]
        landmark = hand.landmark

        # スクロールモードのアクティベーション
        if is_hand_open(landmark):
            y_hand_current = landmark[mp_hands.HandLandmark.WRIST].y
            scroll_speed = (y_hand_previous - y_hand_current) * SENSITIVITY

            if abs(scroll_speed) > SCROLL_THRESHOLD:
                pyautogui.scroll(int(scroll_speed))

            y_hand_previous = y_hand_current
            cv2.putText(frame, 'SCROLL MODE ACTIVE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'SCROLL MODE INACTIVE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Virtual Scroll', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
