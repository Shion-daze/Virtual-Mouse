import cv2
import mediapipe as mp
import pyautogui

# 解像度の設定
MONITOR_WIDTH, MONITOR_HEIGHT = 2560, 1920
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 状態遷移の変数
is_active = False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        hand = hands[0]  # 最初に検出された手の情報を使用します
        landmarks = hand.landmark
        
        # 人差し指と親指のランドマーク取得
        index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

        # 人差し指だけを立てているかの判定
        if (index_tip.y < middle_tip.y) and (index_tip.y < ring_tip.y) and (index_tip.y < pinky_tip.y):
            is_active = True
            cv2.putText(frame, 'Active', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            is_active = False
            cv2.putText(frame, 'Inactive', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if is_active:
            # モニタの解像度に合わせて、カメラの映像の座標を変換する
            index_x = int(index_tip.x * MONITOR_WIDTH)
            index_y = int(index_tip.y * MONITOR_HEIGHT)

            # マウスの移動処理
            pyautogui.moveTo(MONITOR_WIDTH - index_x, index_y)  # X座標は反転させる

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
