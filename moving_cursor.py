import cv2 
import mediapipe as mp
import pyautogui

# カメラの解像度
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080

# モニタの解像度
SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1920

# 感度調整
SENSITIVITY = 1.0

# 感度の計算
x_scale_factor = SCREEN_WIDTH / CAMERA_WIDTH * SENSITIVITY
y_scale_factor = SCREEN_HEIGHT / CAMERA_HEIGHT * SENSITIVITY

# カメラの設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Mediapipeの手の検出モデルの初期化
hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        hand = hands[0]
        landmarks = hand.landmark
        index_landmark = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        
        # カメラの解像度からモニタの解像度への変換
        index_x = int(index_landmark.x * CAMERA_WIDTH * x_scale_factor)
        index_y = int(index_landmark.y * CAMERA_HEIGHT * y_scale_factor)
        
        # マウスの移動処理
        pyautogui.moveTo(index_x, index_y)
    
    cv2.imshow('Virtual Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
