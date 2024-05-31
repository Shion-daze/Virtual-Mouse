import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mysort import Sort

# pyautoguiの初期設定
pyautogui.FAILSAFE = False

# Mediapipeの初期設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# SORTの導入
tracker = Sort()

# カメラとスクリーンの設定
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1440

# スケーリングファクターの計算
x_scale_factor = SCREEN_WIDTH / CAMERA_WIDTH
y_scale_factor = SCREEN_HEIGHT / CAMERA_HEIGHT

# カメラの初期設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# カーソルの感度設定
cursor_sensitivity = 1.0  # 例: 0.5は通常の半分の速さ

# メインループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            screen_x = int(index_landmark.x * SCREEN_WIDTH * cursor_sensitivity)
            screen_y = int(index_landmark.y * SCREEN_HEIGHT * cursor_sensitivity)

            # デバッグ情報の出力
            print(f"Screen X: {screen_x}, Screen Y: {screen_y}")

            # カーソルを移動
            pyautogui.moveTo(screen_x, screen_y)

    # 映像の表示
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
