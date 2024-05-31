import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time

# MediaPipeの手モジュールを初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# カメラを起動
cap = cv2.VideoCapture(0)

# 座標を格納するためのリスト
x_coordinates = [[] for _ in range(21)]  # 21個のランドマーク分のリストを用意
y_coordinates = [[] for _ in range(21)]

# 現在の時間を取得
start_time = time.time()

# リアルタイム処理開始
while time.time() - start_time < 5:  # 5秒間実行
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipeでの処理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 手のランドマークが検出された場合
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                x_coordinates[id].append(lm.x)
                y_coordinates[id].append(lm.y)

# カメラを解放
cap.release()

# グラフを表示
plt.figure(figsize=(10, 8))

# X座標のプロット
plt.subplot(2, 1, 1)
for i, coords in enumerate(x_coordinates):
    plt.plot(coords, label=f"{mp_hands.HandLandmark(i).name}_x")
plt.title('X coordinates over time')
plt.legend()

# Y座標のプロット
plt.subplot(2, 1, 2)
for i, coords in enumerate(y_coordinates):
    plt.plot(coords, label=f"{mp_hands.HandLandmark(i).name}_y")
plt.title('Y coordinates over time')
plt.legend()

plt.tight_layout()
plt.show()
