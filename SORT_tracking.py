import cv2
import numpy as np
from mysort import Sort
import mediapipe as mp

# MediaPipeの初期設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# SORTの初期設定
tracker = Sort()

# イニシアチブの管理
current_initiative_id = None

# サムズアップ検出関数
def is_thumb_up(hand_landmarks, width, height):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    other_fingers_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]

    return all(thumb_tip.y * height < finger_tip.y * height for finger_tip in other_fingers_tips)

# OpenCVでカメラをキャプチャ
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # MediaPipeによる手の検出
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 検出結果をSORTで使用する形式に変換
    detections = []
    thumb_up_detected = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bbox = hand_landmarks.landmark
            x_min = min([lm.x for lm in bbox]) * frame_width
            y_min = min([lm.y for lm in bbox]) * frame_height
            x_max = max([lm.x for lm in bbox]) * frame_width
            y_max = max([lm.y for lm in bbox]) * frame_height
            detections.append([x_min, y_min, x_max, y_max])

            # サムズアップ検出
            if is_thumb_up(hand_landmarks, frame_width, frame_height):
                thumb_up_detected = True

    # SORTによる追跡
    tracked_objects = []  # 追跡されたオブジェクトのリストを初期化
    if len(detections) > 0:
        np_detections = np.array(detections)
        tracked_objects = tracker.update(np_detections)

        # イニシアチブの更新
        if thumb_up_detected and tracked_objects.size > 0:
            current_initiative_id = int(tracked_objects[-1][4])
    else:
        tracked_objects = tracker.update(np.empty((0, 5)))

    # 追跡結果の描画
    for obj in tracked_objects:
        x_min, y_min, x_max, y_max, track_id = obj.astype(int)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # イニシアチブを持つIDの表示
        label = f'ID: {track_id}'
        if track_id == current_initiative_id:
            label += ' (Initiative)'
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
