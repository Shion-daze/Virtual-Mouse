import cv2
import mediapipe as mp
import numpy as np
from sort import Sort

# Mediapipeの初期化
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection.FaceDetection()
tracker = Sort()

# カメラ設定
cap = cv2.VideoCapture(0)

# 主導権を握っているかの判定のための関数
def has_initiative(landmark):
    return landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y < \
           landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y

current_initiative = None  # 現在の主導権を持っているtrack_id
thumb_up_detected = set()  # 親指を上げたことを検出したtrack_idのセット

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = mp_hands.process(rgb_frame)
    face_results = mp_face_detection.process(rgb_frame)

    # 検出された顔と手のランドマークを取得
    detections = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 親指が上がっているかどうかをチェック
            if has_initiative(hand_landmarks.landmark):
                # 親指を上げたユーザーの情報を記録
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == mp.solutions.hands.HandLandmark.THUMB_TIP:
                        thumb_tip_x = int(lm.x * frame.shape[1])
                        thumb_tip_y = int(lm.y * frame.shape[0])
                        detections.append([thumb_tip_x - 10, thumb_tip_y - 10, thumb_tip_x + 10, thumb_tip_y + 10, 'thumb_up'])
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

    # Sortトラッカーを更新
    if detections:
        detections_np = np.array(detections)[:, :4]  # Sortにはバウンディングボックスの情報のみ必要
        track_bbs_ids = tracker.update(detections_np)
        for det in track_bbs_ids:
            bbox, track_id = det[:4], int(det[4])
            if 'thumb_up' in detections[int(track_id)][4]:
                if track_id not in thumb_up_detected:
                    current_initiative = track_id
                    thumb_up_detected.add(track_id)
            else:
                if track_id in thumb_up_detected:
                    thumb_up_detected.remove(track_id)
                    if current_initiative == track_id:  # 現在の主導権を持つtrack_idが親指を下げた場合、主導権をリセット
                        current_initiative = None

            # バウンディングボックスとトラックIDを描画
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # 主導権を持つユーザーのバウンディングボックスの上にテキストを表示
            if track_id == current_initiative:
                cv2.putText(frame, "Initiative", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
