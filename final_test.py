#インポート
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mysort import Sort

# pyautoguiの初期設定
pyautogui.FAILSAFE = False

#Mediapipeの初期設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=8)

#SORTの導入
tracker = Sort()

#イニシアチブ取得のための関数
current_initiative_id = None
thumb_up_start_time = {}
thumb_up_duration = {}

#カメラの解像度　スクリーンの設定
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1440

#カーソルのマッピングの係数
x_scale_factor = SCREEN_WIDTH / CAMERA_WIDTH
y_scale_factor = SCREEN_HEIGHT / CAMERA_HEIGHT

#カメラの起動設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)


click_threshold = 0.05

scroll_threshold = 0.1
scroll_sensitivity = 8000

y_index_previous = 0
y_middle_previous = 0
y_hand_previous = 0

click_mode_counter = 0
CLICK_MODE_DURATION = 30

# カーソルの感度設定
cursor_sensitivity = 1  # 例: 0.5は通常の半分の速さ

current_mode = "cursor"

# サムズアップ検出関数
def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    other_fingers_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    return all(thumb_tip.y < finger_tip.y for finger_tip in other_fingers_tips)

# モード切替関数
def is_index_finger_up(landmark):
    index_finger_tip_y = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_finger_pip_y = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_tip_y = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    return index_finger_tip_y < index_finger_pip_y and middle_finger_tip_y > index_finger_pip_y

def is_hand_open(landmark):
    return all([
        landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmark.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmark.landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmark.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ])

def is_peace_sign(landmark):
    return (
        landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmark.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        landmark.landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmark.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    )

# メインループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame_height, frame_width, _ = frame.shape

    # 手の検出とバウンディングボックスの計算
    detections = []
    hand_id_to_landmarks = {}
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bbox_cords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            x_min = min(bbox_cords, key=lambda item: item[0])[0] * frame_width
            y_min = min(bbox_cords, key=lambda item: item[1])[1] * frame_height
            x_max = max(bbox_cords, key=lambda item: item[0])[0] * frame_width
            y_max = max(bbox_cords, key=lambda item: item[1])[1] * frame_height
            detections.append([x_min, y_min, x_max, y_max])
            hand_id_to_landmarks[len(detections) - 1] = hand_landmarks

    # SORTによる追跡
    tracked_objects = tracker.update(np.array(detections)) if detections else tracker.update(np.empty((0, 5)))

    # イニシアチブの更新と操作
    for i, obj in enumerate(tracked_objects):
        x_min, y_min, x_max, y_max, track_id = obj.astype(int)
        if i in hand_id_to_landmarks:
            if is_thumb_up(hand_id_to_landmarks[i]):
                if track_id not in thumb_up_start_time:
                    thumb_up_start_time[track_id] = cv2.getTickCount()
                thumb_up_duration[track_id] = (cv2.getTickCount() - thumb_up_start_time[track_id]) / cv2.getTickFrequency()
                if thumb_up_duration[track_id] >= 1.5:  # 1.5秒以上サムズアップが続いた場合
                    current_initiative_id = track_id
            else:
                thumb_up_start_time.pop(track_id, None)
                thumb_up_duration.pop(track_id, None)

            # モード切替
            if track_id == current_initiative_id:
                if is_index_finger_up(hand_id_to_landmarks[i]):
                    current_mode = "cursor"
                elif is_hand_open(hand_id_to_landmarks[i]):
                    current_mode = "scroll"
                elif is_peace_sign(hand_id_to_landmarks[i]):
                    current_mode = "click"

        # イニシアチブを持つ手が操作中の場合、操作を実行
        if track_id == current_initiative_id and i in hand_id_to_landmarks:
            index_landmark = hand_id_to_landmarks[i].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_landmark = hand_id_to_landmarks[i].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist_landmark = hand_id_to_landmarks[i].landmark[mp_hands.HandLandmark.WRIST]
            
         # カーソルモード
            if current_mode == "cursor":
                index_landmark = hand_id_to_landmarks[i].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
                # スクリーン解像度に基づいて座標を計算し、感度係数を適用
                screen_x = int(index_landmark.x * SCREEN_WIDTH * cursor_sensitivity)
                screen_y = int(index_landmark.y * SCREEN_HEIGHT * cursor_sensitivity)
    
                # マウスカーソルを移動
                pyautogui.moveTo(screen_x, screen_y)


            # クリックモード
            elif current_mode == "click":
                y_index_current = index_landmark.y
                y_middle_current = middle_landmark.y
                
                if click_mode_counter > 0:
                    if y_index_previous - y_index_current > click_threshold:
                        pyautogui.click(button='left')
                    elif y_middle_previous - y_middle_current > click_threshold:
                        pyautogui.click(button='right')
                        
                y_index_previous = y_index_current
                y_middle_previous = y_middle_current

            # スクロールモード
            elif current_mode == "scroll":
                y_hand_current = wrist_landmark.y
                scroll_speed = (y_hand_previous - y_hand_current) * scroll_sensitivity
                if abs(scroll_speed) > scroll_threshold:
                    pyautogui.scroll(int(scroll_speed))
                y_hand_previous = y_hand_current

        # ラベルの描画
        label = f'ID: {track_id}'
        if track_id == current_initiative_id:
            label += f' (Initiative)'
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
