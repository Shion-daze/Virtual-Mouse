import cv2

# カメラを開く
cap = cv2.VideoCapture(0)

# 最大幅と高さを取得
max_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
max_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"最大幅: {max_width}, 最大高さ: {max_height}")

# カメラを解放
cap.release()
