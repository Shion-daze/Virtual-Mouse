import pyautogui
import cv2
import mediapipe as mp

pyautogui.FAILSAFE = False

# Initialize mediapipe hands
mp_hands = mp.solutions.hands

# Maintain a queue to store recent Y coordinates
y_queue = []
click_detected = False  # Flag to track if click has been detected
click_threshold = 0.02  # Set the click threshold

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Skipping empty frame.")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract y-coordinate of index finger tip
                y_val = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                # Update the queue of recent Y coordinates
                y_queue.append(y_val)
                if len(y_queue) > 2:  # Consider the last 2 frames
                    y_queue.pop(0)

                if len(y_queue) == 2:  # Only make a prediction when we have 2 frames
                    diff = y_queue[1] - y_queue[0]  # Difference between the two frames

                    # If the difference exceeds the threshold, simulate a click
                    if diff > click_threshold and not click_detected:
                        pyautogui.click()
                        click_detected = True
                    elif diff <= click_threshold and click_detected:
                        click_detected = False

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
