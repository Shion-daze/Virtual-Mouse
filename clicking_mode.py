import pyautogui
import cv2
import mediapipe as mp

pyautogui.FAILSAFE = False

# Initialize mediapipe hands
mp_hands = mp.solutions.hands

# Variables for tracking click state
y_index_previous = 0
y_middle_previous = 0
click_threshold = 0.02

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

        click_mode_activated = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check for peace sign
                thumb_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
                index_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_finger_closed = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_finger_closed = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

                if thumb_is_open and index_is_open and middle_is_open and ring_finger_closed and pinky_finger_closed:
                    click_mode_activated = True

                y_index_current = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                y_middle_current = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                if click_mode_activated:
                    # If index finger moved down and passed the threshold, simulate a left click
                    if y_index_previous - y_index_current > click_threshold:
                        pyautogui.click(button='left')
                        
                    # If middle finger moved down and passed the threshold, simulate a right click
                    elif y_middle_previous - y_middle_current > click_threshold:
                        pyautogui.click(button='right')

                # Update the previous y values
                y_index_previous = y_index_current
                y_middle_previous = y_middle_current

            if click_mode_activated:
                cv2.putText(frame, 'ACTIVE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'INACTIVE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
