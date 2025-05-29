import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Gesture Detection Function
def detect_hand_gesture(image):
    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = ""
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            wrist = landmarks[0]

            if (thumb_tip.y < wrist.y and index_tip.y > wrist.y and middle_tip.y > wrist.y):
                text = "Yes 👍"
            elif (thumb_tip.y > wrist.y and index_tip.y > wrist.y and middle_tip.y > wrist.y):
                text = "No 👎"
            elif (thumb_tip.y < wrist.y and index_tip.y < wrist.y and pinky_tip.y < wrist.y and middle_tip.y > wrist.y and ring_tip.y > wrist.y):
                text = "I Love You 🤟"
            elif (index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y < wrist.y and pinky_tip.y < wrist.y and thumb_tip.x < index_tip.x):
                text = "Hello 👋"
            elif (index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y > wrist.y and pinky_tip.y > wrist.y):
                text = "Peace ✌️"
            elif abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
                text = "Okay 👌"
            elif (index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y < wrist.y and pinky_tip.y < wrist.y and abs(thumb_tip.x - wrist.x) > 0.1):
                text = "Stop 🖐️"
            elif (index_tip.y > wrist.y and middle_tip.y > wrist.y and ring_tip.y > wrist.y and pinky_tip.y > wrist.y):
                text = "Fist ✊"

            if text:
                cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    return image, text or "No hand detected"

# Gradio Interface
gr.Interface(
    fn=detect_hand_gesture,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=["image", "text"],
    live=True,
    title="Hand Gesture Detection",
    description="Show a gesture in your webcam to see it recognized!"
).launch()
