import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
model = load_model("gesture_model.keras")

# Class names
class_names = [
    "PALM",
    "L",
    "FIST",
    "FIST_MOVED",
    "THUMB",
    "INDEX",
    "OK",
    "PALM_MOVED",
    "C",
    "DOWN"
]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                x_list.append(x)
                y_list.append(y)

            padding = 20

            x_min = max(min(x_list) - padding, 0)
            y_min = max(min(y_list) - padding, 0)

            x_max = min(max(x_list) + padding, w)
            y_max = min(max(y_list) + padding, h)

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            # Crop hand
            hand_crop = frame[
                y_min:y_max,
                x_min:x_max
            ]

            if hand_crop.size != 0:

                hand_crop = cv2.resize(
                    hand_crop,
                    (64, 64)
                )

                input_img = np.expand_dims(
                    hand_crop,
                    axis=0
                )

                prediction = model.predict(
                    input_img,
                    verbose=0
                )

                class_id = np.argmax(prediction)

                confidence = np.max(prediction)

                label = (
                    f"{class_names[class_id]} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    cv2.imshow(
        "Real Time Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()