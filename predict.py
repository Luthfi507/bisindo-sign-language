import json
import numpy as np
from pathlib import Path
import cv2

from tensorflow.keras.models import model_from_json # type: ignore
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarksConnections, HandLandmarkerOptions,
    drawing_utils
)

asset_dir = Path("assets")

with open(asset_dir / "label_map.json") as lab:
    label_map = json.load(lab)
label_map = {int(k): v for k, v in label_map.items()}

with open(asset_dir / "config.json") as f:
    config = json.load(f)

model = model_from_json(json.dumps(config))
model.load_weights(asset_dir / "model.weights.h5")

MAX_FRAMES = model.input_shape[1]
FEATURE_DIM = model.input_shape[2]
NUM_LANDMARKS = 21

base_options = mp.tasks.BaseOptions(model_asset_path=str(asset_dir / "hand_landmarker.task"))
hand_opt = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(hand_opt)

def extract_landmarks(detector, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = detector.detect(mp_image)
    hands = np.zeros((2, NUM_LANDMARKS, 3), dtype=np.float32)
    hand_present = len(results.hand_landmarks) > 0

    for i, hand_landmark in enumerate(results.hand_landmarks):
        drawing_utils.draw_landmarks(frame, hand_landmark, HandLandmarksConnections.HAND_CONNECTIONS)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmark], dtype=np.float32)
        wrist = coords[0].copy()
        coords -= wrist
        scale = np.linalg.norm(coords[9])
        if scale > 1e-6:
            coords /= scale
        hands[i] = coords

    return hands.flatten(), hand_present

NO_HAND_PATIENCE = 8       
MIN_GESTURE_FRAMES = 10    
CONFIDENCE_THRESHOLD = 0.6

cap = cv2.VideoCapture(0)

state = "IDLE"             
gesture_buffer = []        
no_hand_streak = 0         

label = None
conf = 0.0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, hand_present = extract_landmarks(detector, frame)

        if state == "IDLE":
            if hand_present:
                state = "RECORDING"
                gesture_buffer = [landmarks]
                no_hand_streak = 0

        elif state == "RECORDING":
            gesture_buffer.append(landmarks)

            if hand_present:
                no_hand_streak = 0
            else:
                no_hand_streak += 1

            gesture_ended = no_hand_streak >= NO_HAND_PATIENCE
            buffer_full = len(gesture_buffer) >= MAX_FRAMES  

            if gesture_ended or buffer_full:
                if len(gesture_buffer) >= MIN_GESTURE_FRAMES:
                    seq = np.array(gesture_buffer, dtype=np.float32)
                    if len(seq) < MAX_FRAMES:
                        pad = np.zeros((MAX_FRAMES - len(seq), FEATURE_DIM), dtype=np.float32)
                        seq = np.vstack([seq, pad])
                    else:
                        idx = np.linspace(0, len(seq) - 1, MAX_FRAMES).astype(int)
                        seq = seq[idx]

                    sequence = np.expand_dims(seq, axis=0)
                    pred = model.predict(sequence, verbose=0)[0]
                    conf = float(np.max(pred))
                    idx_pred = int(np.argmax(pred))

                    if conf >= CONFIDENCE_THRESHOLD:
                        label = label_map[idx_pred]
                    else:
                        label = "?"

                state = "IDLE"
                gesture_buffer = []
                no_hand_streak = 0

        status_color = (0, 255, 255) if state == "RECORDING" else (200, 200, 200)
        cv2.putText(
            frame, f"State: {state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2
        )

        if label:
            cv2.putText(
                frame, f"Label: {label} | Conf: {conf:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()