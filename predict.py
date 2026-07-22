import json
import numpy as np
from pathlib import Path
import cv2
import os
import argparse
import sys
from loguru import logger

from tensorflow.keras.models import model_from_json  # type: ignore
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarksConnections, HandLandmarkerOptions,
    drawing_utils
)
from utils.tts import speech

def check_file_exists(path, desc="file"):
    if not Path(path).exists():
        logger.error(f"Missing {desc}: {path}")
        return False
    return True

def load_assets(asset_dir):
    # Check all required assets
    required_files = [
        ("label_map.json", "label map"),
        ("config.json", "model config"),
        ("model.weights.h5", "model weights"),
        ("hand_landmarker.task", "hand landmarker task"),
    ]
    for fname, desc in required_files:
        if not check_file_exists(asset_dir / fname, desc):
            sys.exit(1)
    # Load label map
    try:
        with open(asset_dir / "label_map.json") as lab:
            label_map = json.load(lab)
        label_map = {int(k): v for k, v in label_map.items()}
    except Exception as e:
        logger.error(f"Failed to load label_map.json: {e}")
        sys.exit(1)
    # Load model config
    try:
        with open(asset_dir / "config.json") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        sys.exit(1)
    # Load model
    try:
        model = model_from_json(json.dumps(config))
        model.load_weights(asset_dir / "model.weights.h5")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    # Check hand_landmarker.task
    if not check_file_exists(asset_dir / "hand_landmarker.task", "hand landmarker task"):
        sys.exit(1)
    return label_map, model

def check_env_var(var):
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        return False
    return True

def check_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera at index {index}")
        return False
    cap.release()
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Bisindo Sign Language Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Prediction confidence threshold (default: 0.6)")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    parser.add_argument("--no-log", action="store_true", help="Disable info logging")
    return parser.parse_args()

def extract_landmarks(detector, frame):
    """
    Extract normalized hand landmarks from a frame using MediaPipe.
    Returns (flattened_landmarks, hand_present).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = detector.detect(mp_image)
    NUM_LANDMARKS = 21
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

async def main():
    args = parse_args()
    asset_dir = Path("assets")

    if not check_camera(args.camera):
        sys.exit(1)

    # Check OpenAI API key if TTS is enabled
    if not args.no_tts and not check_env_var("OPENAI_API_KEY"):
        logger.error("TTS is enabled but OPENAI_API_KEY is missing.")
        sys.exit(1)

    label_map, model = load_assets(asset_dir)

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
    try:
        detector = HandLandmarker.create_from_options(hand_opt)
    except Exception as e:
        logger.error(f"Failed to initialize hand detector: {e}")
        sys.exit(1)

    NO_HAND_PATIENCE = 8
    MIN_GESTURE_FRAMES = 10
    CONFIDENCE_THRESHOLD = args.conf_threshold

    cap = cv2.VideoCapture(args.camera)
    state = "IDLE"
    gesture_buffer = []
    no_hand_streak = 0

    label = None
    conf = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            landmarks, hand_present = extract_landmarks(detector, frame)

            if state == "IDLE":
                if hand_present:
                    state = "RECORDING"
                    gesture_buffer = [landmarks]
                    no_hand_streak = 0
                    if not args.no_log:
                        logger.info("Hand detected. Start recording gesture.")

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
                        try:
                            pred = model.predict(sequence, verbose=0)[0]
                            conf = float(np.max(pred))
                            idx_pred = int(np.argmax(pred))
                            if conf >= CONFIDENCE_THRESHOLD:
                                label = label_map[idx_pred]
                                if not args.no_log:
                                    logger.info(f"Predicted: {label} (conf: {conf:.2f})")
                            else:
                                label = "?"
                                if not args.no_log:
                                    logger.info(f"Low confidence ({conf:.2f}), label unknown.")
                        except Exception as e:
                            logger.error(f"Prediction failed: {e}")
                            label = "?"
                    else:
                        if not args.no_log:
                            logger.info("Gesture too short, ignored.")

                    state = "IDLE"
                    gesture_buffer = []
                    no_hand_streak = 0

            status_color = (0, 255, 255) if state == "RECORDING" else (200, 200, 200)
            cv2.putText(
                frame, f"State: {state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2
            )

            if label:
                if not args.no_tts:
                    try:
                        await speech(label)
                    except Exception as e:
                        logger.error(f"TTS failed: {e}")
                cv2.putText(
                    frame, f"Label: {label} | Conf: {conf:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if not args.no_log:
                    logger.info("Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            detector.close()
        except Exception:
            pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())