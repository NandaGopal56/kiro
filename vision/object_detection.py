#!/usr/bin/env python3

import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent      # adjust if needed
MODEL_DIR = PROJECT_ROOT / ".models"

MODEL_NAME = "yolo11n.pt"
MODEL_PATH = MODEL_DIR / MODEL_NAME

CONFIDENCE = 0.4
IMAGE_SIZE = 320

CAMERA_INDEX = 0

# -----------------------------------------------------------------------------
# Ensure model exists
# -----------------------------------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    print("Downloading model...")

    # Ultralytics automatically downloads the model if it doesn't exist
    YOLO(MODEL_NAME)

    # Move downloaded model into .models
    if Path(MODEL_NAME).exists():
        Path(MODEL_NAME).rename(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    else:
        raise RuntimeError("Model download failed.")

print(f"Loading model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera.")

print("Press 'q' to quit.")

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False,
    )

    result = results[0]

    annotated = frame.copy()

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = model.names[cls]

        print(f"{label:15} {conf:.2f}")

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("YOLO11 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()