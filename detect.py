```python
from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image
import time
import cv2

# ---------------- LOAD MODEL ----------------

# lightweight model for faster detection
model = YOLO("yolov8n.pt")

# ---------------- GLOBAL TIMERS ----------------

last_alert_time = 0
last_detection_time = 0

# seconds between telegram alerts
ALERT_INTERVAL = 10

# seconds between YOLO detections
DETECTION_INTERVAL = 0.5


# ---------------- DETECTION FUNCTION ----------------

def detect_objects(frame, chat_id=None):

    global last_alert_time
    global last_detection_time

    current_time = time.time()

    # reduce frame size for faster inference
    frame = cv2.resize(frame, (640, 480))

    # skip detection if called too quickly
    if current_time - last_detection_time < DETECTION_INTERVAL:
        return frame

    last_detection_time = current_time

    # run YOLO detection
    results = model(frame, conf=0.4)

    detected_label = None

    # objects that trigger alerts
    alert_objects = ["person", "knife", "cell phone"]

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            label = model.names[int(box.cls)]

            if label in alert_objects:
                detected_label = label
                break

        if detected_label:
            break

    # ---------------- TELEGRAM ALERT ----------------

    if detected_label:

        if current_time - last_alert_time > ALERT_INTERVAL:

            image_path = "detected.jpg"
            cv2.imwrite(image_path, frame)

            if chat_id:

                send_telegram_alert(
                    chat_id,
                    f"🚨 {detected_label} detected"
                )

                send_telegram_image(chat_id, image_path)

            last_alert_time = current_time

    # draw bounding boxes
    annotated_frame = results[0].plot()

    return annotated_frame
```
