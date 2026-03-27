from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image
import time
import cv2

# load YOLO model (fast nano model)
model = YOLO("yolov8n.pt")

last_alert_time = 0
last_detection_time = 0

ALERT_INTERVAL = 10      # seconds between telegram alerts
DETECTION_INTERVAL = 0.5 # seconds between YOLO runs


def detect_objects(frame, chat_id=None):

    global last_alert_time, last_detection_time

    current_time = time.time()

    # Skip detection if called too fast
    if current_time - last_detection_time < DETECTION_INTERVAL:
        return frame

    last_detection_time = current_time

    results = model(frame)

    detected_label = None

    for r in results:
        for box in r.boxes:

            label = model.names[int(box.cls)]

            alert_objects = ["person", "knife", "cell phone"]

            if label in alert_objects:
                detected_label = label
                break

    # Send Telegram alert (with cooldown)
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

    annotated_frame = results[0].plot()

    return annotated_frame