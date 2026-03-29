import cv2
import time
from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image

model = YOLO("yolov8n.pt")

last_alert = 0
ALERT_INTERVAL = 10
ALERT_OBJECTS = {"person", "knife", "cell phone"}


def detect_objects(frame, chat_id):
    global last_alert

    if frame is None or chat_id is None:
        return frame

    try:
        results = model(frame, conf=0.4, verbose=False)
        result = results[0]

        detected_alert_object = None

        if result.boxes is not None:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                if label in ALERT_OBJECTS:
                    detected_alert_object = label
                    break

        if detected_alert_object:
            current = time.time()
            if current - last_alert > ALERT_INTERVAL:
                try:
                    image_path = "detected.jpg"
                    cv2.imwrite(image_path, frame)

                    send_telegram_alert(chat_id, f"🚨 {detected_alert_object.capitalize()} detected!")
                    send_telegram_image(chat_id, image_path, 
                                      caption=f"🚨 {detected_alert_object.capitalize()} detected")

                    last_alert = current
                except Exception as e:
                    print(f"Telegram send failed: {e}")

        return result.plot()

    except Exception as e:
        print(f"Detection error: {e}")
        return frame
