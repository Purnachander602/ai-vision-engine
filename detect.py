import cv2
import time
from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image

# Load model once
model = YOLO("yolov8n.pt")

# Alert settings
last_alert = 0
ALERT_INTERVAL = 10  # seconds
ALERT_OBJECTS = {"person", "knife", "cell phone"}


def detect_objects(frame, chat_id):
    global last_alert

    if frame is None or chat_id is None:
        return frame

    try:
        # YOLO detection
        results = model(frame, conf=0.4, verbose=False)
        result = results[0]

        detected_alert_object = None

        # Check for alert objects
        if result.boxes is not None:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                if label in ALERT_OBJECTS:
                    detected_alert_object = label
                    break

        # Send Telegram alert if needed
        if detected_alert_object:
            current_time = time.time()
            if current_time - last_alert > ALERT_INTERVAL:
                try:
                    image_path = "detected.jpg"
                    cv2.imwrite(image_path, frame)

                    send_telegram_alert(
                        chat_id, 
                        f"🚨 Alert: **{detected_alert_object.capitalize()}** detected!"
                    )
                    
                    send_telegram_image(
                        chat_id, 
                        image_path,
                        caption=f"🚨 {detected_alert_object.capitalize()} detected"
                    )

                    last_alert = current_time
                except Exception as e:
                    print(f"Telegram error: {e}")

        # Return frame with bounding boxes
        return result.plot()

    except Exception as e:
        print(f"Detection error: {e}")
        return frame
