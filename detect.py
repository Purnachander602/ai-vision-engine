import cv2
import time
from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image

# Load model once at module level (important for performance)
model = YOLO("yolov8n.pt")

# Alert control
last_alert = 0
ALERT_INTERVAL = 10  # seconds

# Objects that should trigger alerts
ALERT_OBJECTS = {"person", "knife", "cell phone"}


def detect_objects(frame, chat_id):
    """
    Detect objects using YOLOv8 and send Telegram alert if dangerous object is found.
    
    Args:
        frame: numpy array (BGR image)
        chat_id: Telegram chat ID for alerts
    
    Returns:
        Annotated frame with bounding boxes
    """
    global last_alert

    if frame is None or chat_id is None:
        return frame

    try:
        # Run YOLO inference
        results = model(frame, conf=0.4, verbose=False)  # verbose=False to reduce console spam

        # Get the first (and usually only) result
        result = results[0]

        detected_alert_object = None

        # Check for alert-worthy objects
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]

                if label in ALERT_OBJECTS:
                    detected_alert_object = label
                    break

        # Send alert if dangerous object detected and cooldown has passed
        if detected_alert_object:
            current_time = time.time()

            if current_time - last_alert > ALERT_INTERVAL:
                try:
                    image_path = "detected.jpg"
                    
                    # Save current frame
                    cv2.imwrite(image_path, frame)

                    # Send alerts
                    send_telegram_alert(
                        chat_id, 
                        f"🚨 Alert: **{detected_alert_object.capitalize()}** detected!"
                    )
                    
                    send_telegram_image(
                        chat_id, 
                        image_path, 
                        caption=f"🚨 {detected_alert_object.capitalize()} detected at {time.strftime('%H:%M:%S')}"
                    )

                    last_alert = current_time

                except Exception as e:
                    print(f"Telegram alert failed: {e}")

        # Return annotated frame (with bounding boxes and labels)
        return result.plot()

    except Exception as e:
        print(f"Detection error: {e}")
        # Return original frame if detection fails
        return frame
    
