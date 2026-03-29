import cv2
import time
from ultralytics import YOLO

# Load the YOLO model once (outside the function for better performance)
model = YOLO("yolov8n.pt")

# Global variables for alert throttling
last_alert = 0
ALERT_INTERVAL = 10  # seconds between alerts

ALERT_OBJECTS = ["person", "knife", "cell phone"]


def detect_objects(frame, chat_id):
    """
    Detect objects using YOLOv8 and send Telegram alert if dangerous object is detected.
    
    Args:
        frame: Input image frame from camera (BGR format)
        chat_id: Telegram chat ID to send alerts
    
    Returns:
        Annotated frame with bounding boxes (BGR)
    """
    global last_alert

    if frame is None:
        return None

    # Run YOLO inference
    results = model(frame, conf=0.4, verbose=False)

    detected_alert_object = None

    # Check for alert-worthy objects
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if label in ALERT_OBJECTS:
                detected_alert_object = label
                break  # No need to check further boxes

        if detected_alert_object:
            break  # No need to check further results

    # Send Telegram alert if needed (with cooldown)
    if detected_alert_object:
        current_time = time.time()
        
        if current_time - last_alert > ALERT_INTERVAL:
            try:
                # Save image for sending
                image_path = "detected.jpg"
                cv2.imwrite(image_path, frame)

                # Send alerts
                send_telegram_alert(chat_id, f"🚨 Alert: {detected_alert_object.upper()} detected!")
                send_telegram_image(chat_id, image_path)

                last_alert = current_time
                print(f"Alert sent: {detected_alert_object}")  # For debugging

            except Exception as e:
                print(f"Failed to send Telegram alert: {e}")

    # Return the annotated frame (with all detections drawn)
    return results[0].plot()


# Optional: Function to reset alert timer (useful on restart)
def reset_alert_timer():
    global last_alert
    last_alert = 0
