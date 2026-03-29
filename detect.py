import time
import cv2
import streamlit as st
from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
return YOLO("yolov8n.pt")   # lightweight model

model = load_model()

# ---------------- GLOBAL TIMERS ----------------

last_alert_time = 0
last_detection_time = 0

ALERT_INTERVAL = 10       # seconds between telegram alerts
DETECTION_INTERVAL = 0.5  # seconds between detections

# ---------------- DETECTION FUNCTION ----------------

def detect_objects(frame, chat_id=None):

global last_alert_time
global last_detection_time

current_time = time.time()

# skip detection if called too quickly
if current_time - last_detection_time < DETECTION_INTERVAL:
    return frame

last_detection_time = current_time

detected_label = None

# run YOLO detection
results = model(frame, conf=0.4)

# objects that trigger alerts
alert_objects = ["person", "knife", "cell phone"]

for r in results:

    if r.boxes is None:
        continue

    for box in r.boxes:

        cls_id = int(box.cls)
        label = model.names[cls_id]

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
                f"🚨 ALERT: {detected_label} detected"
            )

            send_telegram_image(chat_id, image_path)

        last_alert_time = current_time


# ---------------- DRAW BOXES ----------------

try:
    annotated_frame = results[0].plot()
except:
    annotated_frame = frame

return annotated_frame
