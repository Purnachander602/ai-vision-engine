import time
import cv2
import streamlit as st
from ultralytics import YOLO
from telegram_notify import send_telegram_alert, send_telegram_image


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

last_alert_time = 0
ALERT_INTERVAL = 10


def detect_objects(frame, chat_id=None):

    global last_alert_time

    results = model(frame, conf=0.4)

    detected = None

    alert_objects = ["person", "knife", "cell phone"]

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            label = model.names[int(box.cls)]

            if label in alert_objects:
                detected = label
                break

        if detected:
            break


    if detected:

        current_time = time.time()

        if current_time - last_alert_time > ALERT_INTERVAL:

            image_path = "detected.jpg"
            cv2.imwrite(image_path, frame)

            if chat_id:
                send_telegram_alert(chat_id, f"🚨 {detected} detected")
                send_telegram_image(chat_id, image_path)

            last_alert_time = current_time


    try:
        annotated = results[0].plot()
    except:
        annotated = frame

    return annotated
