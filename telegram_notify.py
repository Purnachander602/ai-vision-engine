import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

def send_telegram_alert(chat_id, message):

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message
    }

    requests.post(url, data=data)


def send_telegram_image(chat_id, image_path):

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    files = {"photo": open(image_path, "rb")}

    data = {"chat_id": chat_id}

    requests.post(url, files=files, data=data)