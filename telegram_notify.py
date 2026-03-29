import requests
from typing import Optional

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"   # ← Change this to your actual bot token

def send_telegram_alert(chat_id: str, message: str) -> bool:
    """
    Send a text message to a Telegram user/chat.
    Returns True if successful, False otherwise.
    """
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️ Warning: BOT_TOKEN is not set!")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"          # You can also use "MarkdownV2"
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()   # Raise error for bad status codes
        return response.json().get("ok", False)
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")
        return False


def send_telegram_image(chat_id: str, image_path: str, caption: Optional[str] = None) -> bool:
    """
    Send a photo to Telegram.
    Returns True if successful, False otherwise.
    """
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️ Warning: BOT_TOKEN is not set!")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }

            response = requests.post(url, data=data, files=files, timeout=15)
            response.raise_for_status()
            return response.json().get("ok", False)

    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Error sending Telegram image: {e}")
        return False
