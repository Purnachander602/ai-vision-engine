import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from auth import add_user, login_user, update_chat_id, get_chat_id
from detect import detect_objects

st.set_page_config(page_title="AI Vision Engine", layout="centered")

st.title("AI Vision Engine")

# ---------------- SESSION STATE ----------------

if "user" not in st.session_state:
    st.session_state["user"] = None

if "detect" not in st.session_state:
    st.session_state["detect"] = False

# ---------------- VIDEO PROCESSOR ----------------

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.chat_id = None
        self.frame_count = 0
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

    try:
        img = cv2.resize(img, (640, 480))

        # reduce lag → run detection every 5 frames
        self.frame_count += 1

        if st.session_state["detect"] and self.frame_count % 5 == 0:
            img = detect_objects(img, self.chat_id)

    except Exception as e:
        print("Detection error:", e)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- LOGIN PAGE ----------------

if st.session_state["user"] is None:


login_tab, signup_tab = st.tabs(["Login", "Signup"])

with login_tab:

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = login_user(email, password)

        if user:
            st.session_state["user"] = email
            st.success("Login successful")
            st.rerun()

        else:
            st.error("Invalid email or password")

with signup_tab:

    new_email = st.text_input("Email", key="signup_email")
    new_pass = st.text_input("Password", type="password", key="signup_pass")

    if st.button("Create Account"):

        created = add_user(new_email, new_pass)

        if created:
            st.success("Account created. Please login.")
        else:
            st.error("User already exists")


# ---------------- DASHBOARD ----------------

else:


user = st.session_state["user"]

st.success(f"Logged in as {user}")

if st.button("Logout"):
    st.session_state["user"] = None
    st.rerun()

st.divider()

# ---------------- TELEGRAM ----------------

st.subheader("Connect Telegram Notifications")

st.markdown("Get your chat id here → https://t.me/userinfobot")

chat_id = st.text_input("Enter Telegram Chat ID")

if st.button("Save Chat ID"):
    update_chat_id(user, chat_id)
    st.success("Chat ID saved")

saved_chat = get_chat_id(user)

# ---------------- CAMERA + DETECTION ----------------

if saved_chat:

    st.success("Telegram Connected")

    st.subheader("Live Camera")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Detection"):
            st.session_state["detect"] = True

    with col2:
        if st.button("Stop Detection"):
            st.session_state["detect"] = False

    ctx = webrtc_streamer(
        key="ai-vision-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": 640,
                "height": 480
            },
            "audio": False
        },
        async_processing=True
    )

    if ctx.video_processor:
        ctx.video_processor.chat_id = saved_chat

else:
    st.warning("Please connect Telegram Chat ID to enable alerts.")
