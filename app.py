import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from auth import add_user, login_user, update_chat_id, get_chat_id
from detect import detect_objects

# ========================== PAGE CONFIG & STYLING ==========================
st.set_page_config(
    page_title="AI Vision Engine",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: white; }
    h1 { text-align: center; color: #60a5fa; }
    video { 
        width: 100% !important; 
        border-radius: 12px; 
        border: 2px solid #334155; 
    }
    .stButton>button { width: 100%; }
    .success { color: #4ade80; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Vision Engine Surveillance")

# ========================== SESSION STATE ==========================
if "user" not in st.session_state:
    st.session_state.user = None
if "detect" not in st.session_state:
    st.session_state.detect = False

# ========================== VIDEO PROCESSOR ==========================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.chat_id: str = None
        self.frame_count: int = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))

        self.frame_count += 1

        if st.session_state.detect and self.frame_count % 5 == 0:
            try:
                img = detect_objects(img, self.chat_id)
            except Exception as e:
                print(f"Detection error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ========================== AUTHENTICATION ==========================
if st.session_state.user is None:
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", use_container_width=True):
            if email and password:
                user_data = login_user(email, password)
                if user_data:
                    st.session_state.user = email
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")
            else:
                st.warning("Please enter email and password")

    with tab2:
        st.subheader("Create New Account")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

        if st.button("Create Account", type="primary", use_container_width=True):
            if email and password:
                if add_user(email, password):
                    st.success("✅ Account created successfully! Please login.")
                else:
                    st.error("❌ User with this email already exists")
            else:
                st.warning("Please fill all fields")

else:
    # ========================== MAIN DASHBOARD ==========================
    user = st.session_state.user

    st.success(f"✅ Logged in as **{user}**")

    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.detect = False
        st.rerun()

    st.divider()

    col1, col2 = st.columns([3, 1])

    # ====================== CAMERA SECTION ======================
    with col1:
        st.subheader("📹 Live Surveillance Camera")

        # Camera selection preference
        camera_mode = st.radio(
            "Camera Preference",
            options=[
                "Built-in Laptop Webcam (Recommended)",
                "Any Available Camera"
            ],
            horizontal=True,
            key="camera_preference"
        )

        # Strong constraints to prefer laptop webcam
        if camera_mode == "Built-in Laptop Webcam (Recommended)":
            video_constraints = {
                "width": {"ideal": 1280, "min": 640},
                "height": {"ideal": 720, "min": 480},
                "facingMode": {"exact": "user"},   # This is the strongest preference
                "frameRate": {"ideal": 30, "min": 15}
            }
        else:
            video_constraints = {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "facingMode": "user"
            }

        ctx = webrtc_streamer(
            key="camera",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": video_constraints,
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )

        # Pass chat_id to the processor
        if ctx.video_processor:
            ctx.video_processor.chat_id = get_chat_id(user)

    # ====================== CONTROL PANEL ======================
    with col2:
        st.subheader("⚙️ Control Panel")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🟢 Start Detection", type="primary", use_container_width=True):
                st.session_state.detect = True
                st.rerun()
        with c2:
            if st.button("🔴 Stop Detection", type="secondary", use_container_width=True):
                st.session_state.detect = False
                st.rerun()

        # Detection Status
        status = "🟢 Running" if st.session_state.detect else "🔴 Stopped"
        st.info(f"**Detection Status:** {status}")

        st.divider()

        st.subheader("📩 Telegram Alerts")
        st.markdown("[Get your Chat ID → @userinfobot](https://t.me/userinfobot)")

        chat_id_input = st.text_input(
            "Enter Telegram Chat ID", 
            placeholder="123456789",
            key="chat_id_input"
        )

        if st.button("Save Chat ID", type="primary", use_container_width=True):
            if chat_id_input and chat_id_input.strip():
                update_chat_id(user, chat_id_input.strip())
                st.success("✅ Chat ID saved successfully!")
                st.rerun()
            else:
                st.error("Please enter a valid Chat ID")

        # Show Telegram status
        saved_chat = get_chat_id(user)
        if saved_chat:
            st.success("✅ Telegram Connected")
            st.caption(f"Connected to: `{saved_chat[:4]}...{saved_chat[-4:]}`")
        else:
            st.warning("⚠️ Telegram not connected - Alerts disabled")

    st.caption("AI Vision Engine | Real-time YOLOv8 Object Detection with Telegram Alerts")
