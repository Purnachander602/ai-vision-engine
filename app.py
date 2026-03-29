import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from auth import add_user, login_user, update_chat_id, get_chat_id
from detect import detect_objects

# Page Configuration
st.set_page_config(
    page_title="AI Vision Engine",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ AI Vision Engine")

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "detect" not in st.session_state:
    st.session_state.detect = False

# ====================== VIDEO PROCESSOR ======================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.chat_id: str = None
        self.frame_count: int = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize for faster processing
        img = cv2.resize(img, (640, 480))

        self.frame_count += 1

        # Run detection every 5 frames when enabled
        if st.session_state.detect and self.frame_count % 5 == 0:
            try:
                img = detect_objects(img, self.chat_id)
            except Exception as e:
                st.error(f"Detection error: {e}")
                # Continue with original frame if detection fails

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ====================== AUTHENTICATION ======================
if st.session_state.user is None:
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", type="primary"):
            if email and password:
                user = login_user(email, password)
                if user:
                    st.session_state.user = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
            else:
                st.warning("Please enter email and password")

    with tab2:
        st.subheader("Create Account")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create Account", type="primary"):
            if email and password:
                if add_user(email, password):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("User with this email already exists")
            else:
                st.warning("Please fill all fields")

else:
    # ====================== MAIN APP (Logged In) ======================
    user = st.session_state.user

    st.success(f"✅ Logged in as: **{user}**")

    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.detect = False
        st.rerun()

    st.divider()

    st.subheader("📲 Telegram Alerts")
    st.markdown("Get your Chat ID from → [👉 @userinfobot](https://t.me/userinfobot)")

    chat_id_input = st.text_input("Enter your Telegram Chat ID", placeholder="123456789")

    if st.button("Save Chat ID", type="primary"):
        if chat_id_input.strip():
            update_chat_id(user, chat_id_input.strip())
            st.success("✅ Chat ID saved successfully!")
            st.rerun()
        else:
            st.error("Please enter a valid Chat ID")

    # Check saved chat ID
    saved_chat = get_chat_id(user)

    if saved_chat:
        st.success("✅ Telegram Connected")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ Start Detection", type="primary"):
                st.session_state.detect = True
                st.rerun()

        with col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.detect = False
                st.rerun()

        # Show current status
        status = "🟢 Running" if st.session_state.detect else "🔴 Stopped"
        st.info(f"Detection Status: **{status}**")

        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="camera",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )

        # Pass chat_id to processor
        if ctx.video_processor:
            ctx.video_processor.chat_id = saved_chat

    else:
        st.warning("⚠️ Please add your Telegram Chat ID above to enable alerts")

# Optional: Footer
st.caption("AI Vision Engine | Real-time Object Detection with Telegram Alerts")
