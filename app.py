import streamlit as st
import cv2
import time
from detect import detect_objects
from auth import add_user, login_user, update_chat_id, get_chat_id

st.set_page_config(page_title="AI Vision Engine", layout="wide")

st.title("AI Vision Engine")

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "detect" not in st.session_state:
    st.session_state.detect = False
if "cap" not in st.session_state:
    st.session_state.cap = None

# ---------------- LOGIN ----------------
if st.session_state.user is None:
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = email
                st.rerun()
            else:
                st.error("Invalid email or password")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            if add_user(email, password):
                st.success("Account created successfully! Please login.")
            else:
                st.error("User already exists")

else:
    # ---------------- DASHBOARD ----------------
    user = st.session_state.user
    st.success(f"Logged in as: **{user}**")

    if st.button("Logout"):
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        st.session_state.user = None
        st.session_state.detect = False
        st.rerun()

    st.divider()
    st.subheader("Telegram Alerts")
    st.markdown("Get your Chat ID → [https://t.me/userinfobot](https://t.me/userinfobot)")

    chat_id_input = st.text_input("Enter Telegram Chat ID", value=st.session_state.get("temp_chat_id", ""))

    if st.button("Save Chat ID"):
        if chat_id_input.strip():
            update_chat_id(user, chat_id_input.strip())
            st.success("Chat ID saved successfully!")
            st.rerun()
        else:
            st.error("Please enter a valid Chat ID")

    saved_chat = get_chat_id(user)

    if saved_chat:
        st.success("✅ Telegram Connected")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Detection", type="primary", disabled=st.session_state.detect):
                st.session_state.detect = True
                st.rerun()

        with col2:
            if st.button("Stop Detection", type="secondary", disabled=not st.session_state.detect):
                st.session_state.detect = False
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.rerun()

        # ---------------- REAL-TIME DETECTION (Using Fragment) ----------------
        @st.fragment(run_every=1)  # Adjust to 0.1–0.5 for faster updates if needed
        def detection_fragment():
            if not st.session_state.detect:
                return

            frame_window = st.empty()

            # Initialize camera only once
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    st.error("Failed to access camera. Make sure no other app is using it.")
                    st.session_state.detect = False
                    st.rerun()
                    return

            ret, frame = st.session_state.cap.read()

            if not ret:
                st.error("Failed to read frame from camera")
                st.session_state.detect = False
                st.rerun()
                return

            # Process frame with your detection + Telegram alerts
            try:
                processed_frame = detect_objects(frame, saved_chat)
            except Exception as e:
                st.error(f"Detection error: {e}")
                processed_frame = frame

            # Display
            frame_window.image(processed_frame, channels="BGR", use_column_width=True)

            # Small delay to control CPU usage
            time.sleep(0.03)  # \~30 FPS max

        if st.session_state.detect:
            detection_fragment()

    else:
        st.warning("⚠️ Please add your Telegram Chat ID first to enable detection.")

# Optional: Footer
st.caption("AI Vision Engine • Real-time Object Detection with Telegram Alerts")
