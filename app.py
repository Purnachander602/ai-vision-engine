import streamlit as st
import cv2
from auth import add_user, login_user, update_chat_id, get_chat_id
from detect import detect_objects

st.set_page_config(page_title="AI Vision Engine", layout="centered")

st.title("AI Vision Engine")

# ---------------- SESSION STATE ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if "detect" not in st.session_state:
    st.session_state.detect = False


# ---------------- LOGIN / SIGNUP ----------------
if st.session_state.user is None:
    login_tab, signup_tab = st.tabs(["Login", "Signup"])

    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = email
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with signup_tab:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

        if st.button("Create Account"):
            if add_user(email, password):
                st.success("Account created successfully. Please login.")
            else:
                st.error("User already exists")


# ---------------- DASHBOARD ----------------
else:
    user = st.session_state.user

    st.success(f"Logged in as: **{user}**")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    st.divider()

    # ---------------- TELEGRAM CONNECTION ----------------
    st.subheader("Telegram Alerts")
    st.markdown("Get your chat ID here → [https://t.me/userinfobot](https://t.me/userinfobot)")

    chat_id_input = st.text_input("Enter Chat ID", key="chat_id_input")

    if st.button("Save Chat ID"):
        if chat_id_input.strip():
            update_chat_id(user, chat_id_input.strip())
            st.success("Chat ID saved successfully!")
            st.rerun()
        else:
            st.warning("Please enter a valid Chat ID")

    saved_chat = get_chat_id(user)

    # ---------------- CAMERA + DETECTION ----------------
    if saved_chat:
        st.success("✅ Telegram Connected")

        st.subheader("Live Object Detection")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ Start Detection"):
                st.session_state.detect = True
                st.rerun()

        with col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.detect = False
                st.rerun()

        # Display area for video frames
        frame_window = st.empty()

        if st.session_state.detect:
            cap = cv2.VideoCapture(0)

            while st.session_state.detect:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break

                # Run detection
                processed_frame = detect_objects(frame, saved_chat)

                # Display the frame
                frame_window.image(processed_frame, channels="BGR", use_column_width=True)

            cap.release()

    else:
        st.warning("⚠️ Please add your Telegram Chat ID first to enable detection alerts.")
