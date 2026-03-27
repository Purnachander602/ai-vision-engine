import streamlit as st
import cv2
import time
from auth import add_user, login_user, update_chat_id, get_chat_id
from detect import detect_objects

st.set_page_config(page_title="AI Vision Engine", layout="centered")

st.title("AI Vision Engine")

# ---------------- SESSION STATE ----------------

if "user" not in st.session_state:
    st.session_state["user"] = None

if "camera_running" not in st.session_state:
    st.session_state["camera_running"] = False


# ---------------- LOGIN PAGE ----------------

if st.session_state["user"] is None:

    login_tab, signup_tab = st.tabs(["Login", "Signup"])

    # LOGIN
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

    # SIGNUP
    with signup_tab:

        new_email = st.text_input("Email", key="signup_email")
        new_pass = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create Account"):

            created = add_user(new_email, new_pass)

            if created:
                st.success("Account created. Please login.")
            else:
                st.error("User already exists")


# ---------------- DASHBOARD AFTER LOGIN ----------------

else:

    user = st.session_state["user"]

    st.success(f"Logged in as {user}")

    if st.button("Logout"):
        st.session_state["user"] = None
        st.session_state["camera_running"] = False
        st.rerun()

    st.divider()

    st.subheader("Connect Telegram Notifications")

    st.markdown("Get your chat id here → https://t.me/userinfobot")

    chat_id = st.text_input("Enter Telegram Chat ID")

    if st.button("Save Chat ID"):

        update_chat_id(user, chat_id)

        st.success("Chat ID saved")

    saved_chat = get_chat_id(user)

    if saved_chat:

        st.success("Telegram Connected")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Detection"):
                st.session_state["camera_running"] = True

        with col2:
            if st.button("Stop Detection"):
                st.session_state["camera_running"] = False

        frame_window = st.image([])

        # -------- CAMERA SECTION --------

        if st.session_state["camera_running"]:

            cap = cv2.VideoCapture(0)

            # reduce camera resolution (important for speed)
            cap.set(3, 640)
            cap.set(4, 480)

            while st.session_state["camera_running"]:

                ret, frame = cap.read()

                if not ret:
                    st.error("Camera not working")
                    break

                # resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))

                # object detection
                frame = detect_objects(frame, saved_chat)

                # convert color for streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_window.image(frame)

                # small delay to reduce lag
                time.sleep(0.03)

            cap.release()