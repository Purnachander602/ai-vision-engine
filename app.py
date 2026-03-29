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
if "frame_window" not in st.session_state:
    st.session_state.frame_window = None

# ====================== LOGIN ======================
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
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            if add_user(email, password):
                st.success("Account created successfully! Please login.")
            else:
                st.error("User already exists")
    st.stop()  # Important: stop execution here

# ====================== DASHBOARD ======================
user = st.session_state.user

st.success(f"Logged in as: **{user}**")

if st.button("Logout"):
    st.session_state.user = None
    st.rerun()

st.divider()

st.subheader("Telegram Alerts")
st.markdown("Get your Chat ID → [https://t.me/userinfobot](https://t.me/userinfobot)")

chat_id = st.text_input("Enter Telegram Chat ID", value=get_chat_id(user) or "")

if st.button("Save Chat ID"):
    if chat_id.strip():
        update_chat_id(user, chat_id.strip())
        st.success("Telegram Chat ID saved!")
        st.rerun()
    else:
        st.error("Please enter a valid Chat ID")

saved_chat = get_chat_id(user)

if not saved_chat:
    st.warning("Please add your Telegram Chat ID first to enable alerts.")
    st.stop()

# ====================== DETECTION SECTION ======================
st.subheader("Object Detection")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Start Detection", type="primary", use_container_width=True):
        st.session_state.detect = True
        st.rerun()

with col2:
    if st.button("Stop Detection", type="secondary", use_container_width=True):
        st.session_state.detect = False
        st.rerun()

# Create placeholder once (outside any loop)
if st.session_state.frame_window is None:
    st.session_state.frame_window = st.empty()

frame_window = st.session_state.frame_window

# Main detection logic using checkbox pattern (recommended)
run_detection = st.checkbox("Run Live Detection", value=st.session_state.detect, key="run_check")

if run_detection and saved_chat:
    st.session_state.detect = True
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access camera. Make sure it is not being used by another application.")
        st.stop()

    try:
        while True:   # This runs only during one script execution
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from camera")
                break

            # Process frame (your detection + telegram alert)
            processed_frame = detect_objects(frame, saved_chat)

            # Display (convert BGR → RGB if needed)
            frame_window.image(processed_frame, channels="BGR", use_column_width=True)

            # Small delay to control FPS and reduce CPU usage
            time.sleep(0.03)  # \~30 FPS max

            # Check if user stopped via checkbox
            if not st.session_state.get("run_check", False):
                break

    except Exception as e:
        st.error(f"Error during detection: {e}")
    finally:
        cap.release()
        st.session_state.detect = False

elif not run_detection:
    st.session_state.detect = False
    frame_window.image([])  # Clear the frame

else:
    st.info("Click **Start Detection** or enable the checkbox above.")

st.caption("Note: Webcam access works only when running **locally**. For deployment use `streamlit-webrtc`.")
