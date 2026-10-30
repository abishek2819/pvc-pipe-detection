import os
import time
import cv2
import numpy as np
import requests
import PIL.Image
import streamlit as st
from ultralytics import YOLO

# Page setup
st.set_page_config(
    page_title="Pipe Joint & Defect Detection Dashboard",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Pipe Joint & Defect Detection Dashboard")

# Model Loading
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(MODEL_PATH)

# Sidebar controls (Lowered default confidence to 0.10 to detect all objects)
st.sidebar.header("Detection Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.10, 0.05)
iou_threshold = st.sidebar.slider("IoU Overlap Threshold", 0.10, 1.00, 0.60, 0.05)

if model is None:
    st.sidebar.error(f"Error loading '{MODEL_PATH}'. Ensure it is in the same directory.")
    st.error(f"❌ Cannot find '{MODEL_PATH}'. Please ensure 'best.pt' is placed inside: {os.getcwd()}")
else:
    st.sidebar.success(f"Model '{MODEL_PATH}' loaded successfully!")

    # Tab navigation
    tab1, tab2 = st.tabs(["🖼️ Tab 1: Image Upload", "📹 Tab 2: Live Phone Stream"])

    # --------------------------------------------------
    # TAB 1: IMAGE UPLOAD (STATIC TEST)
    # --------------------------------------------------
    with tab1:
        st.subheader("Single / Batch Image Detection")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        
        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Run Detection"):
                results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="Detection Result", use_container_width=True)
                st.success(f"Detected {len(results[0].boxes)} objects.")

    # --------------------------------------------------
    # TAB 2: LIVE PHONE STREAM (FAIL-SAFE LOOP)
    # --------------------------------------------------
    with tab2:
        st.subheader("Continuous Live Stream")
        
        stream_url = st.text_input(
            "IP Webcam Base URL", 
            value="http://192.168.43.1:8080",
            help="Enter your phone's IP address and port as shown in the IP Webcam app."
        )
        
        run_stream = st.checkbox("Start Continuous Camera Stream")

        metric_spot = st.empty()
        frame_spot = st.empty()
        status_spot = st.empty()

        if run_stream:
            # Clean base URL
            base_url = stream_url.strip().rstrip("/")
            if base_url.endswith(("/video", "/videofeed", "/shot.jpg")):
                base_url = base_url.rsplit("/", 1)[0]
                
            shot_url = f"{base_url}/shot.jpg"
            
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})

            status_spot.info(f"Connecting to {shot_url}...")

            while run_stream:
                try:
                    response = session.get(shot_url, timeout=2)
                    
                    if response.status_code == 200:
                        status_spot.empty()
                        
                        img_arr = np.frombuffer(response.content, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            results = model.predict(
                                frame, 
                                conf=conf_threshold, 
                                iou=iou_threshold, 
                                imgsz=320,
                                verbose=False
                            )
                            
                            annotated_frame = results[0].plot()
                            joint_count = len(results[0].boxes)
                            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                            metric_spot.metric(label="Live Detected Objects", value=joint_count)
                            frame_spot.image(annotated_rgb, use_container_width=True)
                    else:
                        status_spot.error(f"HTTP Status {response.status_code}: Unable to reach phone.")

                except Exception as e:
                    status_spot.warning(f"Reconnecting... Error: {e}")
                
                time.sleep(0.01)