import os
import tempfile

import cv2
import cvzone
import streamlit as st
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Smoking Detection", page_icon="🚭")

# Custom Smoking Model
model = YOLO("smoking-detector.pt")

# Header row with logo and title (UI preserved exactly)
col_logo, col_title, col_credit = st.columns([1, 4, 1])
with col_logo:
    st.image("globe.gif")

with col_title:
    st.markdown("""
        <div style='text-align: left; padding: 15px 0px; margin: 0px;'>
            <h2 style='color: #4682B4; margin: 0; padding: 0;'>SMOKING DETECTION SYSTEM</h2>
            <p style='color: #666; margin: 0; padding: 0;'>AI-Powered Smoking Detection using YOLO</p>
        </div>
    """, unsafe_allow_html=True)

with col_credit:
    st.markdown(
        """
        <div style='text-align: right; color: #666; padding: 15px 0px; margin: 0px;'>
            <p style='margin: 0; padding: 0;'>Made by <a href='https://techtics.ai' target='_blank' 
            style='color: #4682B4; text-decoration: none; font-weight: bold;'>Techtics.ai</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Custom Circular Progress component
class CircularProgress:
    def __init__(self, label, value, key, color="skyblue", track_color="#f0f0f0", size="Large"):
        self.label = label
        self.value = value
        self.key = key
        self.color = color
        self.track_color = track_color
        self.size = size

    def st_circular_progress(self):
        size_map = {"Small": 100, "Medium": 150, "Large": 200}
        size_px = size_map.get(self.size, 150)
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <div style="position: relative; display: inline-block;">
                <svg width="{size_px}" height="{size_px}" viewBox="0 0 120 120">
                    <!-- Track -->
                    <circle cx="60" cy="60" r="54" stroke="{self.track_color}" stroke-width="8" fill="none"/>
                    <!-- Progress -->
                    <circle cx="60" cy="60" r="54" stroke="{self.color}" stroke-width="8" fill="none"
                            stroke-dasharray="339.292" stroke-dashoffset="{339.292 * (1 - self.value / 100)}"
                            transform="rotate(-90 60 60)" stroke-linecap="round"/>
                    <!-- Text -->
                    <text x="60" y="60" text-anchor="middle" dy="7" font-size="24" font-weight="bold" 
                    fill="{self.color}">
                        {int(self.value)}%
                    </text>
                </svg>
            </div>
            <div style="margin-top: 10px; font-weight: bold; color: #666;">{self.label}</div>
        </div>
        """, unsafe_allow_html=True)

    def update_value(self, progress):
        self.value = progress


def process_frame(frame, confidence_threshold, smoking_detected, blink_state, is_video=False):
    h, w = frame.shape[:2]
    scale_factor = (w + h) / 1500

    corner_length = int((40 if is_video else 60) * scale_factor)
    corner_thickness = int(3 * scale_factor)
    font_scale = 0.9 * scale_factor
    font_thickness = max(1, int(3 * scale_factor))
    neon_red_color = (49, 49, 255)

    results = model(frame, stream=True, conf=confidence_threshold, verbose=False)
    smoking_in_frame = False
    max_confidence = 0.0

    for r in results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            if confidence > max_confidence:
                max_confidence = confidence

            smoking_in_frame = True
            x1, y1, x2, y2 = box.xyxy[0]
            x, y, bw, bh = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cvzone.cornerRect(frame, (x, y, bw, bh),
                              l=corner_length, t=corner_thickness,
                              rt=0, colorC=neon_red_color)

    if smoking_in_frame and not smoking_detected:
        smoking_detected = True

    if smoking_detected and blink_state:
        box_width = int(300 * scale_factor)
        box_height = int(50 * scale_factor)
        margin = int(20 * scale_factor)

        x1 = margin
        y1 = frame.shape[0] - box_height - margin
        x2 = x1 + box_width
        y2 = y1 + box_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 50), -1)
        roi = frame[y1:y2, x1:x2]
        overlay_roi = overlay[y1:y2, x1:x2]
        cv2.addWeighted(overlay_roi, 0.5, roi, 0.5, 0, roi)

        text = "SMOKING DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, font_thickness)[0]
        text_x = x1 + (box_width - text_size[0]) // 2
        text_y = y1 + (box_height + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, neon_red_color, font_thickness)

    # Return max_confidence and current frame confidence (both as percentages)
    return frame, smoking_detected, max_confidence * 100, max_confidence * 100


def resize_for_display(frame, target_width=720, target_height=450):
    """
    Resize the frame to fit within (target_width x target_height)
    while preserving aspect ratio. Scales up if smaller, down if larger.
    """
    h, w = frame.shape[:2]
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the frame with preserved aspect ratio
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return frame_resized


# Helper to render statistics block (keeps layout/content identical)
def render_stats_section(processing, total_frames, smoking_detected, current_confidence, max_confidence):
    # This function preserves the exact appearance/markup as before.
    progress_value = current_confidence if processing else max_confidence
    label = "Current Confidence" if processing else "Highest Confidence"
    progress_color = "orangered" if progress_value > 70 else "skyblue"

    progress_bar = CircularProgress(
        label=label,
        value=progress_value,
        key="confidence_bar" if processing else "final_confidence_bar",
        color=progress_color,
        track_color="#f0f0f0",
        size="Medium"
    )
    progress_bar.st_circular_progress()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style='background-color: #e8f5e8; padding: 10px; border-radius: 5px; text-align: center;'>
            <div style='font-size: 12px; color: #666;'>Total Frames</div>
            <div style='font-size: 24px; font-weight: bold; color: #2e7d32;'>{total_frames}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        status_color = "#ffebee" if not smoking_detected else "#e8f5e8"
        text_color = "#c62828" if not smoking_detected else "#2e7d32"
        status_text = "False" if not smoking_detected else "True"
        st.markdown(f"""
        <div style='background-color: {status_color}; padding: 10px; border-radius: 5px; text-align: center;'>
            <div style='font-size: 12px; color: #666;'>Smoking Detected</div>
            <div style='font-size: 24px; font-weight: bold; color: {text_color};'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)


# UI layout preserved (3 columns)
col1, col2, col3 = st.columns([1, 2, 1])

default_state = {
    'processing': False,
    'max_confidence': 0.0,
    'current_confidence': 0.0,
    'smoking_detected': False,
    'total_frames': 0,
    'file_processed': False
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

with col1:
    st.header("User Config")
    upload_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov"], width=300)
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                                     width=300)

with col2:
    st.header("Display")
    placeholder = st.empty()
    video_progress_bar = st.empty()

with col3:
    st.header("Statistics")
    stats_placeholder = st.empty()

# Auto-upload demo file at start
if not st.session_state.file_processed and not upload_file:
    demo_path = "demo.jpg"
    if os.path.exists(demo_path):
        class DemoFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self.path = path

            def read(self):
                with open(self.path, 'rb') as f:
                    return f.read()


        upload_file = DemoFile(demo_path)
        st.session_state.file_processed = True

# Main processing
if upload_file is not None:
    st.session_state.processing = True
    st.session_state.max_confidence = 0.0
    st.session_state.current_confidence = 0.0
    st.session_state.smoking_detected = False
    st.session_state.total_frames = 0

    # Save temp file to process
    tfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        tfile.write(upload_file.read())
        tfile.close()
        file_path = tfile.name
        filename = upload_file.name.lower()

        if filename.endswith((".jpg", ".jpeg", ".png")):
            # IMAGE mode (preserved)
            frame = cv2.imread(file_path)
            if frame is not None:
                frame_out, smoking_detected, max_confidence, current_confidence = process_frame(
                    frame.copy(), confidence_threshold, False, True, is_video=False
                )
                display_frame = resize_for_display(frame_out)
                placeholder.image(display_frame, channels="BGR")

                st.session_state.max_confidence = max_confidence
                st.session_state.current_confidence = current_confidence
                st.session_state.smoking_detected = smoking_detected
                st.session_state.total_frames = 1

                video_progress_bar.empty()

        elif filename.endswith((".mp4", ".mov")):
            # VIDEO mode (preserved behavior)
            cap = cv2.VideoCapture(file_path)
            try:
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    current_frame = 0
                    blink_state = True
                    blink_counter = 0
                    blink_interval = 5

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_out, smoking_detected, max_confidence, current_confidence = process_frame(
                            frame.copy(),
                            confidence_threshold,
                            st.session_state.smoking_detected,
                            blink_state,
                            is_video=True
                        )

                        display_frame = resize_for_display(frame_out)
                        placeholder.image(display_frame, channels="BGR")

                        # update session state
                        st.session_state.max_confidence = max(st.session_state.max_confidence, max_confidence)
                        st.session_state.current_confidence = current_confidence
                        st.session_state.smoking_detected = st.session_state.smoking_detected or smoking_detected

                        current_frame += 1
                        st.session_state.total_frames = current_frame

                        if total_frames > 0:
                            progress = current_frame / total_frames
                            video_progress_bar.progress(progress)

                        # Update stats in real-time (keeps appearance)
                        with col3:
                            stats_placeholder.empty()
                            with stats_placeholder.container():
                                render_stats_section(
                                    processing=True,
                                    total_frames=st.session_state.total_frames,
                                    smoking_detected=st.session_state.smoking_detected,
                                    current_confidence=st.session_state.current_confidence,
                                    max_confidence=st.session_state.max_confidence
                                )

                        if st.session_state.smoking_detected:
                            blink_counter += 1
                            if blink_counter >= blink_interval:
                                blink_state = not blink_state
                                blink_counter = 0

                    # ensure progress bar completes at end
                    video_progress_bar.progress(1.0)
            finally:
                cap.release()

    finally:
        # ensure temp file is removed if it exists
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
        except PermissionError:
            pass

    st.session_state.processing = False

    # Fun end-of-processing effects
    if st.session_state.smoking_detected:
        st.toast("🚭 Smoking Detected!")
    else:
        st.snow()
        st.toast("✅ No Smoking Detected!")

# Final stats display when not processing
if not st.session_state.processing and st.session_state.total_frames > 0:
    with col3:
        stats_placeholder.empty()
        with stats_placeholder.container():
            render_stats_section(
                processing=False,
                total_frames=st.session_state.total_frames,
                smoking_detected=st.session_state.smoking_detected,
                current_confidence=st.session_state.current_confidence,
                max_confidence=st.session_state.max_confidence
            )
