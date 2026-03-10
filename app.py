import streamlit as st
import cv2
import os
import re
import time
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import torch
from ultralytics import YOLO
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Page configuration
st.set_page_config(
    page_title="ParkQ - Indian License Plate Detection", 
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #1f4e79, #2e8b57);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f4e79;
}
.success-plate {
    background: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}
.detected-plate {
    background: #fff3cd;
    color: #856404;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}
.error-msg {
    background: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.yolo_model = None
    st.session_state.easy_reader = None
    st.session_state.trocr_processor = None
    st.session_state.trocr_model = None
    st.session_state.device = None

@st.cache_resource
def load_models():
    """Load all ML models with caching for performance."""
    try:
        with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):
            # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load YOLOv8
            yolo_model = YOLO("yolov8n.pt")
            
            # Load EasyOCR
            easy_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            
            # Load TrOCR
            trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
            trocr_model.eval()
            
            return yolo_model, easy_reader, trocr_processor, trocr_model, device
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

def detect_vehicle(image, yolo_model, conf_threshold=0.3):
    """Detect vehicles in image using YOLO."""
    results = yolo_model(image, conf=conf_threshold, verbose=False)
    
    # Vehicle classes
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    vehicles = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    vehicles.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0]),
                        "area": area
                    })
    
    vehicles.sort(key=lambda x: x["area"], reverse=True)
    return vehicles

def find_plate_regions(car_img):
    """Find potential license plate regions."""
    candidates = []
    h, w = car_img.shape[:2]
    
    # White/yellow plate detection
    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)
    
    # White plates
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))
    
    # Yellow plates
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))
    
    for mask, color in [(white_mask, "white"), (yellow_mask, "yellow")]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < w * h * 0.002 or area > w * h * 0.2:
                continue
                
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / max(ch, 1)
            
            if 2.0 <= aspect_ratio <= 6.0 and ch > 15 and cw > 40:
                center_y = (y + ch/2) / h
                position_score = 3 if center_y > 0.5 else 1
                
                candidates.append({
                    "bbox": (x, y, x + cw, y + ch),
                    "aspect_ratio": aspect_ratio,
                    "area": area,
                    "color": color,
                    "position_score": position_score
                })
    
    # Remove overlaps and sort
    candidates = remove_overlaps(candidates)
    candidates.sort(key=lambda x: (x["position_score"], x["area"]), reverse=True)
    
    return candidates[:5]  # Top 5 candidates

def remove_overlaps(candidates, iou_threshold=0.3):
    """Remove overlapping bounding boxes."""
    if not candidates:
        return []
    
    kept = []
    for cand in candidates:
        overlap = False
        for k in kept:
            iou = compute_iou(cand["bbox"], k["bbox"])
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
    
    return kept

def compute_iou(box1, box2):
    """Compute IoU of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / max(union, 1)

def preprocess_plate(plate_img):
    """Generate multiple preprocessed versions for OCR."""
    variants = {}
    
    # Resize for consistent OCR
    h, w = plate_img.shape[:2]
    target_h = 80
    scale = target_h / max(h, 1)
    resized = cv2.resize(plate_img, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    
    variants["original"] = resized
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    
    # OTSU threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants["otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 5)
    variants["adaptive"] = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    
    return variants

def run_easyocr(img, reader):
    """Extract text using EasyOCR."""
    try:
        results = reader.readtext(img)
        texts = []
        for (_, text, conf) in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper().strip())
            if len(cleaned) >= 3:
                texts.append({"text": cleaned, "raw": text, "confidence": conf})
        return texts
    except:
        return []

def run_trocr(img, processor, model, device):
    """Extract text using TrOCR."""
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        pil_img = Image.fromarray(img_rgb)
        
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=16)
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper().strip())
        
        if len(cleaned) >= 3:
            return [{"text": cleaned, "raw": text, "confidence": 0.5}]
        return []
    except:
        return []

def validate_indian_plate(text):
    """Validate if text matches Indian license plate format."""
    if not text or len(text) < 8:
        return False, 0
    
    score = 0
    
    # Indian patterns
    patterns = [
        (r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$", 100),    # Standard: MH20DV2366
        (r"^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$", 85),     # Old: MH20D2366
        (r"^\d{2}BH\d{4}[A-Z]{2}$", 95),           # BH series
        (r"^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$", 80),     # 3-letter
    ]
    
    for pattern, pattern_score in patterns:
        if re.match(pattern, text):
            score = pattern_score
            break
    
    # Indian state codes
    indian_states = {
        "MH", "DL", "KA", "TN", "AP", "TS", "UP", "RJ", "GJ", "WB", "MP", "KL",
        "HR", "PB", "CH", "JK", "HP", "UK", "GA", "BR", "OR", "JH", "CG", "AS", "SK"
    }
    
    if len(text) >= 2 and text[:2] in indian_states:
        score += 15
    
    # Must have both letters and numbers
    has_alpha = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    if has_alpha and has_digit:
        score += 10
    
    # Length bonus
    if len(text) == 10:
        score += 20
    elif len(text) == 9:
        score += 15
    elif len(text) in [8, 11]:
        score += 5
    
    return score >= 50, score

def process_video_frame(frame, yolo_model, easy_reader, trocr_processor, trocr_model, device):
    """Process a single video frame for license plate detection."""
    try:
        # Detect vehicles
        vehicles = detect_vehicle(frame, yolo_model)
        
        if vehicles:
            # Use largest vehicle
            vehicle = vehicles[0]
            x1, y1, x2, y2 = vehicle["bbox"]
            
            # Add padding
            h, w = frame.shape[:2]
            pad = 20
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            car_img = frame[y1:y2, x1:x2]
        else:
            car_img = frame.copy()
        
        # Find plate regions
        plate_regions = find_plate_regions(car_img)
        
        all_candidates = []
        
        # Process each plate region
        for region in plate_regions:
            px1, py1, px2, py2 = region["bbox"]
            plate_crop = car_img[py1:py2, px1:px2]
            
            if plate_crop.shape[0] < 10 or plate_crop.shape[1] < 20:
                continue
            
            # Preprocess and run OCR
            variants = preprocess_plate(plate_crop)
            
            for variant_name, variant_img in variants.items():
                # EasyOCR
                easy_results = run_easyocr(variant_img, easy_reader)
                for result in easy_results:
                    is_valid, score = validate_indian_plate(result["text"])
                    final_score = score + (region["position_score"] * 10) + (result["confidence"] * 20)
                    
                    all_candidates.append({
                        "text": result["text"],
                        "confidence": result["confidence"],
                        "score": final_score,
                        "method": f"easyocr_{variant_name}",
                        "is_valid": is_valid,
                        "bbox": region["bbox"]
                    })
                
                # TrOCR for selected variants
                if variant_name in ["original", "otsu"]:
                    trocr_results = run_trocr(variant_img, trocr_processor, trocr_model, device)
                    for result in trocr_results:
                        is_valid, score = validate_indian_plate(result["text"])
                        final_score = score + (region["position_score"] * 10) + 10
                        
                        all_candidates.append({
                            "text": result["text"],
                            "confidence": result["confidence"],
                            "score": final_score,
                            "method": f"trocr_{variant_name}",
                            "is_valid": is_valid,
                            "bbox": region["bbox"]
                        })
        
        # Select best result
        if all_candidates:
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = all_candidates[0]
            
            return {
                "plate_number": best["text"],
                "confidence": best["confidence"],
                "score": best["score"],
                "method": best["method"],
                "is_valid": best["is_valid"],
                "bbox": best.get("bbox"),
                "vehicle_bbox": vehicle["bbox"] if vehicles else None
            }
    
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
    
    return {
        "plate_number": None,
        "confidence": 0.0,
        "score": 0.0,
        "method": None,
        "is_valid": False,
        "bbox": None,
        "vehicle_bbox": None
    }

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 ParkQ - Indian License Plate Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Load models
        if not st.session_state.models_loaded:
            if st.button("🔄 Load AI Models", type="primary"):
                models = load_models()
                if all(model is not None for model in models):
                    (st.session_state.yolo_model, st.session_state.easy_reader, 
                     st.session_state.trocr_processor, st.session_state.trocr_model, 
                     st.session_state.device) = models
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Models Ready")
        
        if st.session_state.models_loaded:
            st.subheader("⚙️ Processing Settings")
            frame_skip = st.slider("Frame Skip (for speed)", 1, 30, 10, 
                                  help="Process every Nth frame for faster processing")
            confidence_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.1,
                                            help="Minimum confidence for detections")
            max_frames = st.slider("Max Frames to Process", 10, 500, 100,
                                 help="Limit processing for large videos")
        
        st.subheader("ℹ️ About")
        st.info("""
        **ParkQ License Plate Detection**
        
        🎯 **Specialized for Indian plates**
        - Format: AA00AA0000
        - Multi-OCR validation
        - Real-time processing
        
        🚀 **Perfect for:**
        - Parking management
        - Security systems
        - Traffic monitoring
        """)
    
    # Main content
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load the AI models from the sidebar before proceeding.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🚗 Vehicle Detection</h3>
                <p>YOLOv8n for fast and accurate vehicle detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 Dual OCR Engine</h3>
                <p>EasyOCR + TrOCR for maximum text recognition</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Indian Validation</h3>
                <p>Strict format validation for Indian license plates</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Video upload and processing
    st.header("📹 Upload Video for Processing")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Upload a video containing vehicles with license plates"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video info
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📐 Resolution", f"{width}x{height}")
        with col2:
            st.metric("⏱️ Duration", f"{duration:.1f}s")
        with col3:
            st.metric("🎞️ FPS", fps)
        with col4:
            st.metric("🖼️ Total Frames", total_frames)
        
        # Process video button
        if st.button("🚀 Process Video", type="primary"):
            process_video(video_path, frame_skip, confidence_threshold, max_frames)
        
        # Clean up
        try:
            os.unlink(video_path)
        except:
            pass

def process_video(video_path, frame_skip, confidence_threshold, max_frames):
    """Process the uploaded video and display results."""
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Processing variables
    frame_results = []
    processed_frames = 0
    frame_count = 0
    valid_detections = []
    all_detections = []
    
    start_time = time.time()
    
    # Results display containers
    with results_container:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎬 Processing Results")
            results_placeholder = st.empty()
        
        with col2:
            st.subheader("📊 Live Stats")
            stats_placeholder = st.empty()
    
    # Process frames
    while processed_frames < max_frames and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for efficiency
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        frame_count += 1
        processed_frames += 1
        
        # Update progress
        progress = min(processed_frames / max_frames, frame_count / total_frames)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} (Processed: {processed_frames})")
        
        # Process frame
        result = process_video_frame(
            frame, 
            st.session_state.yolo_model,
            st.session_state.easy_reader,
            st.session_state.trocr_processor,
            st.session_state.trocr_model,
            st.session_state.device
        )
        
        frame_info = {
            "frame_number": frame_count,
            "time_seconds": frame_count / fps,
            "plate_detected": result['plate_number'] is not None,
            "plate_number": result['plate_number'],
            "confidence": result['confidence'],
            "score": result['score'],
            "is_valid_indian": result['is_valid'],
            "method": result['method']
        }
        
        frame_results.append(frame_info)
        
        # Track detections
        if result['plate_number']:
            all_detections.append(frame_info)
            
            if result['is_valid']:
                valid_detections.append(frame_info)
        
        # Update live results every 5 frames
        if processed_frames % 5 == 0:
            update_live_results(results_placeholder, stats_placeholder, 
                              frame_results, all_detections, valid_detections, processed_frames)
    
    cap.release()
    processing_time = time.time() - start_time
    
    # Final results
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    display_final_results(frame_results, all_detections, valid_detections, 
                         processed_frames, processing_time, fps)

def update_live_results(results_placeholder, stats_placeholder, 
                       frame_results, all_detections, valid_detections, processed_frames):
    """Update live results display."""
    
    with stats_placeholder:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔍 Processed", processed_frames)
        with col2:
            st.metric("📱 Detected", len(all_detections))
        with col3:
            st.metric("✅ Valid Indian", len(valid_detections))
    
    # Show recent detections
    if all_detections:
        recent_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)[:5]
        
        with results_placeholder:
            for i, det in enumerate(recent_detections, 1):
                if det['is_valid_indian']:
                    st.markdown(f"""
                    <div class="success-plate">
                        <strong>✅ {det['plate_number']}</strong> | Frame {det['frame_number']} | 
                        Time: {det['time_seconds']:.1f}s | Score: {det['score']:.1f}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="detected-plate">
                        <strong>⚠️ {det['plate_number']}</strong> | Frame {det['frame_number']} | 
                        Time: {det['time_seconds']:.1f}s | Score: {det['score']:.1f}
                    </div>
                    """, unsafe_allow_html=True)

def display_final_results(frame_results, all_detections, valid_detections, 
                         processed_frames, processing_time, fps):
    """Display comprehensive final results."""
    
    st.header("🏁 Final Results")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🖼️ Frames Processed", processed_frames)
    
    with col2:
        detection_rate = len(all_detections) / processed_frames * 100 if processed_frames > 0 else 0
        st.metric("📱 Detection Rate", f"{detection_rate:.1f}%")
    
    with col3:
        valid_rate = len(valid_detections) / processed_frames * 100 if processed_frames > 0 else 0
        st.metric("✅ Valid Rate", f"{valid_rate:.1f}%")
    
    with col4:
        processing_fps = processed_frames / processing_time if processing_time > 0 else 0
        st.metric("⚡ Processing Speed", f"{processing_fps:.1f} fps")
    
    with col5:
        st.metric("⏱️ Total Time", f"{processing_time:.1f}s")
    
    # Detailed results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🎯 Valid Indian License Plates Detected")
        
        if valid_detections:
            # Create DataFrame for valid detections
            valid_df = pd.DataFrame(valid_detections)
            
            # Display unique plates
            unique_plates = valid_df['plate_number'].unique()
            
            for plate in unique_plates:
                plate_detections = valid_df[valid_df['plate_number'] == plate]
                best_detection = plate_detections.loc[plate_detections['score'].idxmax()]
                
                st.markdown(f"""
                <div class="success-plate">
                    <h4>🚗 {plate}</h4>
                    <p><strong>Best Detection:</strong> Frame {best_detection['frame_number']} | 
                    Time: {best_detection['time_seconds']:.1f}s | 
                    Score: {best_detection['score']:.1f} | 
                    Method: {best_detection['method']}</p>
                    <p><strong>Total Detections:</strong> {len(plate_detections)} frames</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No valid Indian license plates detected in this video.")
    
    with col2:
        st.subheader("📊 Detection Statistics")
        
        if frame_results:
            # Create summary chart
            detection_summary = {
                'Valid Indian Plates': len(valid_detections),
                'Other Text Detected': len(all_detections) - len(valid_detections),
                'No Detection': processed_frames - len(all_detections)
            }
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#28a745', '#ffc107', '#6c757d']
            wedges, texts, autotexts = ax.pie(
                detection_summary.values(), 
                labels=detection_summary.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title('Detection Results Distribution', fontsize=16, fontweight='bold')
            st.pyplot(fig)
            plt.close()
    
    # All detections table
    if all_detections:
        st.subheader("📋 All Detections")
        
        # Create DataFrame
        all_df = pd.DataFrame(all_detections)
        
        # Add status column
        all_df['Status'] = all_df['is_valid_indian'].apply(
            lambda x: '✅ Valid Indian' if x else '⚠️ Detected Text'
        )
        
        # Display table
        st.dataframe(
            all_df[['frame_number', 'time_seconds', 'plate_number', 'score', 'Status', 'method']].round(2),
            column_config={
                'frame_number': 'Frame #',
                'time_seconds': 'Time (s)',
                'plate_number': 'Detected Text',
                'score': 'Score',
                'Status': 'Validation',
                'method': 'Detection Method'
            },
            use_container_width=True
        )
    
    # Download results
    if frame_results:
        results_df = pd.DataFrame(frame_results)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"license_plate_results_{int(time.time())}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()