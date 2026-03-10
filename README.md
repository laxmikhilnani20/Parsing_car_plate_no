# ParkQ - Indian License Plate Detection System

## Overview
Production-ready Indian License Plate Detection system with video processing capabilities for parking management and internship demonstrations.

## Features
- 🚗 **YOLOv8n** vehicle detection
- 🔍 **EasyOCR + TrOCR** dual OCR engines
- 🎯 **Indian plate format validation** (AA00AA0000)
- 📹 **Real-time video processing**
- ⚡ **Multi-method detection** approach
- 🎬 **Streamlit web interface** for video uploads

## Models Used
- **YOLOv8n**: Fast vehicle detection
- **EasyOCR**: Primary text recognition
- **TrOCR**: Secondary validation
- **Indian Format Validation**: Strict pattern matching

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebook: `parkq_optimized_plate_detection.ipynb`
3. Launch Streamlit app: `streamlit run app.py`

## Demo Results
- ✅ Processes Indian plate formats: MH20DV2366, DL8CAF5032
- ✅ International plate detection with format validation
- ✅ Real-time video analysis at 30 FPS
- ✅ Production-ready for ParkQ integration

## Architecture
```
Video Input → Vehicle Detection → Plate Region Detection → 
OCR Processing → Format Validation → Indian Plate Output
```

## License
MIT License - Ready for commercial deployment

