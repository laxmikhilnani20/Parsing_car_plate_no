# ParkQ – License Plate Detection Agent

## Identity

You are a **computer vision engineering agent** building the license plate detection module for the ParkQ parking management system.

## Mission

Implement a working prototype that automatically detects and reads vehicle license plates from parking entry camera footage.

---

## Instructions

### Step 1 – Accept Video Input

- Accept video from: recorded file, mobile recording, traffic dataset, or webcam feed.
- Supported formats: MP4, AVI, MOV, or live camera stream.

### Step 2 – Extract Frames

- Do **not** process every frame. Sample **1 frame every 10–15 frames**.
- Example: a 10s video at 30 FPS = 300 frames → process ~30 frames.

### Step 3 – Detect License Plate

- For each sampled frame, run a license plate detection model.
- Output a **bounding box** (top-left, bottom-right coordinates) around the plate.
- If no plate is found in a frame, skip it.

### Step 4 – Crop Plate Region

- Using the bounding box, crop the frame to extract **only the license plate area**.
- This cropped image is the input for OCR.

### Step 5 – Run OCR

- Apply Optical Character Recognition on the cropped plate image.
- Extract the **alphanumeric text** and a **confidence score**.
- Only accept results above the confidence threshold.

### Step 6 – Validate Plate Number

- Filter out false detections using these rules:
  - Minimum **4 characters**
  - Must be **alphanumeric**
  - Should match Indian plate patterns: `AA00AA0000` or `AA00A0000`
- Examples of valid plates: `MH31AB1234`, `DL8CAF5032`

### Step 7 – Output Result

- For each valid detection, output:

```
Vehicle detected
Plate Number: <PLATE_NUMBER>
Time: <TIMESTAMP>
```

- Deduplicate across frames (same plate appearing in multiple frames = 1 entry).

---

## Pipeline Overview

```
Video Source
    ↓
Frame Extraction (1 per 10–15 frames)
    ↓
License Plate Detection (bounding box)
    ↓
Plate Region Cropping
    ↓
OCR (text + confidence)
    ↓
Validation (pattern matching)
    ↓
Deduplicated Vehicle Number Output
```

---

## Constraints

- **In scope:** plate detection, OCR, validation, result output.
- **Out of scope:** car model detection, vehicle database, parking slot management, payment integration.
- Focus on building a **working prototype** — correctness over optimization.

---

## Testing Checklist

1. **Single image** – detect plate and extract text from one car photo.
2. **Multiple images** – different cars, lighting, angles.
3. **Video** – short clip of a car entering a parking gate; verify consistent detection.

---

## Optional: Demo Interface

If building a demo UI, include:

- Video upload control
- "Run Detection" button
- List of detected plate numbers

Example output:

```
Detected Vehicles
─────────────────
MH31AB1234
MH49KX2451
DL8CAF5032
```

---

## Future Context

Once integrated into the full ParkQ system, detected plate numbers will feed into:

- Vehicle entry logging
- Registered user matching
- Parking slot assignment
- Ticket generation
- Duration tracking

The agent does **not** need to implement these — only produce accurate plate numbers.
