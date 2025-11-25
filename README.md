# Smoking Detection System

This project provides real-time smoking detection using a YOLO model (`smoking-detector.pt`) trained to identify smoking behavior in images and videos. The system runs on the web using Streamlit and offers an easy-to-use interface for uploading media, adjusting settings, and viewing detection statistics.

## Live App

(If hosted publicly, add the link here)

## Features

### YOLO-Based Smoking Detection

* Uses a fine-tuned YOLO model trained on smoking-related imagery.
* Detects smoking in images, videos, and frame-by-frame video streams.
* Displays bounding boxes and confidence scores.

### Streamlit Web Application

* Clean and simple UI.
* Supports:

  * Image upload
  * Video upload
  * Real-time confidence threshold adjustment
* Drag-and-drop file uploader (max 200MB).
* Shows detection results instantly.

### Statistics Display

The app shows key metrics:

* Highest confidence detected
* Total frames processed
* Smoking detected (True/False)

### Demo Files Included

The repository includes sample media for testing:

* `demo.jpg`
* `demo.mp4`
* `The_Usual_Suspects.png`
* `globe.gif`

A ZIP download option is available in the app for convenience.

## Project Structure

```
The_Usual_Suspects.png    # Sample image
demo.jpg                  # Sample image
demo.mp4                  # Sample video
globe.gif                 # Extra sample/gif
main.py                   # Main Streamlit application
packages.txt              # System packages (for cloud runtime)
requirements.txt          # Python dependencies
smoking-detector.pt       # YOLO model for smoking detection
```

## Tech Stack

* Python
* YOLO for smoking detection
* Streamlit for the UI
* OpenCV for image and video frame processing

## How It Works

1. User uploads an image or video.
2. The system processes the input using the YOLO model.
3. Detected smoking instances are marked using bounding boxes.
4. Confidence threshold can be adjusted for sensitivity.
5. The interface displays:

   * Detections
   * Confidence levels
   * Total frames analyzed
   * True/False smoking presence indicator

## Installation (Local)

```
pip install -r requirements.txt
streamlit run main.py
```

Ensure the model file `smoking-detector.pt` is placed in the project directory.

## Use Cases

* Surveillance monitoring
* Workplace safety systems
* Public space analysis
* Behaviour detection research
* Policy enforcement support

## Future Improvements

* Multi-class detection (e.g., cigarette, vape, lighter).
* Real-time webcam support.
* Smoke duct or vapor trail detection.
* Cloud or API deployment.
* Notification and alert system.

## Credits

Made by Techtics.ai
