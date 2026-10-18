# VegetableVision AI — Project Overview

## What the project does

VegetableVision is a local web app that detects supported fruits and vegetables in a photo, outlines each detected object, and estimates its visible 2D length, width, and area. It can report measurements in pixels, or in centimetres when the photo contains a usable size reference.

## Why a phone photo gives pixels, not centimetres

A normal photo records image pixels, not the real-world scale of the scene. The same vegetable appears larger in the image when the camera is closer and smaller when it is farther away. Without a known reference or a calibrated camera setup, the app cannot tell how many centimetres correspond to one pixel.

For example, an object spanning 200 pixels could be 10 cm wide nearby or 20 cm wide farther away. The pixels alone do not tell us which is true. This is a limitation of measuring from a single 2D image, not a detection failure.

## Ways to get centimetre measurements

### 1. ArUco marker — recommended for this demo

The marker acts as a small ruler in the same photo. The app detects its black square, compares its width in pixels with its known physical width, and calculates a pixels-per-centimetre scale.

1. Print the provided `backend/aruco_marker_id0.png` clearly, without cropping or stretching it.
2. Measure the printed **black square only** with a ruler.
3. Place the marker next to the vegetable on the same flat surface. Keep the whole marker visible and in focus.
4. In the app, select **ArUco marker** and enter the measured black-square size in centimetres.
5. Upload or capture the photo and select **Detect & measure**. If the marker is detected, the result includes cm and cm² values.

The app uses this calculation:

```text
pixels_per_cm = marker_width_in_pixels / marker_width_in_cm
length_cm     = vegetable_length_in_pixels / pixels_per_cm
width_cm      = vegetable_width_in_pixels / pixels_per_cm
area_cm2      = vegetable_area_in_pixels / (pixels_per_cm * pixels_per_cm)
```

If the marker is missing, blurry, too small, obscured, or its size is entered incorrectly, the scale will be unavailable and results stay in pixels.

### 2. A4 sheet reference

Place a full A4 sheet beside the vegetable and select **A4 paper (experimental)**. The app assumes standard paper dimensions of 29.7 cm × 21.0 cm and attempts to detect the paper outline. This avoids printing a marker, but detection can be less reliable with shadows, plain backgrounds, cropping, or perspective.

### 3. Fixed and calibrated camera setup

For repeated measurements in a controlled setup, calibrate the camera and scene at a known distance and keep the camera position, zoom, and object plane consistent. A scale can then be established for images from that setup. Merely knowing the camera-to-object distance is not enough on its own: camera calibration/field of view and the object's position relative to the camera are also important. Moving the phone or changing zoom invalidates a previously established scale.

The current demo does not implement automatic phone-distance or camera calibration. For arbitrary handheld mobile photos, an in-frame reference such as the ArUco marker is the most practical of these options.


## How the application works

1. The React web interface accepts an uploaded image or captures a webcam frame.
2. It sends the image, confidence threshold, and selected calibration settings to the local FastAPI backend.
3. The backend uses OpenCV to decode the image and, if requested, find the ArUco or A4 reference and calculate the pixel scale.
4. Ultralytics YOLO runs the packaged segmentation model on the image and returns object masks, class labels, and confidence scores.
5. For each usable mask, OpenCV fits a minimum-area rotated rectangle. The longer side is reported as length, the shorter side as width, and the mask contour area as area.
6. The backend converts these pixel values to centimetres only when calibration produced a valid pixels-per-centimetre value. It returns the measurements and an annotated image to the interface.
7. The UI displays the outlined detections, calibration status, and a table of results.

## Model and training notebook

`models/best.pt` is the trained Ultralytics YOLO instance-segmentation model used at runtime. Unlike a model that only draws rectangular boxes, instance segmentation predicts a shape for each detected object. The app uses that shape to estimate visible 2D dimensions. The configured classes are listed in `backend/config.json` and cover 28 supported produce categories; the model is not intended to detect every kind of object.

### Supported fruit and vegetable classes

The model is configured to detect these 28 classes:

Soybean, apple, banana, broccoli, cabbage, capsicum, carrot, cauliflower, chilli, corn, cucumber, eggplant, garlic, ginger, grapes, lemon, mango, onion, orange, pear, peas, pineapple, potato, radish, strawberry, sweetpotato, tomato, and watermelon.

The root Jupyter notebook documents the optional training and export workflow. In short, it prepares polygon-annotated data, trains a YOLO segmentation model, evaluates it, and exports weights and configuration. The notebook tries YOLO11n-seg and YOLOv8n-seg as base models; the included config does not identify which base produced the packaged weights. Training is not needed to run the demo.

## Main technologies

- **Frontend:** React, Vite, and `lucide-react` icons.
- **Backend:** Python, FastAPI, and Uvicorn.
- **AI inference:** Ultralytics YOLO with PyTorch.
- **Image processing and measurement:** OpenCV contrib and NumPy.

## Scope and limitations

- Without a detected reference, measurements are pixels and square pixels; the app cannot infer real-world scale from an unconstrained RGB image alone.
- Centimetres require a valid scale reference or a calibrated, controlled camera setup.
- The reported length and width describe the visible 2D segmentation outline, not the vegetable's full 3D dimensions or hidden parts.
- Detection quality and measurement accuracy depend on image quality, segmentation quality, camera perspective, and calibration.

## Running the project

The complete installation and startup commands for a fresh PC are in [`README.md`](./README.md). The demo runs as two local services: the Python backend and the Vite frontend.
