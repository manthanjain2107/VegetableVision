# VegetableVision AI — Local Demo

VegetableVision is a local web application that detects supported fruit and vegetable instances in a photo, outlines each detected object, and estimates its visible 2D dimensions. The browser interface is built with React and Vite; a Python FastAPI service runs the trained YOLO segmentation model and the OpenCV measurement/calibration code.

This guide covers installing the software on a fresh Windows PC, starting both parts of the app, using the demo, and understanding its model and measurements. You do **not** need a GPU, CUDA, Git, a Roboflow account, or a Kaggle account to run the packaged demo. An internet connection is needed to install Python/npm dependencies; no dataset download or model training is part of the normal startup.

## What is in this project

```text
.
├── README.md
├── notebook8c97219a75 (4).ipynb  # Optional Kaggle model-training/export notebook
├── backend/
│   ├── main.py                   # FastAPI service and YOLO inference
│   ├── measurement.py            # Polygon dimensions and drawing helpers
│   ├── calibration.py            # ArUco/A4 reference detection
│   ├── config.json               # Class names and inference/calibration defaults
│   ├── requirements.txt          # Python runtime dependencies
│   └── aruco_marker_id0.png      # Printable ArUco marker image
├── frontend/
│   ├── index.html
│   ├── package.json              # React/Vite scripts and dependencies
│   └── src/
│       ├── main.jsx              # User interface and API requests
│       └── style.css
└── models/
    └── best.pt                   # Trained YOLO instance-segmentation weights
```

The model file is included in the project. Keep `models\best.pt` in place: the backend loads it at startup using a path relative to `backend\main.py`.

## Fresh Windows PC setup

### 1. Install Python and Node.js

Install these tools before setting up the project:

1. **Python 3.11 (64-bit):** download the Windows installer from [python.org](https://www.python.org/downloads/windows/). During setup, enable the Python launcher (`py`) and, if offered, **Add Python to PATH**. Python 3.11 is recommended for compatibility with the current PyTorch/Ultralytics stack.
2. **Node.js 22 or newer (LTS):** install the Windows LTS release from [nodejs.org](https://nodejs.org/). npm is included with Node.js.
3. If the installer asks, restart PowerShell or open a new terminal so PATH changes take effect.

Check the installations in a new PowerShell window:

```powershell
py -3.11 --version
node --version
npm --version
```

No separate CUDA or GPU software is necessary. Inference can run on the CPU; a compatible GPU may be used by PyTorch if one is available and configured, but this project does not require it.

### 2. Open the project folder

Open PowerShell and change to the folder where you copied or downloaded this project. Replace the example path with the actual path on your PC:

```powershell
Set-Location "C:\path\to\vegetablevision_local_demo"
```

Confirm that the project files are present:

```powershell
Test-Path .\backend\main.py
Test-Path .\frontend\package.json
Test-Path .\models\best.pt
```

Each command should print `True`.

### 3. Install backend dependencies

In the same PowerShell window:

```powershell
Set-Location .\backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual-environment executable directly avoids needing to change PowerShell's script execution policy. The first install downloads FastAPI, Uvicorn, Ultralytics/PyTorch, OpenCV contrib, NumPy, and their dependencies; it may take several minutes.

### 4. Start the backend

Keep this PowerShell window open and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Wait for Uvicorn's startup message. The first startup loads `models\best.pt` and may take longer than later startups.

Check that the service is responding by opening these URLs in a browser:

- Health and model status: <http://127.0.0.1:8000/health>
- Supported class names: <http://127.0.0.1:8000/classes>
- Interactive API documentation: <http://127.0.0.1:8000/docs>

Leave the backend running. To stop it, focus this terminal and press **Ctrl+C**.

### 5. Install and start the frontend

Open a **second** PowerShell window, change to the project root again, and install the JavaScript packages:

```powershell
Set-Location "C:\path\to\vegetablevision_local_demo\frontend"
npm install
npm run dev
```

Keep this window open as well. Open the local address printed by Vite, normally <http://localhost:5173>. The frontend calls the backend at <http://127.0.0.1:8000>; therefore, both services must be running on the same PC. The configured API/CORS setup is for local use, not access by other computers on a network.

To stop the frontend, focus its terminal and press **Ctrl+C**.

### Starting the demo after the first setup

After installing dependencies once, you only need two terminals:

**Terminal 1 — backend**

```powershell
Set-Location "C:\path\to\vegetablevision_local_demo\backend"
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend**

```powershell
Set-Location "C:\path\to\vegetablevision_local_demo\frontend"
npm run dev
```

Then open <http://localhost:5173>.

## Using the application

1. In **Input image**, click the upload area and choose a JPG, PNG, or other browser-supported image. Alternatively, choose **Use webcam**, allow camera access, and select **Capture frame**.
2. Choose a calibration option:
   - **None — pixels only:** no reference object is needed; output dimensions are pixels and area is square pixels.
   - **ArUco marker:** include a supported ArUco marker in the same image. Enter the physical side length of the marker's black square in centimetres. The default is 5 cm.
   - **A4 paper (experimental):** include a visible A4 sheet. The detector assumes the sheet is 29.7 cm × 21.0 cm.
3. Adjust the confidence threshold if needed. Lower values can find less-certain detections but may add false positives.
4. Click **Detect & measure**.
5. Review the annotated image, calibration status, and per-object class, confidence, length, width, and area.

The supplied `backend\aruco_marker_id0.png` is a marker image for the configured `DICT_4X4_50` dictionary with marker ID 0. When printing it, measure the black square itself and set that physical measurement in the UI; do not include the white margin.

## How the application works

1. **Image input (browser):** the React UI accepts one uploaded image or captures a frame from the webcam. Webcam access is provided by the browser and normally works on `localhost` after the user grants permission.
2. **Request (browser → API):** the UI sends the image plus `calibration`, `marker_size_cm`, and `conf` fields as a multipart form POST to `http://127.0.0.1:8000/measure`.
3. **Image decoding and calibration (backend):** FastAPI validates the upload and OpenCV decodes it. `calibration.py` either leaves scale unset, detects an ArUco marker and calculates pixels per centimetre from its black-square corners, or looks for a quadrilateral with the proportions of A4 paper.
4. **Detection and segmentation (backend):** Ultralytics loads `models\best.pt` and predicts at the configured image size (640 pixels by default). The confidence value is limited to the valid 0.01–0.99 interval. The backend uses the model's instance masks and class/confidence outputs; objects without a usable polygon are skipped.
5. **Measurement (backend):** for each segmentation polygon, OpenCV fits a minimum-area rotated rectangle. Its longer side is reported as length, its shorter side as width, and the polygon contour area as area. If calibration was found, pixel measurements are converted to centimetres and square centimetres.
6. **Response and display:** the API returns detections, calibration status/scale, image dimensions, and a JPEG-encoded annotated image as a base64 data URL. The React UI renders the image and a results table.

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Reports service status, whether the configured model path exists, and supported classes. |
| `GET` | `/classes` | Returns the configured class names. |
| `POST` | `/measure` | Accepts multipart form fields `file`, `calibration` (`none`, `aruco`, or `a4`), `marker_size_cm` (default `5.0`), and `conf` (default `0.25`). Returns detections, calibration details, image dimensions, and the annotated image. |

## Model, classes, and tools

### Detection model

- `models\best.pt` is the trained Ultralytics YOLO **instance-segmentation** checkpoint used by the app. It predicts object masks as well as class and confidence values; this is why measurements can be derived from a visible object outline rather than only a rectangular detection box.
- The configured inference image size is `640` (`backend\config.json`); the UI's default confidence threshold is `0.25`.
- The 28 class names in `backend\config.json` are: `Soybean`, `apple`, `banana`, `broccoli`, `cabbage`, `capsicum`, `carrot`, `cauliflower`, `chilli`, `corn`, `cucumber`, `eggplant`, `garlic`, `ginger`, `grapes`, `lemon`, `mango`, `onion`, `orange`, `pear`, `peas`, `pineapple`, `potato`, `radish`, `strawberry`, `sweetpotato`, `tomato`, and `watermelon`.
- These weights are for those trained classes, not a general-purpose detector. Results depend on the training data, object visibility, lighting, image quality, and selected confidence threshold.

The optional Kaggle notebook documents the training/export workflow: it obtains a polygon-annotated dataset from the configured Roboflow project (or a Kaggle input directory), filters/remaps class labels, then tries `yolo11n-seg.pt` followed by `yolov8n-seg.pt` as a base checkpoint. Its training defaults are 50 epochs, image size 640, batch size 16, and patience 15, using a GPU when available or CPU otherwise. It validates segmentation metrics and exports `best.pt` plus configuration. The notebook tries the listed bases in order; `backend\config.json` does not record which one produced the already-exported `best.pt`, so the exact base checkpoint should not be inferred from that config alone.

**Training is optional and is not needed to run the web demo.** Re-running the notebook requires a Kaggle environment or an appropriately prepared local environment, access to the dataset, and (for Roboflow download) a Roboflow API key stored as a Kaggle Secret or environment variable named `ROBOFLOW_API_KEY`. Do not put that key in source code or commit it.

### Runtime components

| Component | Use |
| --- | --- |
| Python | Runs the backend and image processing. |
| FastAPI | Defines the local HTTP API and validates requests. |
| Uvicorn | Runs the FastAPI app at `127.0.0.1:8000`. |
| Ultralytics YOLO + PyTorch | Loads the checkpoint and performs instance-segmentation inference. |
| OpenCV contrib (`opencv-contrib-python`) | Decodes/draws images, measures polygon geometry, and provides `cv2.aruco`. |
| NumPy | Represents image and polygon arrays and calculates distances. |
| React + React DOM | Renders the browser interface. |
| Vite + `@vitejs/plugin-react` | Runs and builds the frontend. |
| `lucide-react` | Supplies the interface icons. |
| Browser camera APIs | Capture webcam frames when permission is granted. |

The runtime dependencies are declared in `backend\requirements.txt` and `frontend\package.json`. Do not install `opencv-python` alongside `opencv-contrib-python` in the same Python environment; both provide the `cv2` module and may conflict.

## Measurement and calibration notes

- Without a detectable reference, the app reports `length_px`, `width_px`, and `area_px`; centimetre fields are `null`.
- With calibration, the app divides pixel lengths by pixels-per-centimetre and pixel area by pixels-per-centimetre squared. Calibration status is returned with each request.
- ArUco scale is calculated from the first detected marker's average side in pixels and the configured black-square size. The printed marker must be from the configured dictionary and its entered size must be accurate.
- A4 detection is experimental. It uses image thresholding/contours and checks for the approximate A4 aspect ratio; plain backgrounds, cropping, shadows, or perspective may prevent detection.
- A single scale assumes the reference and measured object are approximately on the same plane and at a similar distance from the camera. Perspective, tilted surfaces, lens distortion, and objects at different depths can make centimetre results inaccurate.
- Length and width come from a rotated rectangle around the 2D segmentation polygon; area is the polygon's 2D contour area. These are visible-image estimates, **not 3D measurements**, and do not recover hidden parts of an object.
- Model predictions are not guaranteed to be correct. Check the mask, class, and calibration before relying on a measurement.

## Troubleshooting

- **`py -3.11` is not recognized:** install Python 3.11 with the Python launcher, reopen PowerShell, and retry. If only `python` is available, ensure it points to Python 3.11 before creating the virtual environment.
- **Python package installation fails:** verify Python is 64-bit and 3.11, confirm internet access, and retry the `pip install` command. PyTorch wheels may lag newly released Python versions; use Python 3.11 rather than the newest available Python if installation is unsupported.
- **`No module named cv2.aruco`:** confirm the backend virtual environment was installed from `backend\requirements.txt`. Do not add `opencv-python` to that environment.
- **Backend says the model file is missing:** restore `models\best.pt` in the project's `models` folder; the app does not download the checkpoint automatically.
- **`uvicorn` or the backend command is not found:** use the full virtual-environment command shown above, from the `backend` folder.
- **The page cannot reach the backend:** make sure the backend terminal is still running at `http://127.0.0.1:8000`, then reload the Vite page.
- **No objects detected:** test a clear, well-lit image containing a supported class. Lower the confidence threshold and check that the object is visible and not heavily occluded.
- **Centimetres are unavailable:** the selected reference was not detected. Use pixel-only mode or include a clear, correctly sized marker/A4 sheet.
- **Camera access fails:** grant camera permission to the local browser page, close other applications using the camera, or upload an image instead.
- **Port 8000 or 5173 is already in use:** stop the other process using that port, then restart the appropriate service. The frontend currently calls the backend at port 8000.
- **Fonts look different offline:** the stylesheet requests DM Sans and Manrope from Google Fonts; the interface remains usable with local fallback fonts if the font download is unavailable.
