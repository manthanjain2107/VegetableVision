from pathlib import Path
import base64, json
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from measurement import measure_polygon, draw_measurement
from calibration import get_pixels_per_cm

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

app = FastAPI(title="VegetableVision Measurement API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
model = YOLO(str(MODEL_PATH))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "classes": CONFIG.get("classes", [])}

@app.get("/classes")
def classes():
    return {"classes": CONFIG.get("classes", [])}

@app.post("/measure")
async def measure(file: UploadFile = File(...), calibration: str = Form("none"), marker_size_cm: float = Form(5.0), conf: float = Form(0.25)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image.")
    if calibration not in {"none", "aruco", "a4"}:
        raise HTTPException(status_code=400, detail="calibration must be none, aruco, or a4")
    ppcm, calibration_message = get_pixels_per_cm(image, method=calibration, marker_size_cm=marker_size_cm, dict_name=CONFIG.get("aruco_dict", "DICT_4X4_50"))
    results = model.predict(source=image, conf=max(0.01, min(float(conf), 0.99)), imgsz=int(CONFIG.get("imgsz", 640)), verbose=False)
    annotated = image.copy()
    detections = []
    result = results[0]
    if result.masks is not None and result.boxes is not None:
        polygons = result.masks.xy
        for i, poly in enumerate(polygons):
            if len(poly) < 3: continue
            cls_id = int(result.boxes.cls[i].item())
            score = float(result.boxes.conf[i].item())
            class_names = CONFIG.get("classes", [])
            name = class_names[cls_id] if 0 <= cls_id < len(class_names) else result.names.get(cls_id, str(cls_id))
            metrics = measure_polygon(poly, pixels_per_cm=ppcm)
            if metrics is None: continue
            annotated = draw_measurement(annotated, poly, metrics, label=f"{name} {score:.2f}")
            detections.append({"class_name": name, "confidence": round(score, 4), **metrics})
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok: raise HTTPException(status_code=500, detail="Failed to encode output image.")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {"detections": detections, "calibration": {"method": calibration, "pixels_per_cm": round(ppcm, 4) if ppcm else None, "message": calibration_message}, "annotated_image": "data:image/jpeg;base64," + image_b64, "image_width": int(image.shape[1]), "image_height": int(image.shape[0])}
