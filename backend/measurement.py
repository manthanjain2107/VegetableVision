import cv2
import numpy as np


def measure_polygon(polygon_xy, pixels_per_cm=None):
    pts = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    area_px = float(cv2.contourArea(pts))
    out = {
        "length_px": round(length_px, 2),
        "width_px": round(width_px, 2),
        "area_px": round(area_px, 2),
        "angle_degrees": round(float(angle), 2),
        "box_points": box.tolist(),
        "length_cm": None,
        "width_cm": None,
        "area_cm2": None,
    }
    if pixels_per_cm and pixels_per_cm > 0:
        out["length_cm"] = round(length_px / pixels_per_cm, 2)
        out["width_cm"] = round(width_px / pixels_per_cm, 2)
        out["area_cm2"] = round(area_px / (pixels_per_cm ** 2), 2)
    return out


def measure_mask(mask, pixels_per_cm=None):
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return measure_polygon(contour.reshape(-1, 2), pixels_per_cm)


def draw_measurement(image_bgr, polygon_xy, result, label=""):
    img = image_bgr.copy()
    pts = np.asarray(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    box = np.asarray(result["box_points"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    if result["length_cm"] is not None:
        text = f"{label} L={result['length_cm']}cm W={result['width_cm']}cm"
    else:
        text = f"{label} L={result['length_px']}px W={result['width_px']}px"
    x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
    cv2.putText(img, text.strip(), (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img
