import cv2
import numpy as np

A4_LONG_CM = 29.7
A4_SHORT_CM = 21.0


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def generate_aruco_marker(marker_id=0, size_px=600, dict_name="DICT_4X4_50", border_px=100):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, marker_id, size_px)
    else:
        marker = aruco.drawMarker(dictionary, marker_id, size_px)
    return cv2.copyMakeBorder(marker, border_px, border_px, border_px, border_px,
                              cv2.BORDER_CONSTANT, value=255)


def detect_aruco(image_bgr, marker_size_cm, dict_name="DICT_4X4_50"):
    aruco = cv2.aruco
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary,
                                              parameters=aruco.DetectorParameters_create())
    if ids is None or len(corners) == 0:
        return None
    c = corners[0].reshape(4, 2)
    sides = [_dist(c[i], c[(i + 1) % 4]) for i in range(4)]
    return {
        "pixels_per_cm": float(np.mean(sides)) / float(marker_size_cm),
        "corners": c.tolist(),
        "method": "aruco",
    }


def detect_a4(image_bgr, min_area_fraction=0.10, ratio_tolerance=0.20):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]

    quad = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area_fraction * img_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        return None

    tl, tr, br, bl = order_points(quad)
    side_a = (_dist(tl, tr) + _dist(bl, br)) / 2
    side_b = (_dist(tl, bl) + _dist(tr, br)) / 2
    long_px, short_px = max(side_a, side_b), min(side_a, side_b)
    if abs(long_px / short_px - A4_LONG_CM / A4_SHORT_CM) > ratio_tolerance:
        return None
    return {
        "pixels_per_cm": (long_px / A4_LONG_CM + short_px / A4_SHORT_CM) / 2,
        "corners": [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()],
        "method": "a4",
    }


def get_pixels_per_cm(image_bgr, method="a4", marker_size_cm=5.0, dict_name="DICT_4X4_50"):
    if method == "aruco":
        found = detect_aruco(image_bgr, marker_size_cm, dict_name)
    elif method == "a4":
        found = detect_a4(image_bgr)
    else:
        return None, "no calibration requested"
    if found is None:
        return None, f"{method} reference not detected"
    return found["pixels_per_cm"], f"{found['method']} reference detected"
