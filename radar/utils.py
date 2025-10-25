import cv2
import numpy as np
from PyQt5 import QtGui, QtCore


def cv_to_qpixmap(cv_img: np.ndarray) -> QtGui.QPixmap:
    """Преобразует изображения OpenCV в QPixmap"""
    if cv_img is None:
        return None
    h, w = cv_img.shape[:2]
    if cv_img.ndim == 2:
        bytes_per_line = w
        qimg = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    elif cv_img.shape[2] == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    elif cv_img.shape[2] == 4:
        rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        bytes_per_line = 4 * w
        qimg = QtGui.QImage(rgba.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    else:
        return None
    return QtGui.QPixmap.fromImage(qimg)


def draw_pixel_plane(w: int = 60, h: int = 60) -> np.ndarray:
    """Пиксельный силуэт самолета"""
    plane = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.rectangle(plane, (cx - 2, cy - 12), (cx + 2, cy + 12), (0, 0, 0, 255), -1)
    pts = np.array([[cx - 15, cy], [cx + 15, cy], [cx, cy + 4]])
    cv2.fillPoly(plane, [pts], (0, 0, 0, 255))
    cv2.line(plane, (cx, cy - 12), (cx - 8, cy - 18), (0, 0, 0, 255), 3)
    return plane


def overlay_rgba(base_bgr: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Наложение RGBA изображения на BGR"""
    if base_bgr is None or overlay_rgba is None:
        return base_bgr
    bh, bw = base_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    if x >= bw or y >= bh:
        return base_bgr
    w = min(ow, bw - x)
    h = min(oh, bh - y)
    if w <= 0 or h <= 0:
        return base_bgr

    base = base_bgr.copy().astype(np.float32) / 255.0
    overlay = overlay_rgba[:h, :w].astype(np.float32) / 255.0
    alpha = overlay[..., 3:4]
    overlay_rgb = overlay[..., :3]
    base_region = base[y:y + h, x:x + w]
    comp = (1.0 - alpha) * base_region + alpha * overlay_rgb
    base[y:y + h, x:x + w] = comp
    return (base * 255).astype(np.uint8)