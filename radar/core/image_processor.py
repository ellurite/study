import cv2
import numpy as np
import math
from typing import Optional


class ImageProcessor:

    @staticmethod
    def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
        noisy = img.astype(np.float32)
        gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy += gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def add_salt_pepper(img: np.ndarray, amount: float) -> np.ndarray:
        out = img.copy()
        h, w = img.shape[:2]
        num = int(amount * h * w)
        ys = np.random.randint(0, h, num)
        xs = np.random.randint(0, w, num)
        out[ys, xs] = 255
        ys = np.random.randint(0, h, num)
        xs = np.random.randint(0, w, num)
        out[ys, xs] = 0
        return out

    @staticmethod
    def cartesian_to_polar(img: np.ndarray, center: Optional[tuple] = None,
                           radius: Optional[float] = None, size: tuple = (512, 512)) -> np.ndarray:
        if img is None:
            return None
        h, w = img.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = min(center[0], center[1])
        return cv2.warpPolar(img, size, center, radius, flags=cv2.WARP_POLAR_LINEAR)

    @staticmethod
    def polar_to_cartesian(polar_img: np.ndarray, out_size: tuple = (512, 512),
                           center: Optional[tuple] = None, max_radius: Optional[float] = None) -> np.ndarray:
        if polar_img is None:
            return None
        h_out, w_out = out_size
        if center is None:
            center = (w_out // 2, h_out // 2)
        if max_radius is None:
            max_radius = min(center[0], center[1])
        return cv2.warpPolar(polar_img, (w_out, h_out), center, max_radius,
                             flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)

    @staticmethod
    def calc_entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        s = hist.sum()
        if s == 0:
            return 0.0
        p = hist / s
        return -float(np.sum([pp * math.log2(pp) for pp in p if pp > 0]))

    @staticmethod
    def make_circular_alpha(bgr_img: np.ndarray, diameter: Optional[float] = None) -> np.ndarray:
        if bgr_img is None:
            return None
        h, w = bgr_img.shape[:2]
        if diameter is None:
            diameter = min(h, w)
        cx, cy = w // 2, h // 2
        radius = diameter // 2

        if bgr_img.shape[2] == 4:
            bgra = bgr_img.copy()
        else:
            bgra = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        bgra[..., 3] = mask
        return bgra