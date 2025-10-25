import cv2
import numpy as np
import torch
import random
from typing import List, Optional

from radar.domain import ImageData, DetectionResult
from radar.core.radar_generator import RadarImageGenerator
from radar.core.image_processor import ImageProcessor
from radar.utils import draw_pixel_plane, overlay_rgba

class RadarService:

    def __init__(self):
        self.radar_generator = RadarImageGenerator()
        self.image_processor = ImageProcessor()
        self.model = None
        self.image_data = ImageData()

    def load_image(self, file_path: str) -> bool:
        """Загрузка изображения из файла"""
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False

        max_side = 800
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if img.ndim == 3 and img.shape[2] == 4:
            self.image_data.base_image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            self.image_data.base_image = img.copy()

        self.image_data.processed_image = self.image_data.base_image.copy()
        self.image_data.overlay_image = None
        self.image_data.clean_generated = None
        self.image_data.is_generated = False
        self.image_data.has_noise = False
        return True

    def generate_simple_image(self, show_planes: bool, show_houses: bool, show_cars: bool):
        """Генерация простого тестового изображения"""
        polar_h = polar_w = 512
        polar_canvas = np.ones((polar_h, polar_w, 3), dtype=np.uint8) * 255

        if show_planes:
            plane_icon = draw_pixel_plane(60, 60)
            for _ in range(random.randint(1, 4)):
                ang = random.randint(0, polar_w - plane_icon.shape[1])
                rpos = random.randint(20, polar_h - plane_icon.shape[0] - 1)
                polar_canvas = overlay_rgba(polar_canvas, plane_icon, ang, rpos)

        if show_houses:
            for _ in range(random.randint(1, 4)):
                ang = random.randint(0, polar_w - 40)
                rpos = random.randint(30, polar_h - 40)
                cv2.rectangle(polar_canvas, (ang, rpos), (ang + 30, rpos + 30), (0, 0, 0), -1)

        if show_cars:
            for _ in range(random.randint(1, 4)):
                ang = random.randint(0, polar_w - 40)
                rpos = random.randint(20, polar_h - 30)
                cv2.rectangle(polar_canvas, (ang, rpos + 6), (ang + 40, rpos + 18), (0, 0, 0), -1)
                cv2.circle(polar_canvas, (ang + 10, rpos + 22), 5, (0, 0, 0), -1)
                cv2.circle(polar_canvas, (ang + 30, rpos + 22), 5, (0, 0, 0), -1)

        polar_rgba = self.image_processor.make_circular_alpha(polar_canvas, min(polar_h, polar_w))
        self.image_data.polar_image = polar_rgba

        out_size = (512, 512)
        center = (out_size[1] // 2, out_size[0] // 2)
        max_radius = min(center[0], center[1])
        cart = self.image_processor.polar_to_cartesian(polar_canvas, out_size, center, max_radius)

        self.image_data.base_image = cart
        self.image_data.processed_image = cart
        self.image_data.clean_generated = cart
        self.image_data.is_generated = True
        self.image_data.has_noise = False

    def generate_radar_image(self):
        """Генерация реалистичного РЛИ"""
        try:
            image_normalized, massiv_x, massiv_y = self.radar_generator.generate_radar_image(n_rld=1)
            radar_img = (image_normalized[:, :, 0] * 255).astype(np.uint8)
            radar_img_color = cv2.applyColorMap(radar_img, cv2.COLORMAP_JET)

            size = max(radar_img_color.shape[:2])
            square = np.ones((size, size, 3), dtype=np.uint8) * 0
            h, w = radar_img_color.shape[:2]
            y0 = (size - h) // 2
            x0 = (size - w) // 2
            square[y0:y0 + h, x0:x0 + w] = radar_img_color

            polar_rgba = self.image_processor.make_circular_alpha(square, size)
            scale = min(1.0, 512 / size)
            if scale != 1.0:
                new_size = (int(size * scale), int(size * scale))
                polar_rgba = cv2.resize(polar_rgba, new_size, interpolation=cv2.INTER_AREA)

            self.image_data.polar_image = polar_rgba
            radar_img_color_resized = cv2.resize(radar_img_color, (512, 512), interpolation=cv2.INTER_AREA)

            self.image_data.base_image = radar_img_color_resized
            self.image_data.processed_image = radar_img_color_resized
            self.image_data.clean_generated = radar_img_color_resized
            self.image_data.is_generated = True
            self.image_data.has_noise = False
            return True

        except Exception as e:
            print(f"Ошибка генерации РЛИ: {e}")
            return False

    def apply_noise(self, noise_type: str, intensity: float):
        """Применение шума к изображению"""
        if self.image_data.processed_image is None:
            return False

        img = self.image_data.processed_image.copy()
        if noise_type == "Гауссов":
            sigma = intensity * 0.6
            img = self.image_processor.add_gaussian_noise(img, sigma)
        elif noise_type == "Соль-и-перец":
            amount = intensity / 100.0 * 0.18
            img = self.image_processor.add_salt_pepper(img, amount)
        else:
            return False

        self.image_data.processed_image = img
        if self.image_data.is_generated:
            self.image_data.has_noise = True
        return True

    def remove_noise(self):
        """Удаление шума (восстановление чистого изображения)"""
        if not self.image_data.is_generated or self.image_data.clean_generated is None:
            return False

        self.image_data.processed_image = self.image_data.clean_generated.copy()
        self.image_data.has_noise = False
        return True

    def detect_objects(self, confidence_threshold: float, target_classes: List[str]) -> List[DetectionResult]:
        """Детектирование объектов с помощью YOLO"""
        if self.model is None:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            except Exception as e:
                raise Exception(f"Не удалось загрузить модель YOLO: {e}")

        self.model.conf = confidence_threshold

        results = self.model(self.image_data.processed_image)
        df = results.pandas().xyxy[0]

        detections = []
        for _, row in df.iterrows():
            cls_name = row['name']
            if not target_classes or cls_name in target_classes:
                detections.append(DetectionResult(
                    bbox=(int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])),
                    class_name=cls_name,
                    confidence=float(row['confidence'])
                ))

        return detections

    def get_image_stats(self) -> dict:
        """Получение статистики изображения"""
        if self.image_data.processed_image is None:
            return {}

        gray = cv2.cvtColor(self.image_data.processed_image, cv2.COLOR_BGR2GRAY)
        h, w = self.image_data.processed_image.shape[:2]

        return {
            'width': w,
            'height': h,
            'min_brightness': int(gray.min()),
            'max_brightness': int(gray.max()),
            'mean_brightness': float(gray.mean()),
            'std_brightness': float(gray.std()),
            'entropy': self.image_processor.calc_entropy(gray),
            'snr': float(gray.mean() / (gray.std() + 1e-6)),
            'is_generated': self.image_data.is_generated,
            'has_noise': self.image_data.has_noise
        }