import cv2
import numpy as np


class ImageProcessor:
    def polar_to_cartesian(self, image):
        """Преобразование из полярных в декартовы координаты"""
        if image is None:
            return None

        # Если изображение уже в декартовых координатах, возвращаем как есть
        # В реальном приложении здесь будет преобразование
        return image

    def analyze_properties(self, image):
        """Анализ параметров изображения"""
        if image is None:
            return {}

        # Конвертация в grayscale для некоторых метрик
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        properties = {
            'Размер': f"{image.shape[1]}x{image.shape[0]}",
            'Яркость': f"{np.mean(gray):.2f}",
            'Контрастность': f"{np.std(gray):.2f}",
            'Каналы': image.shape[2] if len(image.shape) > 2 else 1,
        }

        return properties