import cv2
import numpy as np


class ObjectDetector:
    def __init__(self, model_name=None):
        print("Инициализация детектора объектов...")
        self.model = None

    def detect_objects(self, image, selected_classes):
        """Обнаружение объектов на изображении"""
        print(f"Поиск объектов: {selected_classes}")

        annotated_image = image.copy()
        detected_objects = []

        # Простая логика для демонстрации
        height, width = image.shape[:2]

        # Ищем яркие области (объекты на черном фоне)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Фильтр по размеру
                x, y, w, h = cv2.boundingRect(contour)

                # Определяем тип объекта по форме
                aspect_ratio = w / h
                if aspect_ratio > 1.5:
                    obj_class = "самолет"
                else:
                    obj_class = "дом"

                if obj_class in selected_classes:
                    detected_objects.append({
                        'class': obj_class,
                        'confidence': 0.7 + i * 0.1,
                        'bbox': [x, y, x + w, y + h]
                    })

                    # Рисование bounding box
                    color = (0, 255, 0)  # Зеленый
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(annotated_image,
                                f'{obj_class} {0.7 + i * 0.1:.2f}',
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"Найдено объектов: {len(detected_objects)}")
        return {
            'objects': detected_objects,
            'annotated_image': annotated_image
        }