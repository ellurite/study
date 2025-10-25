import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
import os

# Исправление для PyTorch 2.6
torch.serialization.add_safe_globals([DetectionModel])

# Проверяем пути

image_path = ('C:/Users/superpro2005/Desktop/study/python/photo/plane_3.png'
              '197'
              '')
model = YOLO('yolov8m.pt')
if not os.path.exists(image_path):
    print("Изображение не найдено!")
    exit()


results = model.predict(image_path, conf=0.25, save=True)

# Результаты
print(f"Найдено объектов: {len(results[0].boxes)}")
for i, box in enumerate(results[0].boxes):
    cls_name = model.names[int(box.cls)] if hasattr(model, 'names') else 'object'
    print(f"{cls_name} {i+1}: confidence = {box.conf.item():.3f}")

# Показать результат
results[0].show()