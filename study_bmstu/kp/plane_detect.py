from ultralytics import YOLO

image_path = ('C:/Users\superpro2005\Desktop\study\python\photo\plane_3.png')
model = YOLO('yolov8m.pt')

results = model.predict(image_path, conf=0.25, save=True)

for i, box in enumerate(results[0].boxes):
    cls_name = model.names[int(box.cls)] if hasattr(model, 'names') else 'object'
    print(f"{cls_name} {i+1}: confidence = {box.conf.item():.3f}")

results[0].show()